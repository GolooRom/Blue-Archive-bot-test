"""Export Blue Archive character models from the game's asset bundles to .glb.

Reads the Unity asset bundles shipped with the Steam build and writes one
self-contained glTF binary per character: skinned meshes with embedded
textures, the bone hierarchy, and every animation clip found alongside them.
Each file is named for the wiki's name for the character rather than the
bundle's dev name, through the wiki repo's devname_map tables and the game's
costume table.

Usage:
    python export_models.py --list
    python export_models.py airi_original aris_original
    python export_models.py "Aru (New Year)"
    python export_models.py --all
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import UnityPy
from UnityPy.enums import ArchiveFlags, CompressionFlags
from UnityPy.environment import simplify_name
from UnityPy.helpers import CompressionHelper
from UnityPy.helpers.MeshHelper import MeshHandler

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE_DIR = (
    HERE.parent / "vendor" / "BlueArchive" / "BlueArchive_Data" / "StreamingAssets"
    / "PUB" / "Resource" / "GameData" / "Windows"
)

CHARACTER_BUNDLE_RE = re.compile(r"^assets-_mx-characters-(.+?)-_mxdependency-")
MODEL_ASSET_RE = re.compile(r"^assets/_mx/characters/[^/]+/model/", re.I)

# The wiki's devname_map tables, kept current by update.py, give the wiki's
# name for each bundle's dev character code.
DEVNAME_MAPS = (HERE.parent / "json" / "devname_map.json",
                HERE.parent / "json" / "devname_map_aux.json")
# Some bundles are named for a model prefab code the maps lack (`ch0061`); the
# costume table ties each prefab to costume dev names the maps do know.
COSTUME_TABLE = HERE.parent / "json" / "CostumeExcelTable.json"
QUALIFIERS = {"cutin": "Cut-in", "nonweapon": "No Weapon", "scenario": "Scenario",
              "carrier": "Carrier"}

# Unity is left-handed, glTF right-handed: mirror X, and with it the triangle
# winding, the quaternions (x, -y, -z, w) and the matrices (FLIP @ M @ FLIP).
UNITY_TO_GLTF = np.array([-1.0, 1.0, 1.0])
QUAT_TO_GLTF = np.array([1.0, -1.0, -1.0, 1.0])
FLIP = np.diag([-1.0, 1.0, 1.0, 1.0])

# The clip the viewer opens on. It decides the rest state of switched parts
# and which props count as part of the character; without it, mark_props()
# uses the first clip in bundle order. Keep in sync with data-anim-default in
# viewer.html and Module:ModelViewer.
DEFAULT_CLIP = "Cafe_Reaction"

# A few characters sit in the cafe as a separate body, `<name>_CafeOnly_Mesh`,
# whose clips are keyed to its own rest frames; it is exported as a model of
# its own, `<label> (Cafe).glb`.
CAFE_RIG = "_cafeonly"
CAFE_LABEL = "Cafe"

# The viewer lists clips in file order: the shared, useful clips first, the
# rest alphabetical.
ANIMATION_PREFIX_ORDER = (
    "cafe_reaction",
    "cafe_idle",
    "cafe_walk",
    "formation_idle",
    "formation_pickup",
    "exs_cutin",
    "exs",
    "tactical",
    "victory",
    "normal_idle",
    "normal",
    "kneel_idle",
    "kneel",
    "stand_idle",
    "stand",
    "move",
    "vital",
    "public",
)
ANIMATION_VARIANT_PREFIX_RE = re.compile(r"^([a-z]+)\d+(?=_|$)")

# The halo's fallback attachment when the runtime prefab gives no follower.
HEAD_BONE = "Bip001 Head"

# Mouth expressions are events, not curves: SetMouthTile(row * 100 + col), the
# row counted from the bottom of the atlas (Unity's UV origin).
MOUTH_EVENT = "SetMouthTile"
MOUTH_DEFAULT_EVENT = "SetMouthTileToDefault"

# A spare face is a mesh of its own, switched by the clips, and recognised by
# its materials rather than its name.  The everyday face holds the mouth the
# clips lip-sync.
FACE_MATERIAL = "_face"
MOUTH_MATERIAL = "_eyemouth"
BODY_MATERIALS = ("_body", "_hair")
RENDERER_EVENTS = {"AniEvt_EnableChildRenderer": True,
                   "AniEvt_DisableChildRenderer": False}

TRANSFORM_CLASS_ID = 4
GAMEOBJECT_CLASS_ID = 1
BIND_ACTIVE = 2086281974  # zlib.crc32(b"m_IsActive"), as Mecanim hashes it
RENDERER_CLASS_IDS = (25, 23, 137)  # Renderer, MeshRenderer, SkinnedMeshRenderer
BIND_ENABLED = 3305885265  # zlib.crc32(b"m_Enabled")
# Transform curve bindings, and how many float curves each one spans.  A
# rotation is keyed either as a quaternion or as Euler angles in degrees.
BIND_POSITION, BIND_ROTATION, BIND_SCALE, BIND_EULER = 1, 2, 3, 4
BINDING_SIZE = {BIND_POSITION: 3, BIND_ROTATION: 4, BIND_SCALE: 3, BIND_EULER: 3}
GLTF_PATH = {BIND_POSITION: "translation", BIND_ROTATION: "rotation", BIND_SCALE: "scale"}
# An Euler binding's customType is Unity's RotationOrder: the axes in the
# order they are applied, each about the parent's axes.  Curves imported from
# the artists' FBX keep its XYZ; Unity's own default is ZXY.
EULER_ORDERS = ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX"]
# Euler curves are interpolated per angle, quaternion keys along the arc
# between them; subdividing keeps the two within a few degrees of each other.
EULER_STEP = 10.0


# --------------------------------------------------------------------------
# cross-bundle references
# --------------------------------------------------------------------------

def bundle_cabs(path: Path) -> list[str]:
    """The names of the serialized files a bundle holds, read from its header
    metadata rather than by decompressing the whole file."""
    with path.open("rb") as handle:
        head = handle.read(4096)
        end = head.index(b"\0")
        if head[:end] != b"UnityFS":
            return []
        offset = end + 1
        version = struct.unpack_from(">I", head, offset)[0]
        offset += 4
        for _ in range(2):  # the player and engine version strings
            offset = head.index(b"\0", offset) + 1
        _size, compressed, uncompressed, flags = struct.unpack_from(">qIII", head, offset)
        offset += 20
        if version >= 7:
            offset = (offset + 15) & ~15
        if flags & ArchiveFlags.BlocksInfoAtTheEnd:
            handle.seek(-compressed, os.SEEK_END)
            blob = handle.read(compressed)
        else:
            if offset + compressed > len(head):
                handle.seek(0)
                head = handle.read(offset + compressed)
            blob = head[offset:offset + compressed]
    compression = CompressionFlags(flags & ArchiveFlags.CompressionTypeMask)
    blob = CompressionHelper.DECOMPRESSION_MAP[compression](blob, uncompressed)

    offset = 16  # the hash of the uncompressed data
    blocks = struct.unpack_from(">i", blob, offset)[0]
    offset += 4 + blocks * 10  # (uncompressed size, compressed size, flags) each
    nodes = struct.unpack_from(">i", blob, offset)[0]
    offset += 4
    names = []
    for _ in range(nodes):
        offset += 20  # (offset, size, flags)
        end = blob.index(b"\0", offset)
        names.append(blob[offset:end].decode("utf-8", "replace"))
        offset = end + 1
    return names


_cab_index: dict[str, str] | None = None


def cab_index(bundle_dir: Path) -> dict[str, str]:
    """Which bundle holds which serialized file, built once per process."""
    global _cab_index
    if _cab_index is None:
        _cab_index = {}
        preload = bundle_dir.parent.parent / "Preload" / bundle_dir.name
        for directory in (bundle_dir, preload):
            for bundle in sorted(directory.glob("*.bundle")):
                try:
                    names = bundle_cabs(bundle)
                except Exception as e:
                    print(f"  skipped a bundle header: {bundle.name}: {type(e).__name__}: {e}")
                    continue
                for name in names:
                    _cab_index.setdefault(simplify_name(name), str(bundle))
    return _cab_index


def resolve_dependencies(env, bundle_dir: Path) -> None:
    """Let pointers follow into bundles beyond the character's own.

    Bundles name their dependencies by serialized file rather than by bundle,
    so the index says which bundle to open the moment a pointer needs it.  They
    are loaded as dependencies, which keeps their own prefabs, clips and
    avatars out of `env.objects`.
    """
    def find_file(name: str, is_dependency: bool = True):
        cab = env.get_cab(name)
        if cab is not None:
            return cab
        bundle = cab_index(bundle_dir).get(simplify_name(name))
        if bundle is None:
            raise FileNotFoundError(f"{name} is in no bundle under {bundle_dir}")
        env.load_file(bundle, is_dependency=True)
        cab = env.get_cab(name)
        if cab is None:
            raise FileNotFoundError(f"{name} not found in {Path(bundle).name}")
        return cab

    env.find_file = find_file


# --------------------------------------------------------------------------
# scene graph
# --------------------------------------------------------------------------

def obj_key(obj) -> tuple[str, int]:
    reader = obj.object_reader
    return (reader.assets_file.name, reader.path_id)


class Scene:
    """The Transform hierarchy of one character, mirrored into glTF nodes."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.roots: list[int] = []
        self._index: dict[tuple[str, int], int] = {}

    def add(self, node: dict) -> int:
        """Add a root node of our own making, for a mesh with no transform."""
        self.nodes.append(node)
        self.roots.append(len(self.nodes) - 1)
        return len(self.nodes) - 1

    def find(self, transform) -> int | None:
        """Index of this transform's node, or None if nothing created one."""
        return self.find_key(obj_key(transform))

    def find_key(self, key: tuple | None) -> int | None:
        return self._index.get(key)

    def parents(self) -> dict[int, int]:
        return {child: i for i, node in enumerate(self.nodes)
                for child in node.get("children", ())}

    def world(self, index: int, parents: dict[int, int]) -> np.ndarray:
        """The node's rest transform in scene space."""
        matrix = np.eye(4)
        while index is not None:
            matrix = node_matrix(self.nodes[index]) @ matrix
            index = parents.get(index)
        return matrix

    def reparent(self, index: int, parent: int) -> None:
        """Move a node under another, keeping the place it rests in."""
        parents = self.parents()
        local = np.linalg.inv(self.world(parent, parents)) @ self.world(index, parents)
        node = self.nodes[index]
        node["translation"], node["rotation"], node["scale"] = decompose(local)
        if index in self.roots:
            self.roots.remove(index)
        elif index in parents:
            self.nodes[parents[index]]["children"].remove(index)
        self.nodes[parent].setdefault("children", []).append(index)

    def node(self, transform) -> int:
        """Index of the glTF node for this transform, creating it if needed."""
        key = obj_key(transform)
        if key in self._index:
            return self._index[key]
        t, r, s = transform.m_LocalPosition, transform.m_LocalRotation, transform.m_LocalScale
        self._index[key] = index = len(self.nodes)
        self.nodes.append({
            "name": transform.m_GameObject.read().m_Name,
            "translation": [-t.x, t.y, t.z],
            "rotation": [r.x, -r.y, -r.z, r.w],
            "scale": [s.x, s.y, s.z],
        })
        father = transform.m_Father
        if father is not None and father.path_id:
            self.nodes[self.node(father.read())].setdefault("children", []).append(index)
        else:
            self.roots.append(index)
        return index


def quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def node_matrix(node: dict) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(node["rotation"]) * np.array(node["scale"])
    matrix[:3, 3] = node["translation"]
    return matrix


def decompose(matrix: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """A 4x4 back into the translation, rotation and scale a glTF node holds."""
    basis = matrix[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:  # a mirrored basis: the flip goes on one axis
        scale[0] = -scale[0]
    rotation = basis / np.where(scale != 0, scale, 1.0)

    # Shepperd's method: build off the largest term for precision.
    trace = rotation.trace()
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = [(rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        q = [0.0, 0.0, 0.0, (rotation[k, j] - rotation[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (rotation[j, i] + rotation[i, j]) / s, \
            (rotation[k, i] + rotation[i, k]) / s
    q = np.array(q)
    return matrix[:3, 3].tolist(), (q / np.linalg.norm(q)).tolist(), scale.tolist()


def bindpose_matrix(bp) -> np.ndarray:
    return np.array([
        [bp.e00, bp.e01, bp.e02, bp.e03],
        [bp.e10, bp.e11, bp.e12, bp.e13],
        [bp.e20, bp.e21, bp.e22, bp.e23],
        [bp.e30, bp.e31, bp.e32, bp.e33],
    ])


# --------------------------------------------------------------------------
# animation clips
# --------------------------------------------------------------------------

def streamed_frames(streamed):
    """Decode m_StreamedClip into (time, [(curve index, coefficients), ...]) frames.

    A flat uint32 array: per frame, a float time and a key count, then one
    (int index, float coeff[4]) record per key.  The coefficients (a, b, c, d)
    are the cubic ((a*x + b)*x + c)*x + d, x being the time since the key,
    which the curve follows until its next key; d is the key's own value.  The
    first and last frames are +/- FLT_MAX helpers and drop out here.
    """
    raw = np.array(streamed.data, dtype=np.uint32).tobytes()
    offset = 0
    while offset + 8 <= len(raw):
        time, count = struct.unpack_from("<fi", raw, offset)
        offset += 8
        keys = []
        for _ in range(count):
            index, *coefficients = struct.unpack_from("<i4f", raw, offset)
            offset += 20
            keys.append((index, coefficients))
        if abs(time) < 1e30:
            yield time, keys


def clip_splines(clip) -> tuple[dict[int, np.ndarray], float]:
    """{curve index: rows of (time, a, b, c, d)} for one AnimationClip, and its
    stop time.  Each row is a cubic segment as streamed_frames() describes,
    lasting until the next row.

    Mecanim packs the curves into three concatenated arrays -- sparse
    (streamed), evenly sampled (dense) and constant -- in that order.  Dense
    curves are interpolated linearly, so their segments are straight lines.
    """
    inner = clip.m_MuscleClip.m_Clip.data
    stop = float(clip.m_MuscleClip.m_StopTime)

    rows: dict[int, list] = defaultdict(list)
    for time, frame in streamed_frames(inner.m_StreamedClip):
        for index, coefficients in frame:
            rows[index].append((time, *coefficients))
    splines = {index: np.array(sorted(r, key=lambda row: row[0])) for index, r in rows.items()}

    dense, base = inner.m_DenseClip, inner.m_StreamedClip.curveCount
    if dense.m_CurveCount:
        samples = np.array(dense.m_SampleArray, dtype=np.float32).astype(np.float64)
        samples = samples.reshape(-1, dense.m_CurveCount)
        times = dense.m_BeginTime + np.arange(len(samples)) / dense.m_SampleRate
        times = times.astype(np.float32).astype(np.float64)  # as streamed times are
        slopes = np.zeros_like(samples)
        slopes[:-1] = np.diff(samples, axis=0) * dense.m_SampleRate
        zeros = np.zeros(len(samples))
        for j in range(dense.m_CurveCount):
            splines[base + j] = np.column_stack([times, zeros, zeros, slopes[:, j], samples[:, j]])

    constant = base + dense.m_CurveCount
    for j, value in enumerate(inner.m_ConstantClip.data):
        splines[constant + j] = np.array([[0.0, 0, 0, 0, value], [stop, 0, 0, 0, value]])
    return splines, stop


def clip_keys(clip) -> tuple[dict[int, list[tuple[float, float]]], float]:
    """{curve index: [(time, value)]} for one AnimationClip, and its stop time."""
    splines, stop = clip_splines(clip)
    return {index: [(float(row[0]), float(row[4])) for row in rows]
            for index, rows in splines.items()}, stop


def binding_runs(clip):
    """(binding, first curve index, curves it spans), in the packed order.

    Walking m_ClipBindingConstant with a running offset says which curve is
    which.
    """
    offset = 0
    for binding in clip.m_ClipBindingConstant.genericBindings:
        size = BINDING_SIZE.get(binding.attribute, 1) if binding.typeID == TRANSFORM_CLASS_ID else 1
        yield binding, offset, size
        offset += size


def clip_curves(clip) -> dict[tuple[int, int], tuple[str, np.ndarray, np.ndarray]]:
    """{(transform path hash, binding): (interpolation, times, values)} for one
    AnimationClip, in glTF sampler terms."""
    splines, stop = clip_splines(clip)
    curves = {}
    for binding, start, size in binding_runs(clip):
        if binding.typeID != TRANSFORM_CLASS_ID or binding.attribute not in BINDING_SIZE:
            continue
        components = [splines.get(start + c, HELD_ZERO) for c in range(size)]
        times = np.array(sorted({0.0, stop} | {float(t) for c in components
                                                for t in c[:, 0] if 0.0 <= t <= stop}))
        if binding.attribute == BIND_EULER:
            times = subdivided(times, components)
            degrees = np.column_stack([spline_at(c, times)[0] for c in components])
            curves[(binding.path, BIND_ROTATION)] = (
                "LINEAR", times.astype(np.float32),
                euler_quaternions(degrees, EULER_ORDERS[binding.customType]))
            continue
        curves[(binding.path, binding.attribute)] = hermite_keys(components, times, stop)
    return curves


HELD_ZERO = np.zeros((1, 5))  # a curve the clip does not carry


def spline_at(rows: np.ndarray, times: np.ndarray,
              side: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """Value and slope of a curve's segments at each time.  side="left" takes
    the limit from before, where a key may jump.  Before its first key a
    curve holds that key's value."""
    index = np.searchsorted(rows[:, 0], times, side) - 1
    before = index < 0
    row = rows[np.maximum(index, 0)]
    x = np.where(before, 0.0, times - row[:, 0])
    _, a, b, c, d = row.T
    return ((a * x + b) * x + c) * x + d, np.where(before, 0.0, (3 * a * x + 2 * b) * x + c)


def hermite_keys(components: list[np.ndarray], times: np.ndarray,
                 stop: float) -> tuple[str, np.ndarray, np.ndarray]:
    """(interpolation, times, values) that reproduce a channel's cubics.

    A cubic is fixed by its end values and slopes, so CUBICSPLINE keys with
    Mecanim's slopes as tangents follow the clip exactly, and splitting a
    segment where another component has a key loses nothing.  glTF cannot
    jump at a key, as Mecanim can; a jump gets a second key one float step
    later.  A channel whose segments are all straight is written LINEAR.
    """
    left = [spline_at(c, times, "left") for c in components]
    right = [spline_at(c, times, "right") for c in components]
    in_value, in_slope = (np.column_stack(part) for part in zip(*left))
    out_value, out_slope = (np.column_stack(part) for part in zip(*right))
    out_value[-1], out_slope[-1] = in_value[-1], in_slope[-1]  # the clip ends there
    jumps = ~np.isclose(in_value, out_value, rtol=1e-5, atol=1e-5).all(axis=1)
    jumps[0] = False

    source = np.repeat(np.arange(len(times)), np.where(jumps, 2, 1))
    later = np.zeros(len(source), dtype=bool)
    later[1:] = source[1:] == source[:-1]
    earlier = jumps[source] & ~later
    key_times = times[source].astype(np.float32)
    key_times[later] = np.nextafter(key_times[later], np.float32(np.inf))
    values = np.where(earlier[:, None], in_value[source], out_value[source])
    if all(is_straight(c, stop) for c in components):
        return "LINEAR", key_times, values
    tangents_in = np.where(later[:, None], 0.0, in_slope[source])
    tangents_out = np.where(earlier[:, None], 0.0, out_slope[source])
    triples = np.stack([tangents_in, values, tangents_out], axis=1)
    return "CUBICSPLINE", key_times, triples.reshape(-1, values.shape[1])


def is_straight(rows: np.ndarray, stop: float) -> bool:
    """True if no segment within the clip strays from the straight line
    between its ends by more than rounding."""
    ends = np.append(rows[1:, 0], max(stop, rows[-1, 0]))
    spans = (np.clip(ends, 0.0, stop) - np.clip(rows[:, 0], 0.0, stop))[:, None]
    x = spans * np.linspace(0.0, 1.0, 9)[1:-1]
    a, b = rows[:, 1:2], rows[:, 2:3]
    bend = a * x * (x ** 2 - spans ** 2) + b * x * (x - spans)
    return bool(np.abs(bend).max(initial=0.0) < 1e-5)


def subdivided(times: np.ndarray, components: list[np.ndarray]) -> np.ndarray:
    """Extra keys wherever an angle moves more than EULER_STEP between two."""
    values = np.column_stack([spline_at(c, times)[0] for c in components])
    steps = np.ceil(np.abs(np.diff(values, axis=0)).max(axis=1, initial=0) / EULER_STEP)
    if not len(steps) or steps.max() <= 1:
        return times
    dense = [times[0]]
    for (a, b), n in zip(zip(times, times[1:]), steps):
        dense += list(np.linspace(a, b, max(int(n), 1) + 1)[1:])
    return np.array(dense)


def euler_quaternions(degrees: np.ndarray, order: str) -> np.ndarray:
    """(x, y, z, w) quaternions for Euler angles applied in the given order."""
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(degrees), 1))
    for axis in order:
        i = "XYZ".index(axis)
        half = np.radians(degrees[:, i]) / 2
        step = np.zeros_like(quats)
        step[:, i], step[:, 3] = np.sin(half), np.cos(half)
        quats = quaternion_product(step, quats)
    return quats


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * b for rows of (x, y, z, w) quaternions: b's rotation, then a's."""
    av, aw, bv, bw = a[:, :3], a[:, 3:], b[:, :3], b[:, 3:]
    return np.hstack([aw * bv + bw * av + np.cross(av, bv),
                      aw * bw - np.sum(av * bv, axis=1, keepdims=True)])


def is_switch(binding) -> bool:
    """Whether a binding animates the GameObject's active flag or its
    renderer's enabled flag, either of which shows or hides it."""
    return ((binding.typeID == GAMEOBJECT_CLASS_ID and binding.attribute == BIND_ACTIVE)
            or (binding.typeID in RENDERER_CLASS_IDS and binding.attribute == BIND_ENABLED))


def active_curves(clip) -> dict[int, dict[float, bool]]:
    """{transform path hash: {time: shown}}: the clip animates the GameObject's
    active flag or its renderer's enabled flag, carried as an ordinary float
    curve bound to the object."""
    if not any(is_switch(b) for b in clip.m_ClipBindingConstant.genericBindings):
        return {}  # most clips have none
    keys, _ = clip_keys(clip)
    tracks = defaultdict(dict)
    for binding, start, _ in binding_runs(clip):
        if is_switch(binding):
            tracks[binding.path].update(
                {time: value > 0.5 for time, value in sorted(keys.get(start, []))})
    return dict(tracks)


def bound_transforms(clip, targets: dict[int, object], scene: Scene) -> set[int]:
    """The nodes a clip names, whether or not its curves move them.

    A clip binds every transform of the arrangement it was written for, still
    ones included; that is the only statement in the files about which props a
    clip has out.  See mark_props().
    """
    nodes = set()
    for binding in clip.m_ClipBindingConstant.genericBindings:
        if binding.typeID != TRANSFORM_CLASS_ID:
            continue
        transform = targets.get(binding.path)
        if transform is not None:
            nodes.add(scene.node(transform))
    return nodes


def transform_paths(transform, prefix: str, out: dict[str, object]) -> None:
    out.setdefault(prefix, transform)
    for child in transform.m_Children:
        child = child.read()
        name = child.m_GameObject.read().m_Name
        transform_paths(child, f"{prefix}/{name}" if prefix else name, out)


def animation_targets(roots) -> dict[int, object]:
    """{path hash: Transform}, from the character rig's Animator Avatar.

    EX bundles also carry Avatars for mobs and cut-in props, whose paths share
    generic bone names with the student; only the selected root's Animator says
    which Avatar is hers, so only its ``m_TOS`` may resolve a clip's bindings.
    Paths resolve only against the selected prefabs, dropping effect-rig clips
    along with their geometry.
    """
    table: dict[int, str] = {}
    paths: dict[str, object] = {}
    for root in roots:
        transform_paths(root, "", paths)
        try:
            components = root.m_GameObject.read().m_Component
        except Exception as e:
            print(f"  skipped an animation root: {type(e).__name__}: {e}")
            continue
        for pair in components:
            if pair.component.type.name != "Animator":
                continue
            try:
                avatar = pair.component.read().m_Avatar
                if avatar is not None and avatar.path_id:
                    # setdefault keeps a second root from clobbering the first
                    for path_hash, path in dict(avatar.read().m_TOS).items():
                        table.setdefault(path_hash, path)
            except Exception as e:
                print(f"  skipped an animator avatar: {type(e).__name__}: {e}")
    return {h: paths[p] for h, p in table.items() if p in paths}


def read_clips(env) -> list:
    clips = []
    for obj in env.objects:
        if obj.type.name != "AnimationClip":
            continue
        try:
            clips.append(obj.read())
        except Exception as e:
            print(f"  skipped a clip: {type(e).__name__}: {e}")
    return clips


CHARACTER_CUTIN = re.compile(r"Exs_Cutin(?:_\d+)?", re.I)
# A majority of bindings resolving is enough to tell the character rig from an
# effect or supporting enemy; full coverage would drop legitimate clips from
# older or split rigs.
MIN_CHARACTER_BINDING_COVERAGE = 0.5


def binding_coverage(clip, targets: dict[int, object]) -> float:
    """The share of a clip's Transform bindings that resolve on a rig."""
    bindings = [binding for binding in clip.m_ClipBindingConstant.genericBindings
                if binding.typeID == TRANSFORM_CLASS_ID]
    if not bindings:
        return 1.0
    return sum(binding.path in targets for binding in bindings) / len(bindings)


def is_character_clip(clip, targets: dict[int, object], character: str) -> bool:
    """Whether a clip was authored for this character's rig.

    A student clip binds essentially the whole Avatar; a mob clip names only
    the bones its rig shares.  EX cut-in names are the exception, kept only in
    the game's student cut-in form (bare or numbered take).
    """
    name = clip_name(clip.m_Name, character)
    if name.lower().startswith("exs_cutin") and not CHARACTER_CUTIN.fullmatch(name):
        return False
    return binding_coverage(clip, targets) >= MIN_CHARACTER_BINDING_COVERAGE


def mouth_events(clips) -> dict[tuple[str, int], list[tuple[float, int | None]]]:
    """{clip key: [(time, tile id)]}; None means SetMouthTileToDefault, which
    restores the tile the material itself is saved with.  Ties go to the last
    event at that time."""
    tracks = {}
    for clip in clips:
        keys: dict[float, int | None] = {}
        for event in getattr(clip, "m_Events", None) or []:
            if event.functionName == MOUTH_EVENT:
                keys[float(event.time)] = int(event.intParameter)
            elif event.functionName == MOUTH_DEFAULT_EVENT:
                keys[float(event.time)] = None
        if keys:
            tracks[obj_key(clip)] = sorted(keys.items())
    return tracks


def switch_children(roots, runtime, prefab_halo=None) -> tuple[list, set]:
    """The objects an AniEvt renderer event counts through, in its order, and
    the keys of those whose renderer starts switched off.

    The number is an index into the children of the runtime prefab the game
    instantiates, which orders them differently from the model FBX -- the
    renderers first, then the bones and the halo.  Each is matched back by
    name to the object exported here; None where nothing was, or where the
    child has no renderer of its own for the event to flip, as `HaloRoot`
    and `bone_root` do not.  The events flip the renderer's enabled flag,
    which the runtime prefab sets off for a few spare bodies and props the
    FBX has on.  Without a runtime prefab, the model root's own children are
    the best guess.
    """
    if not roots:
        return [], set()
    own = [child.read() for child in roots[0].m_Children]
    if runtime is None:
        return own, set()
    by_name = {child.m_GameObject.read().m_Name: child for child in own}
    if prefab_halo is not None:
        by_name.setdefault(prefab_halo.m_GameObject.read().m_Name, prefab_halo)
    children, off = [], set()
    for child in runtime.m_Children:
        game_object = child.read().m_GameObject.read()
        renderers = [pair.component.read() for pair in game_object.m_Component
                     if pair.component.type.name in ("SkinnedMeshRenderer", "MeshRenderer")]
        children.append(by_name.get(game_object.m_Name) if renderers else None)
        if children[-1] is not None and not all(r.m_Enabled for r in renderers):
            off.add(obj_key(children[-1]))
    return children, off


def visibility_tracks(clips, children_of, targets: dict[int, object]) -> tuple[dict, dict]:
    """The event tracks and the curve tracks, each {object: {clip: {time: shown}}}.

    A clip switches an object either by an AniEvt event, whose number is an
    index into `children_of(clip)`, or by a curve on the GameObject's active
    flag or the renderer's enabled flag, which names its target by path.  The
    two are kept apart because plan_switches may drop the events.  Later
    events at the same time win.
    """
    events: dict[tuple, dict[tuple, dict[float, bool]]] = defaultdict(lambda: defaultdict(dict))
    curves: dict[tuple, dict[tuple, dict[float, bool]]] = defaultdict(lambda: defaultdict(dict))
    for clip in clips:
        key, children = obj_key(clip), children_of(clip)
        for event in getattr(clip, "m_Events", None) or []:
            shown = RENDERER_EVENTS.get(event.functionName)
            index = int(event.intParameter)
            if shown is None or not 0 <= index < len(children) or children[index] is None:
                continue
            events[obj_key(children[index])][key][float(event.time)] = shown
        try:
            active = active_curves(clip)
        except Exception as e:
            print(f"  skipped a clip: {type(e).__name__}: {e}")
            continue
        for path, track in active.items():
            transform = targets.get(path)
            if transform is not None:
                curves[obj_key(transform)][key].update(track)
    return ({target: dict(by_clip) for target, by_clip in events.items()},
            {target: dict(by_clip) for target, by_clip in curves.items()})


def merge_tracks(*sources: dict) -> dict:
    """The tracks of several sources, one dict of {object: {clip: {time: shown}}}."""
    merged: dict[tuple, dict[tuple, dict[float, bool]]] = defaultdict(lambda: defaultdict(dict))
    for source in sources:
        for target, by_clip in source.items():
            for clip, track in by_clip.items():
                merged[target][clip].update(track)
    return {target: dict(by_clip) for target, by_clip in merged.items()}


def worn_face(model: ModelData, clip_key: tuple) -> list[tuple[float, tuple]]:
    """[(time, face key)]: the one face the character wears, through one clip.

    Every face ships on and the clips switch the spares off, so the face worn
    is the one the clip leaves on by itself; when nothing says, or two are left
    on at once, the face already worn stays on.
    """
    tracks = {key: model.face_tracks.get(key, {}).get(clip_key, {})
              for key in model.face_nodes}
    state = dict.fromkeys(tracks, True)
    worn, out = model.face_default, []
    for time in sorted({0.0} | {t for track in tracks.values() for t in track}):
        state.update({key: track[time] for key, track in tracks.items() if time in track})
        shown = [key for key in tracks if state[key]]
        if len(shown) == 1:
            worn = shown[0]
        elif shown and worn not in shown:
            worn = shown[-1]
        if out and out[-1][0] >= time:
            out[-1] = (out[-1][0], worn)
        else:
            out.append((time, worn))
    return out


def mesh_node(scene: Scene, data: MeshData) -> int:
    """Hand a mesh its glTF node early, so a channel can name it."""
    if data.node is None:
        if data.skin is None and data.parent is not None:
            scene.nodes.append({"name": data.name})
            data.node = len(scene.nodes) - 1
            scene.nodes[data.parent].setdefault("children", []).append(data.node)
        else:
            # A skinned mesh ignores its own node transform, so it can be a root.
            data.node = scene.add({"name": data.name})
    return data.node


def plan_switches(scene: Scene, model: ModelData, children: list, off: set, clips,
                  targets: dict[int, object], parents: dict[int, int],
                  cafe_children: list | None = None) -> None:
    """Settle what the clips switch on and off, and give each mesh a way to fold.

    Each switched mesh gets one morph target that pulls every vertex onto the
    mesh's centre, as the mouth quads do.  Faces are settled as a set, one worn
    at a time; anything else is on or off as its own track says, and takes
    the meshes beneath it in `parents`, the hierarchy as the prefab has it.
    The objects in `off` start switched off.  `cafe_children` is what the
    cafe clips count through, when the cafe prefab drives this same body but
    orders its children differently.

    A clip that switches off every skinned mesh but the faces and mouth makes the
    character vanish, as Izuna does in her EX.  A fold cannot do that cleanly
    (the skinned vertices smear between their bones, and the halo stays up),
    so such clips keep every switch at rest.
    """
    def children_of(clip) -> list:
        if cafe_children is not None and is_cafe_clip(clip, model.name):
            return cafe_children
        return children

    events, curves = visibility_tracks(clips, children_of, targets)
    if len(model.face_meshes) >= 2:
        for key, index in model.face_meshes.items():
            fold(scene, model.meshes[index])
            model.face_nodes[key] = model.meshes[index].node
        model.face_default = everyday_face(model)
    tracks = merge_tracks(events, curves)
    model.face_tracks = tracks
    # Events that leave the idle clip in a spare face are indexing something
    # this cannot see; drop them all and keep only the curves.
    idle = next((clip for clip in clips
                 if clip_name(clip.m_Name, model.name) == DEFAULT_CLIP), None)
    if (model.face_nodes and idle is not None
            and worn_face(model, obj_key(idle))[0][1] != model.face_default):
        model.face_tracks = tracks = merge_tracks(curves)
    for key, node in model.face_nodes.items():
        scene.nodes[node]["weights"] = [0.0 if key == model.face_default else 1.0]

    # Everything else a clip switches off: whatever meshes hang under it.
    tracks = {key: {} for key in off} | tracks
    nodes = {obj_key(t): scene.find(t) for t in [*children, *(cafe_children or []),
                                                 *targets.values()]
             if t is not None and obj_key(t) in tracks}
    mouth = scene.find_key(model.mouth_owner)
    carried = {}
    for key, by_clip in tracks.items():
        if key in model.face_nodes or nodes.get(key) is None:
            continue
        rest = state_at(by_clip.get(obj_key(idle), {}) if idle is not None else {}, 0.0,
                        key not in off)
        if rest and not any(False in track.values() for track in by_clip.values()):
            continue  # never switched off
        meshes = [data for data in model.meshes
                  if data.owner is not None and data.morph is None
                  and nodes[key] in chain(data.owner, parents)]
        if mouth is not None and nodes[key] in chain(mouth, parents):
            model.mouth_switch = key
        if not meshes and model.mouth_switch != key:
            continue
        model.switch_tracks[key] = by_clip
        model.switch_rest[key] = rest
        carried[key] = {id(data) for data in meshes}
        model.switch_nodes[key] = [fold(scene, data) for data in meshes]
        for node in model.switch_nodes[key]:
            scene.nodes[node]["weights"] = [0.0 if rest else 1.0]

    faces = {id(model.meshes[index]) for index in model.face_meshes.values()}
    mouths = set(model.mouth_nodes.values())
    skinned = {id(data) for data in model.meshes
               if data.skin is not None and id(data) not in faces and data.node not in mouths}
    for clip_key in {clip for by_clip in model.switch_tracks.values() for clip in by_clip}:
        for time in sorted({0.0} | {t for by_clip in model.switch_tracks.values()
                                    for t in by_clip.get(clip_key, {})}):
            folded = set().union(*(carried[key] for key, by_clip in model.switch_tracks.items()
                                   if not state_at(by_clip.get(clip_key, {}), time,
                                                   model.switch_rest[key])))
            if skinned and skinned <= folded:
                model.vanish_clips.add(clip_key)
                break


def fold(scene: Scene, data: MeshData) -> int:
    """Give a mesh the morph target that folds it to its centre; its node.

    A skinned vertex still follows its own bones once folded, so the fold
    tears into shards as the pose leaves the bind pose.  A clip that keeps a
    mesh off throughout also hides it outright; see `write_glb`.
    """
    data.morph = (data.positions.mean(axis=0) - data.positions).astype(np.float32)
    return mesh_node(scene, data)


def chain(index: int | None, parents: dict[int, int]):
    """The node and its ancestors."""
    while index is not None:
        yield index
        index = parents.get(index)


def state_at(track: dict[float, bool], time: float, start: bool) -> bool:
    """Whether a track has its object on at a time, starting from `start`."""
    shown = start
    for moment in sorted(track):
        if moment > time:
            break
        shown = track[moment]
    return shown


def switch_track(model: ModelData, key: tuple, clip_key: tuple) -> list[tuple[float, bool]]:
    """[(time, shown)] for one switched object, through one clip."""
    track = ({} if clip_key in model.vanish_clips
             else model.switch_tracks[key].get(clip_key, {}))
    rest = model.switch_rest[key]
    return [(time, state_at(track, time, rest))
            for time in sorted({0.0} | {t for t in track if t > 0.0})]


def switch_channels(model: ModelData, clip) -> list[Channel]:
    """Morph weights that fold away the meshes a clip has switched off."""
    channels = []
    for key, nodes in model.switch_nodes.items():
        track = switch_track(model, key, obj_key(clip))
        if all(shown == model.switch_rest[key] for _, shown in track):
            continue  # three.js restores the rest weight
        times = np.array([time for time, _ in track], dtype=np.float32)
        values = np.array([[float(not shown)] for _, shown in track], dtype=np.float32)
        channels += [Channel(node, "weights", times, values, "STEP") for node in nodes]
    return channels


def everyday_face(model: ModelData) -> tuple:
    """The face the character wears when a clip does not say otherwise.

    The everyday face is the one holding the mouth atlas; failing that, the
    one named without a number.
    """
    mouths = [key for key in model.face_nodes if key in model.face_mouths]
    if len(mouths) == 1:
        return mouths[0]
    pool = mouths or list(model.face_nodes)
    plain = [key for key in pool
             if not any(char.isdigit() for char
                        in model.meshes[model.face_meshes[key]].name.lower().split("face")[-1])]
    return (plain or pool)[0]


def face_channels(model: ModelData, clip) -> list[Channel]:
    """Morph weights that fold away every face but the one being worn."""
    if len(model.face_nodes) < 2:
        return []
    worn = worn_face(model, obj_key(clip))
    times = np.array([time for time, _ in worn], dtype=np.float32)
    return [
        Channel(node, "weights", times,
                np.array([[float(key != shown)] for _, shown in worn], dtype=np.float32),
                "STEP")
        for key, node in model.face_nodes.items()
    ]


def add_clips(scene: Scene, model: ModelData, clips, targets: dict[int, object]) -> None:
    for clip in clips:
        try:
            channels = []
            for (path_hash, attribute), curve in clip_curves(clip).items():
                interpolation, times, values = curve
                transform = targets.get(path_hash)
                if transform is None:
                    continue
                if attribute == BIND_POSITION:
                    values = values * UNITY_TO_GLTF
                elif attribute == BIND_ROTATION:
                    # three.js normalises CUBICSPLINE quaternions after
                    # interpolating, as Mecanim does
                    values = values * QUAT_TO_GLTF
                    if interpolation == "LINEAR":
                        values = normalized_quaternions(values)
                node, path = scene.node(transform), GLTF_PATH[attribute]
                if interpolation == "LINEAR" and is_rest(values, scene.nodes[node][path]):
                    continue
                channels.append(Channel(node, path, times, values.astype(np.float32),
                                        interpolation))
        except Exception as e:
            print(f"  skipped a clip: {type(e).__name__}: {e}")
            continue
        if not channels:  # a camera track, or a clip for a rig not carried
            continue
        channels += (mouth_channels(model, clip) + face_channels(model, clip)
                     + switch_channels(model, clip))
        # a folded skinned mesh tears into shards once the pose leaves the
        # bind pose, so a switch off the whole clip is also hidden outright
        hides = [key for key in model.switch_nodes
                 if not any(shown for _, shown in switch_track(model, key, obj_key(clip)))]
        model.animations.append(Animation(clip_name(clip.m_Name, model.name), channels,
                                          bound_transforms(clip, targets, scene),
                                          hides=hides))


def mouth_channels(model: ModelData, clip) -> list[Channel]:
    """Morph weights that swap the mouth quads, one channel per expression.

    Every clip binds every quad, so one that never mentions the mouth still
    pins it to the default.
    """
    if not model.mouth_nodes:
        return []
    keys = [(time, model.mouth_default if tile is None else tile)
            for time, tile in model.mouth_events.get(obj_key(clip), [])]
    keys = [(time, tile) for time, tile in keys if tile in model.mouth_nodes]
    if not keys or keys[0][0] > 0.0:
        keys.insert(0, (0.0, model.mouth_default))
    if model.mouth_owner in model.face_nodes or model.mouth_switch is not None:
        keys = without_owner(model, clip, keys)
    times = np.array([time for time, _ in keys], dtype=np.float32)
    return [
        Channel(node, "weights", times,
                np.array([[float(tile != shown)] for _, shown in keys], dtype=np.float32),
                "STEP")
        for tile, node in model.mouth_nodes.items()
    ]


def without_owner(model: ModelData, clip,
                  keys: list[tuple[float, int]]) -> list[tuple[float, int | None]]:
    """The mouth track, blanked out over the stretches the mesh it came off is
    not showing -- another face worn, or the mesh switched off: a tile of None
    folds all the quads away."""
    showing = []
    if model.mouth_owner in model.face_nodes:
        showing.append([(time, face == model.mouth_owner)
                        for time, face in worn_face(model, obj_key(clip))])
    if model.mouth_switch is not None:
        showing.append(switch_track(model, model.mouth_switch, obj_key(clip)))

    def at(track, time, first):
        return next((value for moment, value in reversed(track) if moment <= time), first)

    times = {t for t, _ in keys}.union(*({t for t, _ in track} for track in showing))
    return [(time, at(keys, time, keys[0][1])
             if all(at(track, time, track[0][1]) for track in showing) else None)
            for time in sorted(times)]


def is_rest(values: np.ndarray, rest: list[float]) -> bool:
    """True if the curve holds still at the node's own rest transform.  Such
    channels are pure overhead -- three.js restores a node's rest value when
    no playing clip binds it."""
    if not np.allclose(values, values[0], atol=1e-5):
        return False
    if len(rest) == 4:  # a quaternion and its negation are the same rotation
        return abs(float(values[0] @ np.array(rest))) > 1 - 1e-6
    return np.allclose(values[0], rest, atol=1e-5)


def normalized_quaternions(quats: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.divide(quats, lengths, out=np.tile([0.0, 0.0, 0.0, 1.0], (len(quats), 1)),
                      where=lengths > 1e-8)
    # Linear interpolation takes the short way round only if neighbouring keys
    # sit on the same half of the hypersphere; -q is the same rotation as q.
    flips = np.cumprod(np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0, -1.0, 1.0))
    quats[1:] *= flips[:, None]
    return quats


def clip_name(name: str, character: str) -> str:
    prefix = character.replace("_", "")
    if name.replace("_", "").lower().startswith(prefix.lower()):
        trimmed = name[len(character):].lstrip("_")
        return trimmed or name
    return name


def is_cafe_clip(clip, character: str) -> bool:
    """Whether the game plays a clip on the cafe prefab.  Both prefabs share
    one controller, so only the name tells."""
    return clip_name(clip.m_Name, character).lower().startswith("cafe_")


def animation_sort_key(name: str) -> tuple[int, str]:
    """Sort shared player-facing clips before character-specific ones."""
    name = name.casefold()
    comparable = ANIMATION_VARIANT_PREFIX_RE.sub(r"\1", name)
    for priority, prefix in enumerate(ANIMATION_PREFIX_ORDER):
        if comparable.startswith(prefix):
            return priority, name
    return len(ANIMATION_PREFIX_ORDER), name


# --------------------------------------------------------------------------
# geometry extraction
# --------------------------------------------------------------------------

@dataclass
class Channel:
    node: int
    path: str
    times: np.ndarray
    values: np.ndarray
    interpolation: str = "LINEAR"


@dataclass
class Animation:
    name: str
    channels: list[Channel]
    bound: set[int] = field(default_factory=set)   # nodes the clip names at all
    shows: list[str] = field(default_factory=list)  # prop groups it switches on
    hides: list[tuple] = field(default_factory=list)  # switches off the whole clip


@dataclass
class Skin:
    joints: list[int]
    inverse_binds: np.ndarray


@dataclass
class MeshData:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    joints: np.ndarray | None
    weights: np.ndarray | None
    primitives: list[tuple[np.ndarray, str]]  # (indices, material name)
    skin: int | None                          # index into ModelData.skins
    parent: int | None                        # node to hang an unskinned mesh off
    node: int | None = None                   # existing node to put the mesh on
    morph: np.ndarray | None = None           # folds the mesh to a point
    owner: int | None = None                  # node of the renderer's GameObject
    group: str | None = None                  # prop group key, if this is a prop


@dataclass
class ModelData:
    name: str
    meshes: list[MeshData] = field(default_factory=list)
    skins: list[Skin] = field(default_factory=list)
    animations: list[Animation] = field(default_factory=list)
    materials: dict[str, dict] = field(default_factory=dict)  # name -> {color, texture}
    textures: dict[str, bytes] = field(default_factory=dict)  # name -> png bytes
    seen_meshes: set = field(default_factory=set)
    # mouth expressions: each clip's event track, and the node carrying each
    # tile's copy of the mouth quad
    mouth_events: dict[tuple[str, int], list[tuple[float, int | None]]] = \
        field(default_factory=dict)
    mouth_nodes: dict[int, int] = field(default_factory=dict)
    mouth_default: int | None = None
    mouth_owner: tuple | None = None            # transform the quads came off
    # alternate faces, keyed by the transform of the mesh each one is
    face_meshes: dict[tuple, int] = field(default_factory=dict)
    face_mouths: set = field(default_factory=set)
    face_tracks: dict[tuple, dict[tuple, dict[float, bool]]] = field(default_factory=dict)
    face_nodes: dict[tuple, int] = field(default_factory=dict)
    face_default: tuple | None = None
    # other GameObjects the clips switch on and off, keyed like the faces:
    # each one's track per clip, whether it is on at rest, and the nodes of
    # the meshes it carries
    switch_tracks: dict[tuple, dict[tuple, dict[float, bool]]] = field(default_factory=dict)
    switch_rest: dict[tuple, bool] = field(default_factory=dict)
    switch_nodes: dict[tuple, list[int]] = field(default_factory=dict)
    mouth_switch: tuple | None = None           # the switch carrying the mouth
    vanish_clips: set = field(default_factory=set)  # clips left at rest, see plan_switches


def build_skin(scene: Scene, renderer, mesh) -> Skin | None:
    """Bone nodes plus Unity's bind poses, which are already inverse binds."""
    bones, bindposes = renderer.m_Bones, mesh.m_BindPose
    if not bones or not bindposes or len(bones) != len(bindposes):
        return None
    joints, inverse_binds = [], np.empty((len(bones), 4, 4))
    for i, bone in enumerate(bones):
        if not bone.path_id:
            return None
        joints.append(scene.node(bone.read()))
        inverse_binds[i] = FLIP @ bindpose_matrix(bindposes[i]) @ FLIP
    return Skin(joints, inverse_binds)


def skin_weights(handler, bone_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex bone indices and weights, repaired and normalised."""
    idx = np.clip(np.array(handler.m_BoneIndices, dtype=np.int32), 0, bone_count - 1)
    if handler.m_BoneWeights:
        wts = np.array(handler.m_BoneWeights, dtype=np.float64)
    else:
        # A rigidly bound mesh (weapons, hand props) ships only the indices;
        # the implied weight is 1.0.
        idx = idx[:, :1]
        wts = np.ones(idx.shape, dtype=np.float64)

    # Compressed meshes leave junk in the implied-weight slot; drop it and hand
    # back the leftover weight.
    bad = ~np.isfinite(wts) | (wts < 0) | (wts > 1)
    wts = np.where(bad, 0.0, wts)
    residual = np.clip(1.0 - wts.sum(1, keepdims=True), 0.0, None)
    wts += np.where(bad, residual / np.maximum(bad.sum(1, keepdims=True), 1), 0.0)
    # glTF weights are VEC4; Unity stores only the influences used, so pad the
    # narrow ones and keep the four heaviest of any wider one.
    if wts.shape[1] > 4:
        keep = np.argsort(-wts, axis=1, kind="stable")[:, :4]
        rows = np.arange(len(wts))[:, None]
        idx, wts = idx[rows, keep], wts[rows, keep]
    elif wts.shape[1] < 4:
        pad = ((0, 0), (0, 4 - wts.shape[1]))
        idx, wts = np.pad(idx, pad), np.pad(wts, pad)

    total = wts.sum(1, keepdims=True)
    wts = np.divide(wts, total, out=np.zeros_like(wts), where=total > 1e-6)
    wts[total[:, 0] <= 1e-6, 0] = 1.0
    return idx.astype(np.uint16), wts.astype(np.float32)


def texture_png(texture) -> bytes | None:
    try:
        image = texture.image
    except Exception:
        return None
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def tile_material(model: ModelData, texture, image, grid, tile: int, prefix: str,
                  alpha: tuple[str, float]) -> str | None:
    """Crop one tile out of the expression atlas and give it a material."""
    row, column = divmod(tile, 100)
    if not (0 <= column < grid[0] and 0 <= row < grid[1]):
        print(f"  mouth tile {tile} falls outside the atlas; ignored")
        return None
    size = np.array(image.size) / grid
    # Unity's UV origin is bottom-left, the image's is top-left.
    left, top = column * size[0], image.size[1] - (row + 1) * size[1]
    name = f"{texture.m_Name}_{column}_{row}"
    if name not in model.textures:
        crop = image.crop((round(left), round(top), round(left + size[0]), round(top + size[1])))
        buf = io.BytesIO()
        crop.convert("RGBA").save(buf, format="PNG")
        model.textures[name] = buf.getvalue()
    material = f"{prefix}_Mouth_{tile}"
    model.materials.setdefault(
        material, {"color": [1.0, 1.0, 1.0, 1.0], "texture": name, "mouth": None,
                   "alpha": alpha})
    return material


def mouth_tile(name: str, tex_envs: dict, floats: dict, model: ModelData,
               alpha: tuple[str, float]) -> dict | None:
    """How to paint this material's mouth, or None if it does not have one.

    The mouth quad shares the eye submesh, its UVs in a blank corner that the
    shader redirects into one tile of an expression atlas, `_MouthTileTex`'s
    scale and offset picking the tile.  Every tile the clips ask for is
    cropped out and given a material of its own; the one the material is
    saved with is the default.

    That tile is `uv * scale + offset` read back, and neither term is tidy: an
    offset past 1 counts a tile off from a second copy of the atlas, which the
    sampler wraps away (1.625 on an 8-wide grid is column 5, not 13), and a
    negative scale mirrors the axis, putting the offset on the tile's far edge
    and the tile itself one cell back.
    """
    tex_env = tex_envs.get("_MouthTileTex")
    if tex_env is None or tex_env.m_Texture is None or not tex_env.m_Texture.path_id:
        return None
    try:
        texture = tex_env.m_Texture.read()
        image = texture.image
    except Exception as e:
        print(f"  no mouth atlas for {name}: {type(e).__name__}: {e}")
        return None

    grid = np.array([floats.get("_MouthTileCols", 8.0), floats.get("_MouthTileRows", 8.0)])
    scale = np.array([tex_env.m_Scale.x, tex_env.m_Scale.y])
    offset = np.array([tex_env.m_Offset.x, tex_env.m_Offset.y])
    mirror = scale < 0
    column, row = ((offset * grid).round().astype(int) - mirror) % grid.astype(int)
    default = row * 100 + column

    wanted = {default} | {tile for track in model.mouth_events.values()
                          for _, tile in track if tile is not None}
    tiles = {}
    for tile in sorted(wanted):
        material = tile_material(model, texture, image, grid, tile, name, alpha)
        if material is not None:
            tiles[tile] = material
    if default not in tiles:
        return None
    return {"material": tiles[default], "window": 1.0 / (grid * np.abs(scale)),
            "mirror": mirror, "default": default, "tiles": tiles}


def split_mouth(uv: np.ndarray, primitives: list, model: ModelData) -> tuple[int, dict] | None:
    """Move the mouth off the eye material and onto its own atlas tile.

    The mouth usually has no submesh of its own, so it is found the way the
    shader finds it: by the UV window that maps onto the tile.  Rescaling
    those UVs in place is safe because nothing else uses the mouth quad's
    vertices, which is checked rather than assumed.  Some characters give the
    mouth its own primitive, making the whole primitive the quad; otherwise a
    shared eye/mouth primitive is split.  A whole primitive whose UVs all sit
    near zero is geometry without UVs, not the quad.
    """
    for index, (triangles, material) in enumerate(list(primitives)):
        mouth = model.materials.get(material, {}).get("mouth")
        if mouth is None:
            continue
        inside = np.all((uv >= -1e-3) & (uv <= mouth["window"] + 1e-3), axis=1)
        picked = inside[triangles].all(axis=1)
        if not picked.any():
            continue
        vertices = np.unique(triangles[picked])
        if picked.all() and (uv[vertices].max(axis=0) < mouth["window"] / 2).any():
            continue
        others = np.concatenate([triangles[~picked].ravel()]
                                + [t.ravel() for i, (t, _) in enumerate(primitives) if i != index])
        if np.isin(vertices, others).any():
            continue
        uv[vertices] /= mouth["window"]  # the tile window becomes the whole tile
        if mouth["mirror"].any():  # and a mirrored axis is read back to front
            axes = np.ix_(vertices, mouth["mirror"])
            uv[axes] = 1.0 - uv[axes]
        if picked.all():
            primitives[index] = (triangles, mouth["material"])
            return index, mouth
        primitives[index] = (triangles[~picked], material)
        primitives.append((triangles[picked], mouth["material"]))
        return len(primitives) - 1, mouth
    return None


def mouth_quads(scene: Scene, model: ModelData, skin: int | None, joints, weights,
                verts, normals, uvs, primitives: list, found: tuple[int, dict]) -> None:
    """Give every expression the clips use its own copy of the mouth quad.

    glTF cannot animate a UV offset but can animate morph weights, which leave
    the skinning alone -- some characters rig the mouth to a bone of its own.
    Each copy carries one target that folds it to a point.
    """
    index, mouth = found
    if len(mouth["tiles"]) < 2 or skin is None or joints is None:
        return
    triangles, _ = primitives[index]
    vertices = np.unique(triangles)
    remap = np.zeros(len(verts), dtype=np.uint32)
    remap[vertices] = np.arange(len(vertices))
    quad = verts[vertices]
    collapse = (quad.mean(axis=0) - quad).astype(np.float32)

    del primitives[index]
    for tile, material in sorted(mouth["tiles"].items()):
        node = scene.add({"name": f"Mouth_{tile}",
                          "weights": [0.0 if tile == mouth["default"] else 1.0]})
        model.mouth_nodes[tile] = node
        model.meshes.append(MeshData(
            f"Mouth_{tile}", quad, normals[vertices], uvs[vertices],
            joints[vertices], weights[vertices], [(remap[triangles], material)],
            skin, None, node, collapse))
    model.mouth_default = mouth["default"]


# Unity's BlendMode enum, for the two factors that say "alpha blending".
SRC_ALPHA, ONE_MINUS_SRC_ALPHA = 5, 10


def pass_blend(shader) -> tuple[float, float, float, bool] | None:
    """The first pass's src/dst blend factors and ZWrite, and its RenderType.

    Zeroes mean the pass writes `Blend [_SrcBlend] [_DstBlend]` and the
    material's floats decide.
    """
    try:
        sub = shader.m_ParsedForm.m_SubShaders[0]
        state = sub.m_Passes[0].m_State
    except Exception:
        return None
    tags = sub.m_Tags
    tags = {t[0]: t[1] for t in tags.tags} if hasattr(tags, "tags") else dict(tags)
    val = lambda x: float(getattr(x, "val", x))
    return (val(state.rtBlend0.srcBlend), val(state.rtBlend0.destBlend),
            val(state.zWrite), tags.get("RenderType") == "Transparent")


def alpha_mode(material, floats: dict) -> tuple[str, float]:
    """How a material means its `_MainTex` alpha, as a glTF alphaMode.

    Not every alpha channel is an opacity: the body and weapon shaders keep
    other data there, so the alpha is only an opacity where the shader actually
    blends on it, source SrcAlpha over OneMinusSrcAlpha -- usually left to the
    material's `_SrcBlend`/`_DstBlend`.  A blending pass that still writes
    depth (the eyes and mouth, drawn over the face) becomes MASK; `_AlphaClip`
    asks for a cutout outright.
    """
    if floats.get("_AlphaClip", 0.0) >= 0.5:
        return ("MASK", floats.get("_Cutoff", 0.5))
    try:
        blend = pass_blend(material.m_Shader.read()) if material.m_Shader.path_id else None
    except Exception:
        blend = None
    if blend is None:
        return ("OPAQUE", 0.5)
    src, dst, zwrite, transparent = blend
    if src == 0.0 and dst == 0.0:
        src = floats.get("_SrcBlend", 1.0)
        dst = floats.get("_DstBlend", 0.0)
        zwrite = floats.get("_ZWrite", zwrite)
    if src == SRC_ALPHA and dst == ONE_MINUS_SRC_ALPHA:
        return ("MASK", 0.5) if zwrite else ("BLEND", 0.5)
    return ("BLEND", 0.5) if transparent else ("OPAQUE", 0.5)


def read_material(material, model: ModelData) -> str:
    name = material.m_Name
    if name in model.materials:
        return name
    props = material.m_SavedProperties
    colors = {k: v for k, v in props.m_Colors}
    floats = {k: v for k, v in props.m_Floats}
    tex_envs = {k: v for k, v in props.m_TexEnvs}
    alpha = alpha_mode(material, floats)

    tex_name = None
    main = tex_envs.get("_MainTex")
    if main is not None and main.m_Texture is not None and main.m_Texture.path_id:
        try:
            texture = main.m_Texture.read()
            png = texture_png(texture)
        except Exception:
            png = None
        if png is not None:
            tex_name = texture.m_Name
            model.textures.setdefault(tex_name, png)

    # Materials without a texture (eyebrows) carry their colour in _Tint / _Color.
    tint = colors.get("_Tint") or colors.get("_Color")
    color = [1.0, 1.0, 1.0, 1.0]
    if tex_name is None and tint is not None:
        color = [tint.r, tint.g, tint.b, tint.a]

    model.materials[name] = {"color": color, "texture": tex_name, "alpha": alpha,
                             "mouth": mouth_tile(name, tex_envs, floats, model, alpha)}
    return name


def mesh_of(renderer, obj_type: str, game_object):
    if obj_type == "SkinnedMeshRenderer":
        ptr = renderer.m_Mesh
        return ptr.read() if ptr is not None and ptr.path_id else None
    for pair in game_object.m_Component:
        try:
            comp = pair.component.read()
        except Exception:
            continue
        if type(comp).__name__ == "MeshFilter" and comp.m_Mesh.path_id:
            return comp.m_Mesh.read()
    return None


def model_prefabs(env) -> set[tuple[str, int]]:
    """Keys of the GameObjects a bundle files under `<character>/Model/`.

    The AssetBundle container sorts the artists' fbx files from the effect
    prefabs that ship alongside them.
    """
    prefabs = set()
    for obj in env.objects:
        if obj.type.name != "AssetBundle":
            continue
        try:
            container = obj.read().m_Container
        except Exception as e:
            print(f"  skipped a bundle index: {type(e).__name__}: {e}")
            continue
        for path, entry in container:
            if MODEL_ASSET_RE.match(path):
                prefabs.add((obj.assets_file.name, entry.asset.m_PathID))
    return prefabs


def skinned_count(transform) -> int:
    n = sum(1 for pair in transform.m_GameObject.read().m_Component
            if pair.component.type.name == "SkinnedMeshRenderer")
    return n + sum(skinned_count(child.read()) for child in transform.m_Children)


def model_roots(env, name: str, rig: str = "") -> list:
    """The character's own prefabs: the rig it is animated on, and its halo.

    `Model/` still covers props the character never wears, so take the two
    prefabs named after the bundle, `<name>_Mesh` and `<name>_Halo`, falling
    back to the prefab with the most skinned renderers where the rig is named
    something else.  A `rig` suffix such as CAFE_RIG asks for that body,
    `<name><rig>_Mesh`, instead, with no fallback: none when it is absent.
    """
    prefabs = model_prefabs(env)
    candidates = []
    for obj in env.objects:
        if obj.type.name != "Transform":
            continue
        try:
            transform = obj.read()
            father = transform.m_Father
            if father is not None and father.path_id:
                continue
            if (obj.assets_file.name, transform.m_GameObject.path_id) not in prefabs:
                continue
            candidates.append((transform.m_GameObject.read().m_Name.lower(), transform))
        except Exception as e:
            print(f"  skipped a prefab root: {type(e).__name__}: {e}")
    body = [t for n, t in candidates if n == f"{name}{rig}_mesh"]
    if rig and not body:
        return []
    if not body:
        others = [c for c in candidates if not c[0].endswith(f"{CAFE_RIG}_mesh")]
        if others:
            body = [max(others, key=lambda c: skinned_count(c[1]))[1]]
    # Most halo prefabs repeat the bundle name, but several costumes use a
    # different prefix.  They are still unambiguous among Model/ roots. Keep
    # the conventional name first when both it and an alternate are present.
    expected_halo = f"{name}_halo"
    halos = [t for n, t in candidates if n == expected_halo]
    halos += [t for n, t in candidates if n != expected_halo and is_halo(n)]
    return body + halos


def renderers_under(transform, out: list, skip_halos: bool = False) -> None:
    """Every renderer in a prefab, in hierarchy order, inactive parts skipped."""
    game_object = transform.m_GameObject.read()
    if skip_halos and is_halo(game_object.m_Name):
        return
    if not game_object.m_IsActive:
        return
    for pair in game_object.m_Component:
        if pair.component.type.name in ("SkinnedMeshRenderer", "MeshRenderer"):
            out.append(pair.component)
    for child in transform.m_Children:
        renderers_under(child.read(), out, skip_halos)


def find_transform(transform, name: str):
    """Depth-first search of a prefab for the transform of a named GameObject."""
    if transform.m_GameObject.read().m_Name == name:
        return transform
    for child in transform.m_Children:
        found = find_transform(child.read(), name)
        if found is not None:
            return found
    return None


def is_halo(name: str) -> bool:
    """Recognize the runtime halo assembly and the model halo naming convention."""
    name = name.lower().rstrip("_")
    return name == "haloroot" or name.endswith("_halo")


def halo_transforms(transform, out: list) -> None:
    """The `*_Halo` objects of a prefab, the outermost one of a nest only:
    a halo's own subtree is one halo, so stop at the first hit."""
    if is_halo(transform.m_GameObject.read().m_Name):
        out.append(transform)
        return
    for child in transform.m_Children:
        halo_transforms(child.read(), out)


def has_ancestor(transform, ancestor) -> bool:
    key = obj_key(ancestor)
    while transform is not None:
        if obj_key(transform) == key:
            return True
        father = transform.m_Father
        transform = father.read() if (father is not None and father.path_id) else None
    return False


def attach_halo(scene: Scene, roots: list, prefab_halo=None) -> None:
    """Hang the halo off a bone, the way the game does at load time.

    The runtime HaloRoot's follower names the bone and the offset from it;
    its saved world pose can be stale, so it is not used.  Without a
    follower, the halo goes under the head bone and keeps its rest placement.
    """
    halos = [prefab_halo] if prefab_halo is not None else []
    if prefab_halo is None:
        for root in roots:
            halo_transforms(root, halos)
    if not halos or not roots:
        return
    if prefab_halo is not None:
        for component in prefab_halo.m_GameObject.read().m_Component:
            if component.component.type.name != "MonoBehaviour":
                continue
            follower = component.component.read()
            follow_target = getattr(follower, "FollowTarget", None)
            position = getattr(follower, "TargetRelativePosition", None)
            rotation = getattr(follower, "TargetRelativeRotation", None)
            if (not getattr(follower, "m_Enabled", True) or not follow_target
                    or not follow_target.path_id or position is None or rotation is None):
                continue
            target_name = follow_target.read().m_GameObject.read().m_Name
            target = find_transform(roots[0], target_name)
            index = scene.find(prefab_halo)
            if target is not None and index is not None:
                scene.reparent(index, scene.node(target))
                node = scene.nodes[index]
                node["translation"] = [-position.x, position.y, position.z]
                node["rotation"] = [rotation.x, -rotation.y, -rotation.z, rotation.w]
                return
            print(f"  halo follower target {target_name!r} unavailable; using rest placement")
    head = find_transform(roots[0], HEAD_BONE)
    if head is None:
        print(f"  halo left where it stands: no {HEAD_BONE} in the rig")
        return
    for halo in halos:
        index = scene.find(halo)
        if index is not None and not has_ancestor(halo, head):
            scene.reparent(index, scene.node(head))


# How near the floor an unrigged prop has to rest to count as standing on it,
# in the model's own units.
FLOOR_TOLERANCE = 0.02


def stands_on_floor(scene: Scene, parents: dict[int, int], mesh: MeshData) -> bool:
    """True if the mesh rests on the ground the character stands on.  Set
    dressing is modelled standing on the floor; a prop waiting for the game to
    hang it off a hand is not."""
    if mesh.owner is None:
        return False
    world = scene.world(mesh.owner, parents)
    floor = (mesh.positions @ world[:3, :3].T + world[:3, 3])[:, 1].min()
    return abs(floor) <= FLOOR_TOLERANCE


def mark_props(scene: Scene, model: ModelData) -> None:
    """Work out which meshes are props, and which clips have each one out.

    A clip binds every transform of the arrangement it was written for, still
    ones included, so a mesh the default clip binds nothing of is a prop, and
    the clips that do bind it are the clips that have it out.  Props are
    flagged `optional` in the node's glTF `extras`, which the viewer's
    `data-hide-parts` keeps off, and each clip lists the groups it wants back
    in its own `extras`.  Props are grouped because a prop is rarely one mesh;
    a group is the whole subtree the prop hangs under, climbing until the next
    step up would swallow something the default clip binds.
    """
    if not model.animations:
        return
    default = next((anim for anim in model.animations if anim.name == DEFAULT_CLIP),
                   model.animations[0])
    if not default.bound:
        return

    parents = scene.parents()
    children = {i: node.get("children", ()) for i, node in enumerate(scene.nodes)}

    # nodes with something the default clip binds somewhere beneath them
    held = set()

    def mark(index: int) -> bool:
        hit = index in default.bound
        for child in children.get(index, ()):
            hit = mark(child) or hit
        if hit:
            held.add(index)
        return hit

    for root in scene.roots:
        mark(root)

    def subtree(index: int) -> set[int]:
        out, stack = set(), [index]
        while stack:
            index = stack.pop()
            out.add(index)
            stack += list(children.get(index, ()))
        return out

    def reach(index: int) -> set[int]:
        """The prop's whole subtree, from as high up as it stays a prop."""
        while True:
            parent = parents.get(index)
            if parent is None or parent in held:
                return subtree(index)
            index = parent

    bones = {joint for skin in model.skins for joint in skin.joints}
    driven = [{channel.node for channel in anim.channels} for anim in model.animations]

    # the mouth quads answer this the same way the body does
    props: list[tuple[int, set[int], set[int]]] = []
    for i, mesh in enumerate(model.meshes):
        nodes = set(model.skins[mesh.skin].joints) if mesh.skin is not None else set()
        nodes.update(index for index in (mesh.owner, mesh.node, mesh.parent)
                     if index is not None)
        if not nodes or is_halo(mesh.name):
            continue
        # A mesh on a bone is out whenever a clip binds it, still or not; a
        # mesh the rig does not carry at all is placed by its own curves or
        # not at all, so it goes by which clips drive it instead.
        rigged = any(node in bones for index in nodes for node in chain(index, parents))
        if not rigged and stands_on_floor(scene, parents, mesh):
            continue
        out = [anim.bound if rigged else drive
               for anim, drive in zip(model.animations, driven)]
        if nodes & out[model.animations.index(default)]:
            continue
        found = set().union(*(reach(index) for index in nodes))
        props.append((i, found, {k for k, seen in enumerate(out) if found & seen}))
    # a default clip that leaves out the character herself is one read wrong
    if not props or len(props) == len(model.meshes):
        return

    # Props sharing a rig share a group, so they come and go together.
    groups: list[tuple[set[int], set[int], set[int]]] = []
    for index, nodes, shown in props:
        merged = [g for g in groups if g[1] & nodes]
        for group in merged:
            groups.remove(group)
        groups.append(({index}.union(*(g[0] for g in merged)),
                       nodes.union(*(g[1] for g in merged)),
                       shown.union(*(g[2] for g in merged))))

    for i, (meshes, _nodes, shown) in enumerate(groups):
        # a made-up name: the viewer matches node names, and the prop's own
        # could collide with one of the character's
        key = f"prop{i}"
        for index in meshes:
            model.meshes[index].group = key
        for k in sorted(shown):
            model.animations[k].shows.append(key)


def extract(env, name: str, animations: bool = True, runtime=None, rig: str = "",
            keep_clip=None, cafe_runtime=None) -> tuple[Scene, ModelData]:
    """The scene and model of one rig; `keep_clip`, given a clip, says whether
    it is this rig's when the bundle holds clips of another body.
    `cafe_runtime` is the cafe prefab, when it drives this same body."""
    scene = Scene()
    prefab_halo = character_halo(runtime)
    model = ModelData(name=name)
    roots = model_roots(env, name, rig)
    # the expression track decides how many copies of the mouth quad are needed
    clips = read_clips(env) if animations else []
    targets = animation_targets(roots) if clips else {}
    clips = [clip for clip in clips if is_character_clip(clip, targets, name)
             and (keep_clip is None or keep_clip(clip))]
    model.mouth_events = mouth_events(clips)

    renderers: list = []
    for root in roots:
        renderers_under(root, renderers, skip_halos=prefab_halo is not None)
    if prefab_halo is not None:
        renderers_under(prefab_halo, renderers)
    for obj in renderers:
        try:
            add_renderer(scene, model, obj)
        except Exception as e:
            # skip renderers pointing into bundles we did not load
            print(f"  skipped a renderer: {type(e).__name__}: {e}")

    if model.meshes:
        parents = scene.parents()  # before the halo moves to the head
        attach_halo(scene, roots, prefab_halo)
        cafe_children = (switch_children(roots, cafe_runtime, prefab_halo)[0]
                         if cafe_runtime is not None else None)
        # even a model with no clips wears one face, not all of them
        plan_switches(scene, model, *switch_children(roots, runtime, prefab_halo),
                      clips, targets, parents, cafe_children)
    if model.meshes and targets:
        add_clips(scene, model, clips, targets)
        mark_props(scene, model)
        # mark_props() falls back on the bundle's clip order; sort only after
        # it has chosen its rest pose
        model.animations.sort(key=lambda animation: animation_sort_key(animation.name))
    return scene, model


def add_renderer(scene: Scene, model: ModelData, obj) -> None:
    renderer = obj.read()
    game_object = renderer.m_GameObject.read()
    mesh = mesh_of(renderer, obj.type.name, game_object)
    if mesh is None or mesh.m_Name.startswith("OL_"):
        return  # OL_ meshes are the inverted-hull outline duplicates
    if obj_key(mesh) in model.seen_meshes:
        return  # the same mesh is sometimes referenced from several bundles
    model.seen_meshes.add(obj_key(mesh))

    handler = MeshHandler(mesh)
    handler.process()
    if not handler.m_Vertices or not handler.m_VertexCount:
        return

    # vertices stay in mesh space: the bind poses are the inverse binds, so
    # the skin places them at runtime
    verts = (np.array(handler.m_Vertices)[:, :3] * UNITY_TO_GLTF).astype(np.float32)
    if handler.m_Normals:
        normals = (np.array(handler.m_Normals)[:, :3] * UNITY_TO_GLTF).astype(np.float32)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-8)
    else:
        normals = np.zeros_like(verts)
    if handler.m_UV0:
        uv = np.array(handler.m_UV0, dtype=np.float32)[:, :2]
    else:
        uv = np.zeros((len(verts), 2), dtype=np.float32)

    transform = game_object.m_Transform.read() if game_object.m_Transform.path_id else None
    owner = scene.node(transform) if transform is not None else None
    skin_index = joints = weights = parent = None
    skin = build_skin(scene, renderer, mesh) if obj.type.name == "SkinnedMeshRenderer" else None
    if skin is not None and handler.m_BoneIndices:
        joints, weights = skin_weights(handler, len(skin.joints))
        model.skins.append(skin)
        skin_index = len(model.skins) - 1
    else:
        parent = owner

    materials = list(renderer.m_Materials)
    primitives = []
    for i, triangles in enumerate(handler.get_triangles()):
        if not triangles:
            continue
        tris = np.array(triangles, dtype=np.uint32)[:, ::-1]
        mat_name = "default"
        if i < len(materials) and materials[i].path_id:
            try:
                mat_name = read_material(materials[i].read(), model)
            except Exception as e:
                print(f"  {mesh.m_Name} keeps an untextured submesh: "
                      f"{type(e).__name__}: {e}")
        primitives.append((tris, mat_name))
    found = split_mouth(uv, primitives, model)
    uvs = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)
    if found is not None and not model.mouth_nodes:
        mouth_quads(scene, model, skin_index, joints, weights, verts, normals, uvs,
                    primitives, found)
        model.mouth_owner = obj_key(transform) if transform is not None else None
    if primitives:
        model.meshes.append(MeshData(mesh.m_Name, verts, normals, uvs, joints, weights,
                                     primitives, skin_index, parent, owner=owner))
        names = [name.lower() for _, name in primitives]
        painted = (any(name.endswith(FACE_MATERIAL) for name in names)
                   and not any(name.endswith(tail) for name in names
                               for tail in BODY_MATERIALS))
        if transform is not None and painted:
            model.face_meshes[obj_key(transform)] = len(model.meshes) - 1
            if any(MOUTH_MATERIAL in name for name in names):
                model.face_mouths.add(obj_key(transform))


# --------------------------------------------------------------------------
# glb writing
# --------------------------------------------------------------------------

COMPONENT_TYPE = {"float32": 5126, "uint32": 5125, "uint16": 5123, "uint8": 5121}
CHANNEL_KIND = {1: "SCALAR", 3: "VEC3", 4: "VEC4"}  # weights; translation, scale; rotation


class GlbBuilder:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []
        self._cache: dict[tuple, int] = {}

    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.blob.extend(data)
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, array: np.ndarray, kind: str, target: int | None = None) -> int:
        data = np.ascontiguousarray(array).tobytes()
        key = (data, kind, target)
        if key in self._cache:
            return self._cache[key]
        acc = {
            "bufferView": self._view(data, target),
            "componentType": COMPONENT_TYPE[array.dtype.name],
            "count": len(array),
            "type": kind,
        }
        if array.dtype.kind == "f":  # required for POSITION and animation inputs
            flat = array.reshape(len(array), -1)
            acc["min"] = flat.min(axis=0).tolist()
            acc["max"] = flat.max(axis=0).tolist()
        self.accessors.append(acc)
        self._cache[key] = len(self.accessors) - 1
        return self._cache[key]

    def image(self, png: bytes) -> int:
        return self._view(png)


def write_glb(scene: Scene, model: ModelData, path: Path) -> None:
    b = GlbBuilder()
    nodes = [dict(node) for node in scene.nodes]

    images, textures, tex_index = [], [], {}
    for name, png in model.textures.items():
        images.append({"bufferView": b.image(png), "mimeType": "image/png", "name": name})
        textures.append({"source": len(images) - 1, "sampler": 0})
        tex_index[name] = len(textures) - 1

    materials, mat_index = [], {}
    for name, info in model.materials.items():
        pbr = {"baseColorFactor": info["color"], "metallicFactor": 0.0, "roughnessFactor": 0.85}
        if info["texture"] in tex_index:
            pbr["baseColorTexture"] = {"index": tex_index[info["texture"]]}
        materials.append({
            "name": name,
            "pbrMetallicRoughness": pbr,
            "doubleSided": True,
            "alphaMode": info["alpha"][0],
        })
        if info["alpha"][0] == "MASK":
            materials[-1]["alphaCutoff"] = info["alpha"][1]
        mat_index[name] = len(materials) - 1

    skins = [{
        "joints": skin.joints,
        "inverseBindMatrices": b.accessor(
            # glTF matrices are column-major, numpy's are not.
            np.ascontiguousarray(skin.inverse_binds.transpose(0, 2, 1), dtype=np.float32)
            .reshape(-1, 16), "MAT4"),
    } for skin in model.skins]

    meshes, mesh_nodes = [], []
    for data in model.meshes:
        attributes = {
            "POSITION": b.accessor(data.positions, "VEC3", 34962),
            "NORMAL": b.accessor(data.normals, "VEC3", 34962),
            "TEXCOORD_0": b.accessor(data.uvs, "VEC2", 34962),
        }
        if data.joints is not None:
            attributes["JOINTS_0"] = b.accessor(data.joints, "VEC4", 34962)
            attributes["WEIGHTS_0"] = b.accessor(data.weights, "VEC4", 34962)
        targets = ([{"POSITION": b.accessor(data.morph, "VEC3", 34962)}]
                   if data.morph is not None else None)
        primitives = []
        for indices, material in data.primitives:
            entry = {"attributes": attributes,
                     "indices": b.accessor(indices.reshape(-1), "SCALAR", 34963)}
            if material in mat_index:
                entry["material"] = mat_index[material]
            if targets is not None:
                entry["targets"] = targets
            primitives.append(entry)
        meshes.append({"name": data.name, "primitives": primitives})

        node = {"name": data.name, "mesh": len(meshes) - 1}
        if data.node is not None:
            node = nodes[data.node]
            node["mesh"] = len(meshes) - 1
            if data.skin is not None:
                node["skin"] = data.skin
        elif data.skin is not None:
            # A skinned mesh ignores its own node transform; the joints carry it.
            node["skin"] = data.skin
            nodes.append(node)
            mesh_nodes.append(len(nodes) - 1)
        else:
            nodes.append(node)
            if data.parent is not None:
                nodes[data.parent].setdefault("children", []).append(len(nodes) - 1)
            else:
                mesh_nodes.append(len(nodes) - 1)
        if data.group is not None:
            # GLTFLoader copies extras onto userData, where the viewer's
            # data-hide-parts looks; the group key is what a clip's show names
            node["extras"] = {"optional": True, data.group: True}

    # A switch answers to a made-up name, as a prop group does, unless its
    # nodes carry meshes it does not switch: hiding a node hides its subtree.
    switch_names = {}
    for i, (key, indices) in enumerate(model.switch_nodes.items()):
        subtree, stack = set(), list(indices)
        while stack:
            index = stack.pop()
            subtree.add(index)
            stack += nodes[index].get("children", [])
        if any("mesh" in nodes[index] and index not in indices for index in subtree):
            continue
        switch_names[key] = f"switch{i}"
        for index in indices:
            nodes[index].setdefault("extras", {})[switch_names[key]] = True

    animations = []
    for anim in model.animations:
        samplers, channels = [], []
        for channel in anim.channels:
            samplers.append({
                "input": b.accessor(channel.times, "SCALAR"),
                "output": b.accessor(channel.values, CHANNEL_KIND[channel.values.shape[1]]),
                "interpolation": channel.interpolation,
            })
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": channel.node, "path": channel.path}})
        entry = {"name": anim.name, "samplers": samplers, "channels": channels}
        hides = [switch_names[key] for key in anim.hides if key in switch_names]
        if anim.shows or hides:
            # GLTFLoader copies these onto AnimationClip.userData as this
            # clip's show and hide rules
            entry["extras"] = {k: v for k, v in (("show", anim.shows), ("hide", hides)) if v}
        animations.append(entry)

    gltf = {
        "asset": {"version": "2.0", "generator": "blue-archive model_viewer/export_models.py"},
        "scene": 0,
        "scenes": [{"nodes": scene.roots + mesh_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": len(b.blob)}],
        "bufferViews": b.views,
        "accessors": b.accessors,
    }
    for key, value in (("materials", materials), ("skins", skins), ("animations", animations)):
        if value:
            gltf[key] = value
    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(b.blob)
    bin_chunk += b"\0" * (-len(bin_chunk) % 4)

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def list_characters(bundle_dir: Path) -> list[str]:
    names = set()
    for bundle in bundle_dir.glob("assets-_mx-characters-*.bundle"):
        match = CHARACTER_BUNDLE_RE.match(bundle.name)
        if match:
            names.add(match.group(1))
    return sorted(names)


def bundles_for(bundle_dir: Path, name: str) -> list[Path]:
    return sorted(bundle_dir.glob(f"assets-_mx-characters-{name}-_mxdependency-*.bundle"))


def prolog_bundles_for(bundle_dir: Path, name: str) -> list[Path]:
    """Preload bundles that can carry a legacy character's full Model prefab."""
    preload_dir = bundle_dir.parent.parent / "Preload" / bundle_dir.name
    return sorted(preload_dir.glob(
        f"prologdepengroup-assets-_mx-characters-{name}-_mxprolog-*.bundle"))


def read_devname_map() -> dict[str, str]:
    """Bundle name (lowercased) -> the wiki's name for that character."""
    names = {}
    for path in DEVNAME_MAPS:
        if not path.exists():
            print(f"  no {path.name}; run update.py to fetch it")
            continue
        for dev, entry in json.loads(path.read_text(encoding="utf-8")).items():
            variant = entry["variant"]
            names[dev.lower()] = entry["firstname"] + (f" ({variant})" if variant else "")
    if not COSTUME_TABLE.exists():
        print(f"  no {COSTUME_TABLE.name}; run update.py to fetch it")
        return names
    costumes = json.loads(COSTUME_TABLE.read_text(encoding="utf-8"))["DataList"]
    for costume in costumes:
        prefab = costume["ModelPrefabName"].lower()
        name = names.get(costume["DevName"].lower().removesuffix("_default"))
        # NPC and event copies share the prefab; the first costume the maps know wins
        if prefab and name:
            names.setdefault(prefab, name)
    return names


def qualifier(tail: str) -> str:
    """`carrier2`, `scenario_01` -> `Carrier 2`, `Scenario 1`."""
    parts = []
    for token in tail.split("_"):
        match = re.fullmatch(r"([a-z]*)([0-9]*)", token)
        if match is None:
            parts.append(token)
            continue
        word, digits = match.groups()
        label = QUALIFIERS.get(word, word.capitalize())
        parts.append(f"{label} {int(digits)}".strip() if digits else label)
    return " ".join(parts)


def model_names(bundles: list[str]) -> dict[str, str]:
    """Bundle name -> the file name its model is written under, minus `.glb`.

    `aru_newyear` becomes `Aru (New Year)`. A bundle whose name the map covers
    only up to a suffix carries that suffix as a second parenthesis —
    `ch0334_carrier` is `Arisu (Battle) (Carrier)` — and one the map does not
    know at all keeps its own name.  Bundles that land on the same name get
    their own trailing number appended.
    """
    devnames = read_devname_map()
    names = {}
    for bundle in bundles:
        stem = bundle.removesuffix("_original").split("_")
        names[bundle] = bundle
        for i in range(len(stem), 0, -1):
            name = devnames.get("_".join(stem[:i]))
            if name:
                tail = "_".join(stem[i:])
                names[bundle] = f"{name} ({qualifier(tail)})" if tail else name
                break
    # two bundles can share a name; their own trailing number tells them apart
    taken = Counter(names.values())
    for bundle, name in names.items():
        number = re.search(r"([0-9]+)$", bundle)
        if taken[name] > 1 and number:
            names[bundle] = f"{name} ({int(number.group(1))})"
    return names


def runtime_prefabs(bundle_dir: Path, name: str) -> dict[str, object]:
    """The root Transforms of the character prefabs the game instantiates,
    by rig: `""` for `<name>.prefab`, CAFE_RIG for `<name>_CafeOnly.prefab`.

    Their bundles are loaded separately so runtime rigs and cut-in clips do
    not enter the model export.  Only their halo, their children's order and
    renderer flags, and their Animator are read.
    """
    files = sorted(bundle_dir.glob(f"character-{name}-_mxload-*.bundle"))
    if not files:
        return {}
    env = UnityPy.load(*map(str, files))
    resolve_dependencies(env, bundle_dir)
    expected = {f"assets/_mx/addressableasset/character/{name}/{name}{rig}.prefab".lower(): rig
                for rig in ("", CAFE_RIG)}
    prefabs = {}
    for obj in env.objects:
        if obj.type.name != "AssetBundle":
            continue
        for path, entry in obj.read().m_Container:
            rig = expected.get(path.lower())
            if rig is None:
                continue
            for component in entry.asset.read().m_Component:
                if component.component.type.name == "Transform":
                    prefabs[rig] = component.component.read()
    return prefabs


def runtime_avatar(runtime) -> str | None:
    """The name of the Avatar a runtime prefab's Animator drives."""
    if runtime is None:
        return None
    for pair in runtime.m_GameObject.read().m_Component:
        if pair.component.type.name != "Animator":
            continue
        try:
            avatar = pair.component.read().m_Avatar
            if avatar is not None and avatar.path_id:
                return avatar.read().m_Name
        except Exception as e:
            print(f"  skipped a runtime animator: {type(e).__name__}: {e}")
    return None


def character_halo(runtime):
    """The halo assembly of the runtime character prefab.

    Costume prefabs can reference another character's halo.  The prefab's
    HaloRoot supplies both those references and placement.
    """
    halo = find_transform(runtime, "HaloRoot") if runtime is not None else None
    if halo is not None:
        renderers = []
        renderers_under(halo, renderers)
        if renderers:
            return halo
    return None


def export(bundle_dir: Path, name: str, out: Path, animations: bool = True,
           overwrite: bool = False) -> Path | None:
    if out.exists() and not overwrite:
        print(f"  skipped {out.name}: already exists")
        return None
    files = bundles_for(bundle_dir, name)
    if not files:
        print(f"  no bundles found for {name}")
        return None
    env = UnityPy.load(*[str(f) for f in files])
    resolve_dependencies(env, bundle_dir)
    runtimes = runtime_prefabs(bundle_dir, name)
    runtime = runtimes.get("")
    # A cafe prefab either drives the main body, in which case the cafe clips'
    # renderer events count through its children, or swaps in a body with its
    # own Avatar, which gets a model of its own.  Both controllers list the
    # cafe clips, so each goes to the rig it covers more of, ties to the main.
    cafe_roots = model_roots(env, name, CAFE_RIG)
    cafe_avatar = runtime_avatar(runtimes.get(CAFE_RIG))
    is_cafe = keep_clip = None
    cafe_runtime = runtimes.get(CAFE_RIG)
    if cafe_roots and cafe_avatar not in (None, runtime_avatar(runtime)):
        cafe_runtime = None
        main_targets = animation_targets(model_roots(env, name))
        cafe_targets = animation_targets(cafe_roots)

        def is_cafe(clip) -> bool:
            return binding_coverage(clip, cafe_targets) > binding_coverage(clip, main_targets)

        def keep_clip(clip) -> bool:
            return not is_cafe(clip)
    scene, model = extract(env, name, animations, runtime, keep_clip=keep_clip,
                           cafe_runtime=cafe_runtime)
    if not model.skins:
        # A few legacy sets retain only a loose weapon; the complete Model
        # prefab is in the preload bundles, loaded only for the affected export.
        prolog = prolog_bundles_for(bundle_dir, name)
        if prolog:
            print("  no skinned character rig in Model/; trying preload bundle")
            for bundle in prolog:
                env.load_file(str(bundle))
            scene, model = extract(env, name, animations, runtime, keep_clip=keep_clip,
                                   cafe_runtime=cafe_runtime)
    if not model.meshes:
        print(f"  no meshes found for {name}")
        return None
    write_model(scene, model, out)
    if is_cafe is not None:
        scene, model = extract(env, name, animations, runtimes[CAFE_RIG], CAFE_RIG, is_cafe)
        if model.meshes:
            write_model(scene, model, out.with_name(f"{out.stem} ({CAFE_LABEL}){out.suffix}"))
    return out


def write_model(scene: Scene, model: ModelData, out: Path) -> None:
    write_glb(scene, model, out)
    verts = sum(len(m.positions) for m in model.meshes)
    joints = sum(len(s.joints) for s in model.skins)
    print(f"  {out.name}: {len(model.meshes)} meshes, {verts} verts, {joints} joints, "
          f"{len(model.textures)} textures, {len(model.animations)} clips, "
          f"{out.stat().st_size // 1024} KiB")
    if model.animations:
        print(f"    clips: {', '.join(a.name for a in model.animations)}")


def export_one(bundle_dir: Path, name: str, out: Path, animations: bool,
               overwrite: bool) -> Path | None:
    print(f"{name} -> {out.name}", flush=True)
    try:
        return export(bundle_dir, name, out, animations, overwrite)
    except Exception as e:  # a broken bundle should not stop the batch
        print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("characters", nargs="*",
                        help="bundle names (airi_original) or wiki names (Aru (New Year))")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, default=HERE / "models")
    parser.add_argument("--list", action="store_true", help="list available characters and exit")
    parser.add_argument("--all", action="store_true", help="export every character")
    parser.add_argument("--no-animations", action="store_true", help="export the bind pose only")
    parser.add_argument("--overwrite", action="store_true", help="replace existing model files")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 4),
                        help="concurrent export jobs (default: hardware threads minus 4, minimum 1)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    available = list_characters(args.bundle_dir)
    # the bundle name stays the handle the export works from: it names the
    # prefabs to walk and the prefix to strip off clip names
    names = model_names(available)
    if args.list:
        width = max(map(len, available), default=0)
        print("\n".join(f"{b:<{width}}  {names[b]}" for b in available))
        return

    by_wiki_name = {wiki.lower(): bundle for bundle, wiki in names.items()}
    targets = list(dict.fromkeys(
        available if args.all else
        [by_wiki_name.get(c.lower(), c) for c in args.characters]))
    if not targets:
        parser.error("give at least one character name, or --all / --list")

    args.out.mkdir(parents=True, exist_ok=True)
    exported = []
    skipped = []
    pending = []
    for name in targets:
        out = args.out / f"{names.get(name, name)}.glb"
        if out.exists() and not args.overwrite:
            print(f"  skipped {out.name}: already exists")
            skipped.append(out.name)
        else:
            pending.append((name, out))

    jobs = min(args.jobs, len(pending))
    if jobs == 1:
        for name, out in pending:
            out = export_one(args.bundle_dir, name, out, not args.no_animations,
                             args.overwrite)
            if out is not None:
                exported.append(out.name)
    elif jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(export_one, args.bundle_dir, name, out,
                            not args.no_animations, args.overwrite): name
                for name, out in pending
            }
            for future in as_completed(futures):
                try:
                    out = future.result()
                except Exception as e:
                    print(f"  {futures[future]} failed: {type(e).__name__}: {e}")
                    continue
                if out is not None:
                    exported.append(out.name)

    # the mtime is a cache-busting token: the viewer appends it to the URL
    manifest = args.out / "models.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else None
    models = {p.name: int(p.stat().st_mtime) for p in sorted(args.out.glob("*.glb"))}
    if models != existing:
        manifest.write_text(json.dumps(models, indent=2))
    print(f"\n{len(exported)} model(s) exported, {len(skipped)} skipped; manifest at {manifest}")


if __name__ == "__main__":
    main()
