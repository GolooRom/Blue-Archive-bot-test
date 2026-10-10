"""Mecanim animation clips: decoding their curves into glTF samplers, and
telling the character's own clips from the rest of the bundle's."""

from __future__ import annotations

import re
import struct
from collections import defaultdict

import numpy as np

from .scene import Scene

# The clip the viewer opens on. It decides the rest state of switched parts
# and which props count as part of the character; without it, mark_props()
# uses the first clip in bundle order. Keep in sync with data-anim-default in
# viewer.html and Module:ModelViewer.
DEFAULT_CLIP = "Cafe_Reaction"

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
