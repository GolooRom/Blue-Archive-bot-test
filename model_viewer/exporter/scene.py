"""The Transform hierarchy as glTF nodes, and the data an export collects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Unity is left-handed, glTF right-handed: mirror X, and with it the triangle
# winding, the quaternions (x, -y, -z, w) and the matrices (FLIP @ M @ FLIP).
UNITY_TO_GLTF = np.array([-1.0, 1.0, 1.0])
QUAT_TO_GLTF = np.array([1.0, -1.0, -1.0, 1.0])
FLIP = np.diag([-1.0, 1.0, 1.0, 1.0])


def obj_key(obj) -> tuple[str, int]:
    reader = obj.object_reader
    return (reader.assets_file.name, reader.path_id)


class Scene:
    """The Transform hierarchy of one character, mirrored into glTF nodes."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.roots: list[int] = []
        self._index: dict[tuple[str, int], int] = {}
        # the model prefab's transforms for the runtime prefab's; see runtime_parts()
        self.stand_ins: dict[tuple[str, int], object] = {}

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
        transform = self.stand_ins.get(obj_key(transform), transform)
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


def chain(index: int | None, parents: dict[int, int]):
    """The node and its ancestors."""
    while index is not None:
        yield index
        index = parents.get(index)


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
class Mouth:
    """One mouth's quads: the node carrying each tile's copy, keyed by
    (tile id, flipped), and the tile the material is saved with."""
    nodes: dict[tuple[int, bool], int]
    default: tuple[int, bool]
    owner: tuple | None = None                  # transform the quads came off
    switch: tuple | None = None                 # the switch carrying the quads


@dataclass
class ModelData:
    name: str
    meshes: list[MeshData] = field(default_factory=list)
    skins: list[Skin] = field(default_factory=list)
    animations: list[Animation] = field(default_factory=list)
    materials: dict[str, dict] = field(default_factory=dict)  # name -> {color, texture}
    textures: dict[str, bytes] = field(default_factory=dict)  # name -> png bytes
    seen_meshes: set = field(default_factory=set)
    # mouth expressions: each clip's event track, and the quads of every mouth
    # the events drive
    mouth_events: dict[tuple[str, int], list[tuple[float, tuple[int, bool] | None]]] = \
        field(default_factory=dict)
    mouths: list[Mouth] = field(default_factory=list)
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
    vanish_clips: set = field(default_factory=set)  # clips left at rest, see plan_switches
