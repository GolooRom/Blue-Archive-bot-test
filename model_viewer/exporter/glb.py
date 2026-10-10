"""Writing the scene and model out as a glTF binary."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .scene import ModelData, Scene

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
