"""Renderers into glTF meshes: vertices, skins and primitives."""

from __future__ import annotations

import numpy as np
from UnityPy.helpers.MeshHelper import MeshHandler

from .fx import particle_mesh
from .materials import read_material
from .mouth import mouth_quads, split_mouth
from .scene import (FLIP, UNITY_TO_GLTF, MeshData, ModelData, Scene, Skin, bindpose_matrix,
                    obj_key)

# A spare face is a mesh of its own, switched by the clips, and recognised by
# its materials rather than its name.  The everyday face holds the mouth the
# clips lip-sync.
FACE_MATERIAL = "_face"
MOUTH_MATERIAL = "_eyemouth"
BODY_MATERIALS = ("_body", "_hair")


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


def add_renderer(scene: Scene, model: ModelData, obj, outline_mesh: bool = False) -> None:
    renderer = obj.read()
    game_object = renderer.m_GameObject.read()
    size = None
    if obj.type.name == "ParticleSystemRenderer":
        mesh, size = particle_mesh(obj) or (None, None)
    else:
        mesh = mesh_of(renderer, obj.type.name, game_object)
    if mesh is None or (mesh.m_Name.startswith("OL_") and not outline_mesh):
        return  # OL_ meshes are the inverted-hull outline duplicates
    # the same mesh is sometimes referenced from several bundles; particle
    # emitters share one on purpose, so theirs count once per emitter
    key = obj_key(mesh) if size is None else (obj_key(mesh), obj_key(game_object))
    if key in model.seen_meshes:
        return
    model.seen_meshes.add(key)

    handler = MeshHandler(mesh)
    handler.process()
    if not handler.m_Vertices or not handler.m_VertexCount:
        return

    # vertices stay in mesh space: the bind poses are the inverse binds, so
    # the skin places them at runtime
    scale = np.ones(3) if size is None else size  # a particle's size scales its mesh
    verts = (np.array(handler.m_Vertices)[:, :3] * scale * UNITY_TO_GLTF).astype(np.float32)
    if handler.m_Normals:
        normals = (np.array(handler.m_Normals)[:, :3] / scale * UNITY_TO_GLTF).astype(np.float32)
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
    if found is not None:
        mouth_quads(scene, model, skin_index, joints, weights, verts, normals, uvs,
                    primitives, found, obj_key(transform) if transform is not None else None)
    if primitives:
        model.meshes.append(MeshData(mesh.m_Name.removeprefix("OL_") if outline_mesh else mesh.m_Name,
                                     verts, normals, uvs, joints, weights,
                                     primitives, skin_index, parent, owner=owner))
        names = [name.lower() for _, name in primitives]
        painted = (any(name.endswith(FACE_MATERIAL) for name in names)
                   and not any(name.endswith(tail) for name in names
                               for tail in BODY_MATERIALS))
        if transform is not None and painted:
            model.face_meshes[obj_key(transform)] = len(model.meshes) - 1
            if any(MOUTH_MATERIAL in name for name in names):
                model.face_mouths.add(obj_key(transform))
