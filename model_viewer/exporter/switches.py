"""What the clips switch on and off: spare parts and faces, folded away by
morph weights, and the mouth quads swapped the same way."""

from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np
from UnityPy.helpers.MeshHelper import MeshHandler

from .clips import DEFAULT_CLIP, active_curves, clip_name, is_cafe_clip, transform_paths
from .geometry import add_renderer, mesh_of
from .prefabs import is_halo
from .scene import UNITY_TO_GLTF, Channel, MeshData, ModelData, Mouth, Scene, chain, obj_key

RENDERER_EVENTS = {"AniEvt_EnableChildRenderer": True,
                   "AniEvt_DisableChildRenderer": False}


def switch_children(roots, runtime, prefab_halo=None,
                    counterparts: dict | None = None) -> tuple[list, list]:
    """The objects an AniEvt renderer event counts through, in its order, and
    the keys of those whose renderer starts switched off, in the same order.

    The number is an index into the children of the runtime prefab the game
    instantiates, which orders them differently from the model FBX -- the
    renderers first, then the bones and the halo.  Each is matched back by
    name to the object exported here; None where nothing was, or where the
    child has no renderer of its own for the event to flip, as `HaloRoot`
    and `bone_root` do not.  The events flip the renderer's enabled flag,
    which the runtime prefab sets off for a few spare bodies and props the
    FBX has on.  Without a runtime prefab, the model root's own children are
    the best guess.  For `counterparts`, see runtime_parts().
    """
    if not roots:
        return [], []
    own = [child.read() for child in roots[0].m_Children]
    if runtime is None:
        return own, []
    by_name = {child.m_GameObject.read().m_Name: child for child in own}
    if prefab_halo is not None:
        by_name.setdefault(prefab_halo.m_GameObject.read().m_Name, prefab_halo)
    by_name |= counterparts or {}
    children, off = [], []
    for child in runtime.m_Children:
        game_object = child.read().m_GameObject.read()
        renderers = [pair.component.read() for pair in game_object.m_Component
                     if pair.component.type.name in ("SkinnedMeshRenderer", "MeshRenderer")]
        children.append(by_name.get(game_object.m_Name) if renderers else None)
        if children[-1] is not None and not all(r.m_Enabled for r in renderers):
            off.append(obj_key(children[-1]))
    return children, off


# how closely a runtime part's bounds must match a model part's, relative to its size
REMADE_PART_TOLERANCE = 0.01


def mesh_positions(renderer, obj_type: str, game_object) -> np.ndarray | None:
    """A renderer's vertices, in glTF space as MeshData keeps them."""
    mesh = mesh_of(renderer, obj_type, game_object)
    if mesh is None:
        return None
    handler = MeshHandler(mesh)
    handler.process()
    if not handler.m_Vertices or not handler.m_VertexCount:
        return None
    return np.array(handler.m_Vertices)[:, :3] * UNITY_TO_GLTF


def same_bounds(a: np.ndarray, b: np.ndarray) -> bool:
    low, high = b.min(axis=0), b.max(axis=0)
    tolerance = REMADE_PART_TOLERANCE * np.linalg.norm(high - low)
    return (np.allclose(a.min(axis=0), low, atol=tolerance)
            and np.allclose(a.max(axis=0), high, atol=tolerance))


def runtime_parts(scene: Scene, model: ModelData, roots, runtime) -> tuple[dict, dict]:
    """Add the runtime prefab's own parts, which the model prefab lacks.

    A part whose bounds match a model part absent from the runtime prefab is
    a copy of it and is skipped.  Returns {runtime child name: the transform
    its renderer events mean} for switch_children(), and {path hash:
    transform} for the new transforms the clips animate.
    """
    if runtime is None or not roots:
        return {}, {}
    own = {child.read().m_GameObject.read().m_Name for child in roots[0].m_Children}
    model_paths: dict[str, object] = {}
    transform_paths(roots[0], "", model_paths)
    by_name = {path.rsplit("/", 1)[-1]: transform for path, transform in model_paths.items()}
    runtime_paths: dict[str, object] = {}
    transform_paths(runtime, "", runtime_paths)
    runtime_names = {path.rsplit("/", 1)[-1] for path in runtime_paths}
    exported = [(scene.nodes[data.owner]["name"], data.positions) for data in model.meshes
                if data.owner is not None]

    counterparts, added = {}, []
    for child in runtime.m_Children:
        child = child.read()
        game_object = child.m_GameObject.read()
        name = game_object.m_Name
        if name in own or is_halo(name) or not game_object.m_IsActive:
            continue
        renderers = [pair.component for pair in game_object.m_Component
                     if pair.component.type.name in ("SkinnedMeshRenderer", "MeshRenderer")]
        positions = [mesh_positions(obj.read(), obj.type.name, game_object) for obj in renderers]
        positions = [p for p in positions if p is not None]
        if not positions:
            continue  # Asuna's Box001 has no mesh
        match = next((owner for owner, vertices in exported if owner not in runtime_names
                      and any(same_bounds(p, vertices) for p in positions)), None)
        if match is not None:
            counterparts[name] = by_name.get(match)
        else:
            counterparts[name] = child
            added.append((child, renderers))

    for path, transform in runtime_paths.items():
        if path in model_paths:
            scene.stand_ins[obj_key(transform)] = model_paths[path]
    path_of = {obj_key(transform): path for path, transform in runtime_paths.items()}
    animated = set()
    for child, renderers in added:
        name = child.m_GameObject.read().m_Name
        subtree: dict[str, object] = {}
        transform_paths(child, name, subtree)
        animated |= subtree.keys()
        for obj in renderers:
            add_renderer(scene, model, obj, outline_mesh=True)
            for bone in getattr(obj.read(), "m_Bones", None) or []:
                path = path_of.get(obj_key(bone.read())) if bone.path_id else None
                while path and path not in model_paths:
                    animated.add(path)
                    path = path.rpartition("/")[0]
    # Mecanim names a transform by the CRC-32 of its path
    return counterparts, {zlib.crc32(path.encode()): runtime_paths[path] for path in animated}


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


def plan_switches(scene: Scene, model: ModelData, children: list, off: list, clips,
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
    owners = [(mouth, scene.find_key(mouth.owner)) for mouth in model.mouths]
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
        carries_mouth = False
        for mouth, owner in owners:
            if owner is not None and nodes[key] in chain(owner, parents):
                mouth.switch = key
                carries_mouth = True
        if not meshes and not carries_mouth:
            continue
        model.switch_tracks[key] = by_clip
        model.switch_rest[key] = rest
        carried[key] = {id(data) for data in meshes}
        model.switch_nodes[key] = [fold(scene, data) for data in meshes]
        for node in model.switch_nodes[key]:
            scene.nodes[node]["weights"] = [0.0 if rest else 1.0]

    faces = {id(model.meshes[index]) for index in model.face_meshes.values()}
    mouths = {node for mouth in model.mouths for node in mouth.nodes.values()}
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


def mouth_channels(model: ModelData, clip) -> list[Channel]:
    """Morph weights that swap the mouth quads, one channel per expression.

    Every clip binds every quad, so one that never mentions the mouth still
    pins it to the default.  The events reach every mouth at once, as the
    game's CharacterMultiMouthEventReceiver passes them on (Tsukuyo's tree).
    """
    channels = []
    for mouth in model.mouths:
        keys = [(time, mouth.default if tile is None else tile)
                for time, tile in model.mouth_events.get(obj_key(clip), [])]
        keys = [(time, tile) for time, tile in keys if tile in mouth.nodes]
        if not keys or keys[0][0] > 0.0:
            keys.insert(0, (0.0, mouth.default))
        if mouth.owner in model.face_nodes or mouth.switch is not None:
            keys = without_owner(model, mouth, clip, keys)
        times = np.array([time for time, _ in keys], dtype=np.float32)
        channels += [
            Channel(node, "weights", times,
                    np.array([[float(tile != shown)] for _, shown in keys], dtype=np.float32),
                    "STEP")
            for tile, node in mouth.nodes.items()
        ]
    return channels


def without_owner(model: ModelData, mouth: Mouth, clip,
                  keys: list[tuple[float, int]]) -> list[tuple[float, int | None]]:
    """The mouth track, blanked out over the stretches the mesh it came off is
    not showing -- another face worn, or the mesh switched off: a tile of None
    folds all the quads away."""
    showing = []
    if mouth.owner in model.face_nodes:
        showing.append([(time, face == mouth.owner)
                        for time, face in worn_face(model, obj_key(clip))])
    if mouth.switch is not None:
        showing.append(switch_track(model, mouth.switch, obj_key(clip)))

    def at(track, time, first):
        return next((value for moment, value in reversed(track) if moment <= time), first)

    times = {t for t, _ in keys}.union(*({t for t, _ in track} for track in showing))
    return [(time, at(keys, time, keys[0][1])
             if all(at(track, time, track[0][1]) for track in showing) else None)
            for time in sorted(times)]
