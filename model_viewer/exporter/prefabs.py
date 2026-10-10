"""The character's prefabs: the model prefab's rig and halo, and the runtime
prefab the game instantiates."""

from __future__ import annotations

import re
from pathlib import Path

import UnityPy

from .bundles import resolve_dependencies
from .fx import particle_mesh
from .scene import Scene, obj_key

MODEL_ASSET_RE = re.compile(r"^assets/_mx/characters/[^/]+/model/", re.I)

# A few characters sit in the cafe as a separate body, `<name>_CafeOnly_Mesh`,
# whose clips are keyed to its own rest frames; it is exported as a model of
# its own, `<label> (Cafe).glb`.
CAFE_RIG = "_cafeonly"
CAFE_LABEL = "Cafe"

# The halo's fallback attachment when the runtime prefab gives no follower.
HEAD_BONE = "Bip001 Head"


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
        kind = pair.component.type.name
        if (kind in ("SkinnedMeshRenderer", "MeshRenderer")
                or kind == "ParticleSystemRenderer" and particle_mesh(pair.component)):
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
