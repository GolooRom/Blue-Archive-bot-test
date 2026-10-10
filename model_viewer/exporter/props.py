"""Which meshes are props, and which clips have each one out."""

from __future__ import annotations

from .clips import DEFAULT_CLIP
from .prefabs import is_halo
from .scene import MeshData, ModelData, Scene, chain

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
        # a halo's particle meshes are named for the effect, not the halo
        if any(is_halo(scene.nodes[node].get("name", ""))
               for index in (mesh.owner, mesh.node, mesh.parent)
               for node in chain(index, parents)):
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
