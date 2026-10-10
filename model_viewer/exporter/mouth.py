"""Mouth expressions: the atlas tiles the clips switch between, and a copy of
the mouth quad for each tile."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .scene import MeshData, ModelData, Mouth, Scene, obj_key

# Mouth expressions are events, not curves: SetMouthTile(row * 100 + col), the
# row counted from the bottom of the atlas (Unity's UV origin).  The flipped
# event shows the same tile mirrored left to right.
MOUTH_EVENTS = {"SetMouthTile": False, "SetHorizontallyFlippedMouthTile": True}
MOUTH_DEFAULT_EVENT = "SetMouthTileToDefault"


def mouth_events(clips) -> dict[tuple[str, int], list[tuple[float, tuple[int, bool] | None]]]:
    """{clip key: [(time, (tile id, flipped))]}; None means
    SetMouthTileToDefault, which restores the tile the material itself is
    saved with.  Ties go to the last event at that time."""
    tracks = {}
    for clip in clips:
        keys: dict[float, tuple[int, bool] | None] = {}
        for event in getattr(clip, "m_Events", None) or []:
            if event.functionName in MOUTH_EVENTS:
                keys[float(event.time)] = (int(event.intParameter),
                                           MOUTH_EVENTS[event.functionName])
            elif event.functionName == MOUTH_DEFAULT_EVENT:
                keys[float(event.time)] = None
        if keys:
            tracks[obj_key(clip)] = sorted(keys.items())
    return tracks


def tile_label(key: tuple[int, bool]) -> str:
    tile, flipped = key
    return f"{tile}_Flipped" if flipped else str(tile)


def tile_material(model: ModelData, texture, image, grid, key: tuple[int, bool], prefix: str,
                  alpha: tuple[str, float]) -> str | None:
    """Crop one tile out of the expression atlas, mirrored if flipped, and give
    it a material."""
    tile, flipped = key
    row, column = divmod(tile, 100)
    if not (0 <= column < grid[0] and 0 <= row < grid[1]):
        print(f"  mouth tile {tile} falls outside the atlas; ignored")
        return None
    size = np.array(image.size) / grid
    # Unity's UV origin is bottom-left, the image's is top-left.
    left, top = column * size[0], image.size[1] - (row + 1) * size[1]
    name = f"{texture.m_Name}_{column}_{row}" + ("_flipped" if flipped else "")
    if name not in model.textures:
        crop = image.crop((round(left), round(top), round(left + size[0]), round(top + size[1])))
        if flipped:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        buf = io.BytesIO()
        crop.convert("RGBA").save(buf, format="PNG")
        model.textures[name] = buf.getvalue()
    material = f"{prefix}_Mouth_{tile_label(key)}"
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

    A horizontal mirror is what the flipped event sets, so it belongs to the
    tile: the default is that tile flipped, and the quad's UVs stay unmirrored
    in U so that the events' unflipped tiles read the right way round.
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
    default = (int(row * 100 + column), bool(mirror[0]))

    wanted = {default} | {key for track in model.mouth_events.values()
                          for _, key in track if key is not None}
    tiles = {}
    for key in sorted(wanted):
        material = tile_material(model, texture, image, grid, key, name, alpha)
        if material is not None:
            tiles[key] = material
    if default not in tiles:
        return None
    return {"material": tiles[default], "window": 1.0 / (grid * np.abs(scale)),
            "mirror": mirror & [False, True], "default": default, "tiles": tiles}


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
                verts, normals, uvs, primitives: list, found: tuple[int, dict],
                owner: tuple | None) -> None:
    """Give every expression the clips use its own copy of the mouth quad.

    glTF cannot animate a UV offset but can animate morph weights, which leave
    the skinning alone -- some characters rig the mouth to a bone of its own.
    Each copy carries one target that folds it to a point.
    """
    index, mouth = found
    if len(mouth["tiles"]) < 2 or skin is None or joints is None:
        return
    # the first mouth keeps the plain names
    prefix = f"Mouth{len(model.mouths) + 1}" if model.mouths else "Mouth"
    nodes = {}
    triangles, _ = primitives[index]
    vertices = np.unique(triangles)
    remap = np.zeros(len(verts), dtype=np.uint32)
    remap[vertices] = np.arange(len(vertices))
    quad = verts[vertices]
    collapse = (quad.mean(axis=0) - quad).astype(np.float32)

    del primitives[index]
    for key, material in sorted(mouth["tiles"].items()):
        node = scene.add({"name": f"{prefix}_{tile_label(key)}",
                          "weights": [0.0 if key == mouth["default"] else 1.0]})
        nodes[key] = node
        model.meshes.append(MeshData(
            f"{prefix}_{tile_label(key)}", quad, normals[vertices], uvs[vertices],
            joints[vertices], weights[vertices], [(remap[triangles], material)],
            skin, None, node, collapse))
    model.mouths.append(Mouth(nodes, mouth["default"], owner))
