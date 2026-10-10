"""Effect parts: particle systems that amount to a fixed mesh, and the DSFX
effect shaders' textures, baked into one image."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# the largest side of a baked FX texture
FX_TEXTURE_LIMIT = 2048


def fx_texture(name: str, tex_envs: dict, colors: dict) -> bytes | None:
    """The DSFX effect shaders' `_Tex_Main`, tinted by `_Color` and cut out by
    `_Tex_Mask`, baked into one image of the mesh's UV square.

    The shader tiles and offsets each texture on its own, and scrolls them by
    particle custom data; the bake keeps the saved tiling and offset only.
    The mask is grey in its RGB or in its alpha, so it counts as R times A.
    An HDR tint, there for bloom, keeps its hue and loses its overbright:
    clipping it would wash the gradient out to white.
    """
    layers = []
    for key in ("_Tex_Main", "_Tex_Mask"):
        env = tex_envs.get(key)
        if env is None or env.m_Texture is None or not env.m_Texture.path_id:
            continue
        try:
            image = env.m_Texture.read().image.convert("RGBA")
        except Exception as e:
            print(f"  {name} bakes without {key}: {type(e).__name__}: {e}")
            continue
        layers.append((key, np.asarray(image, dtype=np.float32) / 255.0,
                       np.array([env.m_Scale.x, env.m_Scale.y]),
                       np.array([env.m_Offset.x, env.m_Offset.y])))
    if not layers or layers[0][0] != "_Tex_Main":
        return None
    # enough pixels for the most tiled texture to keep its own resolution
    width = min(FX_TEXTURE_LIMIT, max(round(p.shape[1] * abs(s[0])) for _, p, s, _ in layers))
    height = min(FX_TEXTURE_LIMIT, max(round(p.shape[0] * abs(s[1])) for _, p, s, _ in layers))
    # Unity's UV origin is bottom-left, the image's is top-left.
    u, v = np.meshgrid((np.arange(width) + 0.5) / width, 1.0 - (np.arange(height) + 0.5) / height)

    def sample(pixels, scale, offset):
        rows, cols = pixels.shape[:2]
        su = (u * scale[0] + offset[0]) % 1.0
        sv = (v * scale[1] + offset[1]) % 1.0
        return pixels[np.minimum(((1.0 - sv) * rows).astype(int), rows - 1),
                      np.minimum((su * cols).astype(int), cols - 1)]

    out = sample(*layers[0][1:]).copy()
    tint = colors.get("_Color")
    if tint is not None:
        rgb = np.array([tint.r, tint.g, tint.b])
        out *= [*(rgb / max(1.0, rgb.max())), min(1.0, tint.a)]
    for _, pixels, scale, offset in layers[1:]:
        mask = sample(pixels, scale, offset)
        out[..., 3] *= mask[..., 0] * mask[..., 3]
    buf = io.BytesIO()
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8), "RGBA").save(
        buf, format="PNG")
    return buf.getvalue()


# ParticleSystemRenderMode.Mesh and ParticleSystemRenderSpace.Local
PARTICLE_MESH_MODE, PARTICLE_LOCAL_ALIGNMENT = 4, 2
# modules that leave a lone particle where and as it was born; the spin of
# RotationModule and the shader data of CustomDataModule are not exported
STILL_PARTICLE_MODULES = {"InitialModule", "EmissionModule", "RotationModule", "CustomDataModule"}


def particle_mesh(obj):
    """The mesh and size of a particle renderer that amounts to a fixed mesh,
    else None.

    Mika's halo trail is one mesh particle: born on its emitter with no speed
    or turn, aligned with it, outliving the loop, so always there.
    """
    try:
        renderer = obj.read()
        if (not renderer.m_Enabled or renderer.m_RenderMode != PARTICLE_MESH_MODE
                or renderer.m_RenderAlignment != PARTICLE_LOCAL_ALIGNMENT
                or renderer.m_Mesh is None or not renderer.m_Mesh.path_id):
            return None
        system = next(pair.component.read_typetree()
                      for pair in renderer.m_GameObject.read().m_Component
                      if pair.component.type.name == "ParticleSystem")
    except Exception:
        return None
    initial = system["InitialModule"]
    enabled = {k for k, v in system.items() if isinstance(v, dict) and v.get("enabled")}
    curves = ("startSize", "startSizeY", "startSizeZ", "startSpeed", "startRotation",
              "startLifetime")
    if (initial["maxNumParticles"] != 1 or not enabled <= STILL_PARTICLE_MODULES
            or initial["rotation3D"] or not system["looping"]
            or any(initial[k]["minMaxState"] != 0 for k in curves)
            or initial["startSpeed"]["scalar"] or initial["startRotation"]["scalar"]
            or initial["startLifetime"]["scalar"] < system["lengthInSec"]):
        return None
    size = initial["startSize"]["scalar"]
    if initial["size3D"]:
        size = [size, initial["startSizeY"]["scalar"], initial["startSizeZ"]["scalar"]]
    return renderer.m_Mesh.read(), np.broadcast_to(np.array(size, dtype=np.float64), 3)
