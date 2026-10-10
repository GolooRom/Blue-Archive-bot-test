"""Unity materials as glTF materials: base texture, colour and alpha mode."""

from __future__ import annotations

import io

from .fx import fx_texture
from .mouth import mouth_tile
from .scene import ModelData


def texture_png(texture) -> bytes | None:
    try:
        image = texture.image
    except Exception:
        return None
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


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
    elif "_Tex_Main" in tex_envs:
        png = fx_texture(name, tex_envs, colors)
        if png is not None:
            tex_name = f"{name}_Baked"
            model.textures.setdefault(tex_name, png)

    # Materials without a texture (eyebrows) carry their colour in _Tint / _Color.
    tint = colors.get("_Tint") or colors.get("_Color")
    color = [1.0, 1.0, 1.0, 1.0]
    if tex_name is None and tint is not None:
        color = [tint.r, tint.g, tint.b, tint.a]

    model.materials[name] = {"color": color, "texture": tex_name, "alpha": alpha,
                             "mouth": mouth_tile(name, tex_envs, floats, model, alpha)}
    return name
