"""One character's export, from loaded bundles to .glb files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import UnityPy

from .bundles import bundles_for, prolog_bundles_for, resolve_dependencies
from .clips import (BIND_POSITION, BIND_ROTATION, GLTF_PATH, animation_sort_key,
                    animation_targets, binding_coverage, bound_transforms, clip_curves,
                    clip_name, is_character_clip, is_rest, normalized_quaternions, read_clips)
from .geometry import add_renderer
from .glb import write_glb
from .mouth import mouth_events
from .prefabs import (CAFE_LABEL, CAFE_RIG, attach_halo, character_halo, model_roots,
                      renderers_under, runtime_avatar, runtime_prefabs)
from .props import mark_props
from .scene import QUAT_TO_GLTF, UNITY_TO_GLTF, Animation, Channel, ModelData, Scene, obj_key
from .switches import (face_channels, mouth_channels, plan_switches, runtime_parts,
                       switch_channels, switch_children, switch_track)


def add_clips(scene: Scene, model: ModelData, clips, targets: dict[int, object]) -> None:
    for clip in clips:
        try:
            channels = []
            for (path_hash, attribute), curve in clip_curves(clip).items():
                interpolation, times, values = curve
                transform = targets.get(path_hash)
                if transform is None:
                    continue
                if attribute == BIND_POSITION:
                    values = values * UNITY_TO_GLTF
                elif attribute == BIND_ROTATION:
                    # three.js normalises CUBICSPLINE quaternions after
                    # interpolating, as Mecanim does
                    values = values * QUAT_TO_GLTF
                    if interpolation == "LINEAR":
                        values = normalized_quaternions(values)
                node, path = scene.node(transform), GLTF_PATH[attribute]
                if interpolation == "LINEAR" and is_rest(values, scene.nodes[node][path]):
                    continue
                channels.append(Channel(node, path, times, values.astype(np.float32),
                                        interpolation))
        except Exception as e:
            print(f"  skipped a clip: {type(e).__name__}: {e}")
            continue
        if not channels:  # a camera track, or a clip for a rig not carried
            continue
        channels += (mouth_channels(model, clip) + face_channels(model, clip)
                     + switch_channels(model, clip))
        # a folded skinned mesh tears into shards once the pose leaves the
        # bind pose, so a switch off the whole clip is also hidden outright
        hides = [key for key in model.switch_nodes
                 if not any(shown for _, shown in switch_track(model, key, obj_key(clip)))]
        model.animations.append(Animation(clip_name(clip.m_Name, model.name), channels,
                                          bound_transforms(clip, targets, scene),
                                          hides=hides))


def extract(env, name: str, animations: bool = True, runtime=None, rig: str = "",
            keep_clip=None, cafe_runtime=None) -> tuple[Scene, ModelData]:
    """The scene and model of one rig; `keep_clip`, given a clip, says whether
    it is this rig's when the bundle holds clips of another body.
    `cafe_runtime` is the cafe prefab, when it drives this same body."""
    scene = Scene()
    prefab_halo = character_halo(runtime)
    model = ModelData(name=name)
    roots = model_roots(env, name, rig)
    # the expression track decides how many copies of the mouth quad are needed
    clips = read_clips(env) if animations else []
    targets = animation_targets(roots) if clips else {}
    clips = [clip for clip in clips if is_character_clip(clip, targets, name)
             and (keep_clip is None or keep_clip(clip))]
    model.mouth_events = mouth_events(clips)

    renderers: list = []
    for root in roots:
        renderers_under(root, renderers, skip_halos=prefab_halo is not None)
    if prefab_halo is not None:
        renderers_under(prefab_halo, renderers)
    for obj in renderers:
        try:
            add_renderer(scene, model, obj)
        except Exception as e:
            # skip renderers pointing into bundles we did not load
            print(f"  skipped a renderer: {type(e).__name__}: {e}")
    counterparts, runtime_targets = runtime_parts(scene, model, roots, runtime)
    if targets:
        targets |= runtime_targets

    if model.meshes:
        parents = scene.parents()  # before the halo moves to the head
        attach_halo(scene, roots, prefab_halo)
        cafe_children = (switch_children(roots, cafe_runtime, prefab_halo)[0]
                         if cafe_runtime is not None else None)
        # even a model with no clips wears one face, not all of them
        plan_switches(scene, model, *switch_children(roots, runtime, prefab_halo, counterparts),
                      clips, targets, parents, cafe_children)
    if model.meshes and targets:
        add_clips(scene, model, clips, targets)
        mark_props(scene, model)
        # mark_props() falls back on the bundle's clip order; sort only after
        # it has chosen its rest pose
        model.animations.sort(key=lambda animation: animation_sort_key(animation.name))
    return scene, model


def export(bundle_dir: Path, name: str, out: Path, animations: bool = True,
           overwrite: bool = False) -> Path | None:
    if out.exists() and not overwrite:
        print(f"  skipped {out.name}: already exists")
        return None
    files = bundles_for(bundle_dir, name)
    if not files:
        print(f"  no bundles found for {name}")
        return None
    env = UnityPy.load(*[str(f) for f in files])
    resolve_dependencies(env, bundle_dir)
    runtimes = runtime_prefabs(bundle_dir, name)
    runtime = runtimes.get("")
    # A cafe prefab either drives the main body, in which case the cafe clips'
    # renderer events count through its children, or swaps in a body with its
    # own Avatar, which gets a model of its own.  Both controllers list the
    # cafe clips, so each goes to the rig it covers more of, ties to the main.
    cafe_roots = model_roots(env, name, CAFE_RIG)
    cafe_avatar = runtime_avatar(runtimes.get(CAFE_RIG))
    is_cafe = keep_clip = None
    cafe_runtime = runtimes.get(CAFE_RIG)
    if cafe_roots and cafe_avatar not in (None, runtime_avatar(runtime)):
        cafe_runtime = None
        main_targets = animation_targets(model_roots(env, name))
        cafe_targets = animation_targets(cafe_roots)

        def is_cafe(clip) -> bool:
            return binding_coverage(clip, cafe_targets) > binding_coverage(clip, main_targets)

        def keep_clip(clip) -> bool:
            return not is_cafe(clip)
    scene, model = extract(env, name, animations, runtime, keep_clip=keep_clip,
                           cafe_runtime=cafe_runtime)
    if not model.skins:
        # A few legacy sets retain only a loose weapon; the complete Model
        # prefab is in the preload bundles, loaded only for the affected export.
        prolog = prolog_bundles_for(bundle_dir, name)
        if prolog:
            print("  no skinned character rig in Model/; trying preload bundle")
            for bundle in prolog:
                env.load_file(str(bundle))
            scene, model = extract(env, name, animations, runtime, keep_clip=keep_clip,
                                   cafe_runtime=cafe_runtime)
    if not model.meshes:
        print(f"  no meshes found for {name}")
        return None
    write_model(scene, model, out)
    if is_cafe is not None:
        scene, model = extract(env, name, animations, runtimes[CAFE_RIG], CAFE_RIG, is_cafe)
        if model.meshes:
            write_model(scene, model, out.with_name(f"{out.stem} ({CAFE_LABEL}){out.suffix}"))
    return out


def write_model(scene: Scene, model: ModelData, out: Path) -> None:
    write_glb(scene, model, out)
    verts = sum(len(m.positions) for m in model.meshes)
    joints = sum(len(s.joints) for s in model.skins)
    print(f"  {out.name}: {len(model.meshes)} meshes, {verts} verts, {joints} joints, "
          f"{len(model.textures)} textures, {len(model.animations)} clips, "
          f"{out.stat().st_size // 1024} KiB")
    if model.animations:
        print(f"    clips: {', '.join(a.name for a in model.animations)}")
