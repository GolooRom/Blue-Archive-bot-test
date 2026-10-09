> [!WARNING]
> This submodule of Blue Archive Bot is almost entirely written by AI.


# Blue Archive model viewer

`export_models.py` reads character models from the game's Unity asset bundles
and writes one self-contained `.glb` per character into `models/`: skinned
meshes, embedded textures, the bone hierarchy and the character's animation
clips. `viewer.html` displays them with the
[ModelViewer gadget](https://dev.miraheze.org/wiki/Template:ModelViewer/doc).

```sh
# list exportable bundles, each with the wiki name it is saved under
python export_models.py --list

python export_models.py airi_original aris_original
python export_models.py "Aru (New Year)"               # wiki names work too
python export_models.py --all
python export_models.py --all --jobs 4
python export_models.py --overwrite airi_original      # regenerate an existing model
python export_models.py --overwrite --no-animations airi_original   # bind pose only

# the viewer needs HTTP; file:// blocks the fetch of models.json
python -m http.server 8000   # then open http://localhost:8000/viewer.html
```

Bundles are read from the Steam build under `vendor/BlueArchive/`; use
`--bundle-dir` to point elsewhere and `--out` to change the output directory.
Existing `.glb` files are skipped unless `--overwrite` is given. `--jobs N` sets
the number of worker processes (default: hardware threads minus 4, at least 1).

Each run rewrites `models/models.json` if it changed. It maps every `.glb` in
the output directory to its mtime. `viewer.html` builds its model list from it
and appends the mtime to each URL, so the browser does not serve a stale copy.
The animation list comes from the clips inside the selected `.glb`. Clips are
sorted with `Cafe_Reaction` first, then the other shared cafe and formation
clips, EX clips and the common combat states; character-specific clips follow
alphabetically. The viewer opens `Cafe_Reaction`.

## Wiki manifest

From the bot repository root:

```sh
python -m model_viewer.save_manifest --dry-run
python -m model_viewer.save_manifest
```

The second command saves `Module:ModelViewer/data.json` with the configured
pywikibot account, and only if the content changed. The manifest reads
`models/models.json` and groups models by character: `Aru (New Year).glb` is
the `New Year` skin of `Aru`. `Module:ModelViewer` itself is maintained on the
wiki, not in this repository.

GLBs are fetched from the root of `lihaohong6/BlueArchiveModels` through
`https://cdn.jsdelivr.net/gh/lihaohong6/BlueArchiveModels@main/`. Push them
there separately. Use `--base URL` to pin a commit instead of the cached
branch.

On the wiki, invoke the module directly:

```wikitext
{{#invoke:ModelViewer|character|Aru}}
{{#invoke:ModelViewer|model|Aru (New Year)}}
{{#invoke:ModelViewer|all}}
```

`character` offers every model of one character, `model` shows one exact
model, and `all` creates one click-to-load viewer per character. Gadget options
such as `height`, `camera` and `anim-default` can be passed as named arguments.
The module is adapted from
[Stella Sora's module, revision 34049](https://stellasora.miraheze.org/wiki/Module:ModelViewer?oldid=34049).

## Export details

**Names.** Bundles use the developers' names (`ch0069`, `aru_newyear`,
`shiroko_ridingsuit`). Each model is saved under the wiki's name instead
(`Mika.glb`, `Aru (New Year).glb`), looked up in `json/devname_map.json` and
`json/devname_map_aux.json` (refreshed by `update.py`). Bundles named after a
model prefab code that the maps lack, such as `ch0061`, are resolved through
`json/CostumeExcelTable.json`. A suffix the tables do not cover becomes a
second parenthesis: `ch0334_carrier` is `Arisu (Battle) (Carrier).glb`. Two
bundles that map to the same name get their trailing number appended. A bundle
that no table covers (`sm030001`, `ladies_biker_ar`) keeps its own name.
`--list` prints both names, and the CLI accepts either. Internally the exporter
works with the bundle name: it names the prefabs to read and the prefix to
strip from clip names.

**What counts as the character.** A character's bundles also carry every
effect it can spawn, and many `Model/` prefabs are props the character never
wears (Cherino's breakable walls, Neru's drones). So the exporter reads only
the prefab `<name>_Mesh` and the halo prefabs (`<name>_Halo`, or another
`*_Halo` prefab under `Model/`). If there is no `<name>_Mesh`, it takes the
`Model/` prefab with the most skinned renderers. Inactive GameObjects and
outline meshes are skipped. A few older characters keep only a loose weapon in
their own bundles; their full model is read from the preload `_mxprolog`
bundles.

A clip is kept only if at least half of its transform bindings resolve on the
character's Avatar. This drops clips for enemies, effects and cut-in props that
share bone names with the character. EX cut-in clips are kept only when named
`Exs_Cutin` or `Exs_Cutin_<n>`. A clip whose curves move nothing exported (a
camera track, for example) is dropped.

**Cafe bodies.** Saori (Dress) and Toki use a separate cafe body,
`<name>_CafeOnly_Mesh`, with its own Avatar and rest pose. Cafe clips played on
the main body distort it, so these characters get a second model,
`<label> (Cafe).glb`. Both runtime controllers list the cafe clips, so each
clip goes to the rig whose Avatar resolves more of its bindings; ties go to the
main body. Niko, Kurumi and Otogi also have a `_CafeOnly` prefab, but it drives
the main Avatar, so they export as one model. Their cafe prefab orders its
children differently, so renderer events in `Cafe_` clips (see *Switching
parts*) are resolved against the cafe prefab's children.

**The halo.** The runtime character prefab (`character-<name>-_mxload-*`
bundles) has a `HaloRoot` subtree with the halo mesh, material and placement.
The exporter imports only that subtree and follows its references into other
bundles, so costumes can reuse their base character's halo. When `HaloRoot` is
present, halo copies in the model prefabs are skipped. The halo's follower
component gives a `FollowTarget` bone and a `TargetRelativePosition` /
`TargetRelativeRotation`; the halo is attached to that bone with that offset.
Without a follower, the halo is attached to `Bip001 Head` and keeps its rest
placement. Without a renderable `HaloRoot`, the halo objects in the model
prefabs are used.

Mika's halo trail (`trail1`, `trail2`) is a particle system, not a mesh. Each
emitter draws a single mesh particle that stays on its emitter all the time.
A `ParticleSystemRenderer` that matches this is exported as a fixed mesh,
scaled by the particle's start size. The rules: mesh render mode, local
alignment, at most one particle, no start speed or start rotation, a lifetime
that outlasts the loop, and no modules except Initial, Emission, Rotation and
CustomData. Its DSFX shader has no `_MainTex`. The exporter bakes `_Tex_Main`,
tinted by `_Color`, with its alpha multiplied by `_Tex_Mask` (R times A). Each
texture keeps its saved tiling and offset. An HDR `_Color` keeps its hue but
is scaled down to 1, because clipping it turns the gradient white. Other
particles (sparkles, drips) are not exported.

**Parts only the runtime prefab has.** The runtime prefab adds a few top-level
parts that the model prefab lacks: Ibuki's second face, Koharu's and Takane's
weapons, Momoi's and Midori's gamepad, Hanae (Christmas)'s sack. It draws them
with `OL_` meshes. An `OL_` mesh has the same vertices, UVs, bones and triangles
as its original, with the outline normals in the tangents. The exporter imports
these meshes without the `OL_` prefix, together with their transforms and
bones from the runtime prefab. Where a runtime path also exists in the model
prefab, the model prefab's node stands in for it. Clips bind the new transforms
by path hash (`crc32` of the path below the root), so their curves are kept
(the gamepad's `Bone_GamePad`).

A runtime part whose bounds match an exported part's within 1% of its size,
where the exported part's name is absent from the runtime prefab, is a renamed
copy (Nodoka (Hot Spring)'s body) or a remade copy (Yuzu (Battle)'s cockpit).
It is not imported; renderer events that name it switch the exported part.

Inactive runtime parts are skipped. Parts that only the model prefab has (the
drone of Shiroko's EX) are kept.

**Meshes and the skeleton.** Vertices stay in mesh space, because Unity's
`m_BindPose` matrices are the inverse bind matrices glTF wants. The Transform
hierarchy becomes glTF nodes, with one skin per skinned renderer. To convert
from Unity's left-handed space, the exporter mirrors X, reverses triangle
winding, negates `y`/`z` of every quaternion and conjugates matrices as
`FLIP @ M @ FLIP`. `_MainTex` is embedded as PNG; materials without one use
`_Tint` or `_Color` as a flat colour.

Rigidly bound meshes (weapons, hand props: every vertex on one bone at full
weight) have no BlendWeight channel, only BlendIndices. The exporter supplies
the implied weight of 1.0; otherwise these meshes stay at the bind pose while
the hand moves. Compressed meshes can return junk in the implied last weight
slot; the exporter replaces it with the remaining weight. glTF allows four
influences per vertex, so wider vertices keep their four heaviest.

**Transparency.** `_MainTex` alpha is not always opacity.
`MX/C-General/Layer4`, the usual body shader, stores a layer selector there,
and the weapon and halo shaders store other data. So the exporter reads the
shader's first pass. Most passes write `Blend [_SrcBlend] [_DstBlend]`, which
serialises as 0/0, so the material's `_SrcBlend`/`_DstBlend`/`_ZWrite` decide.

- SrcAlpha / OneMinusSrcAlpha with depth write (eyes, mouth): `MASK`.
- SrcAlpha / OneMinusSrcAlpha without depth write: `BLEND`.
- `_AlphaClip` set: `MASK` at `_Cutoff`.
- Other blending with `RenderType` `Transparent`: `BLEND`.
- Everything else: `OPAQUE`.

**Bundles outside the character's own set.** Materials, textures and the mouth
atlas often live in other bundles: another character's, the shared shaders, or
the preload group. Bundles name dependencies by serialized file, not by bundle.
So, the first time a pointer leaves the loaded bundles, the exporter reads the
header of every bundle (main and preload directories) to map file to bundle,
then loads the needed bundle as a dependency. Dependencies do not add their own
prefabs or clips to the export.

**Faces.** The face texture has eyes but no mouth. The mouth is a quad whose
UVs sit in a small corner window; the shader maps that window to one tile of an
expression atlas named by the material's `_MouthTileTex`. Most characters share
`Character_Mouth`, an 8x8 grid; a few have their own atlas and grid
(`_MouthTileCols`, `_MouthTileRows`). The exporter finds the quad by its UV
window, crops the tiles it needs and gives the quad its own primitive and
material. The quad is either part of the eye submesh or a submesh of its own.

The default tile is `uv * scale + offset` read back from `_MouthTileTex`. An
offset past 1 wraps around (Ui's 1.625 is column 5 of 8, not 13), and a
negative scale mirrors the axis, which puts the offset on the far edge of the
tile, so the tile is one cell back. A horizontal mirror counts as part of the
tile (see below): Reisa's material is saved with tile 704 flipped.

**Expressions.** Clips change the mouth with `AnimationEvent`s:
`SetMouthTile(row * 100 + col)`, with rows counted from the bottom of the atlas,
`SetHorizontallyFlippedMouthTile(row * 100 + col)` for the same tile mirrored
left to right, or `SetMouthTileToDefault()` for the material's own tile. A
flipped tile is cropped and mirrored as a tile of its own (`Mouth_602_Flipped`).
glTF cannot animate a UV offset. So the exporter crops every tile the clips use
and gives each one its own copy of the mouth quad. Each copy has one morph
target that collapses it to a point. A clip shows a tile by holding that copy's
weight at 0 and all the others at 1, with `STEP` interpolation. The copies keep
their skinning, because some characters rig the mouth to its own `bone_mouth`.
A character can have more than one mouth: Shigure (Hot Spring) has one on her
body and one on her cut-in body, and Tsukuyo's tree has its own (driven
through `CharacterMultiMouthEventReceiver`). Each mouth gets its own set of
copies (`Mouth_*`, `Mouth2_*`, ...), and every set follows the same event
track. A set is folded away while the mesh it came off is hidden.

**Spare faces.** Some characters have extra face meshes for brief expressions
(Kasumi's star eyes, Aoba's blank stare, Neru's joke glasses). They ship enabled
beside the everyday face, and the clips switch them. A face mesh is one that
uses a `*_Face` material and no `*_Body` or `*_Hair` material. The everyday face
is the one with the `*_EyeMouth` material, since the clips lip-sync that mouth;
otherwise the one whose name has no number after "face". When a character has
two or more faces, each gets the collapsing morph target, and every face except
the worn one is held at weight 1. The worn face is the one a clip leaves on by
itself. If that reading of `Cafe_Reaction` does not give the everyday face,
the renderer events are ignored for that character and only the curves are
used. A single face mesh is switched like any other part (Tsukuyo's closed-eye
face).

**Switching parts.** Clips also switch other parts: Toki's two bodies, the
body swaps of Hoshino (Swimsuit) and Arisu (Battle), Megu's weapon in the cafe,
Makoto's spare hairdo. A clip switches an object in one of two ways:

- A curve on the GameObject's active flag (`attribute` 2086281974,
  `crc32("m_IsActive")`) or the renderer's `enabled` flag (renderer `typeID`,
  `attribute` 3305885265, `crc32("m_Enabled")`). The binding names its target by
  path.
- An `AniEvt_EnableChildRenderer(i)` / `AniEvt_DisableChildRenderer(i)` event.
  It flips the renderer of the `i`th child of the runtime prefab root
  (`assets/_mx/addressableasset/character/<name>/<name>.prefab`), not the model
  prefab. The runtime prefab orders its children differently, so the exporter
  matches each child back by name. Children with no renderer (`HaloRoot`,
  `bone_root`) make the event do nothing.

The runtime prefab also stores each renderer's starting `enabled` flag; a few
start off (Toki's second body, Kisaki's weapon and table, Yuzu (Battle)'s
prop). An object's rest state is its state at the start of `Cafe_Reaction`,
starting from that flag. Every mesh under a switched object gets the collapsing
morph target. Each clip has a `STEP` weight channel for it only where its state
differs from rest.

A collapsed skinned mesh still follows its bones, so it tears into shards when
the pose moves away from the bind pose. So a clip that keeps an object off for
its whole length also lists it under `hide` in the clip's glTF `extras`, and the
viewer hides it. The key is `switch<n>`, which the object's mesh nodes carry in
their `extras`. A switch whose subtree contains meshes it does not switch gets no
key, because hiding a node hides its subtree.

A clip that at some point switches off every skinned mesh except faces and
mouth makes the whole character vanish (Izuna's and Haruka (New Year)'s EX).
Collapsing cannot do that cleanly, so such clips keep every switch at rest.
Objects the runtime prefab stores as inactive GameObjects (Hoshino's shield)
are left to prop detection below.

**Props the pose does not use.** A rig carries props the character holds only
in some clips: Hoshino's shield in her EX, Cherino's tank in one cut-in, Hanae's
rabbit. A clip that does not use a prop leaves its bones at rest, which puts the
prop wherever it was modelled. A clip binds every transform of the arrangement
it was written for, including still ones. So a mesh on the rig that the default
clip (`Cafe_Reaction`, else the first clip) binds nothing of is a prop, and the
clips that bind it are the clips that show it.

Props get `{"optional": true, "prop<n>": true}` in their node's `extras`.
`data-hide-parts="optional"` hides them, and a clip that uses a prop lists its
`prop<n>` group under `show` in its own `extras`. A group is the whole subtree
the prop hangs under, climbing until the next parent would include something
the default clip binds. Props that share nodes share a group, so a multi-mesh
prop (Hanae's rabbit and its eyes) comes and goes together.

A few meshes hang off the model root with no bone above them. For these, a
binding that only holds the mesh still means nothing, so the exporter uses the
clips that actually animate it. Such a mesh that rests on the floor (`y = 0`,
within 0.02) counts as set dressing and is never a prop (Yoshimi's amplifier).
The halo is never a prop, and neither is anything under it, such as Mika's
trail meshes.

**Animations.** The clips are Mecanim. Curves are packed in
`m_MuscleClip.m_Clip` as three concatenated arrays: streamed (sparse keys),
dense (evenly sampled) and constant. `m_ClipBindingConstant` says which
property each run of curves belongs to. The exporter decodes the streamed data
itself, since UnityPy has no helper, and resolves binding path hashes to bones
through the Avatar's `m_TOS` table.

Each streamed key stores the cubic the curve follows until the next key. Those
channels are written as `CUBICSPLINE` with the cubic's slopes as tangents, so
they play back exactly; linear keys made tails, hair and straps twitch. A cubic
can jump at a key, which glTF cannot, so a jump gets a second key one float step
later. Dense curves and channels with only straight segments are `LINEAR`.

A rotation is keyed as a quaternion (binding attribute 2) or as Euler angles in
degrees (attribute 4). For Euler bindings, `customType` is Unity's
`RotationOrder`: 0 (XYZ) for curves kept from the FBX, 4 (ZXY) for Unity's
own. The 3ds Max rigs break if every clip is read as ZXY. Euler curves are
sampled to quaternions, with extra keys wherever an angle moves more than 10°
between keys.

## Known gaps

- Euler-keyed rotations become `LINEAR` quaternion keys, not cubics.
- Channels that hold still at the node's rest transform are dropped; keeping
  them roughly triples the file size. The viewer restores unbound nodes to rest; a
  viewer that does not would need them.
- Clip names keep the game's naming (`Formation_Idle`, `Exs`, `Vital_Death`);
  only the character prefix is removed.
- Expressions follow each clip's event track. During voiced lines the game
  uses a lipsync layer driven by voice data, which is not in the model bundles.
- Apart from the mouth atlas, only `_MainTex` is used. The game's toon shader
  also uses mask and specular textures, so shading is flat;
  `data-shading="unlit"` is the closest cheap match.
- The halo is attached rigidly to its bone. In the cafe and the world raid map,
  the game damps the halo so it lags the head; glTF cannot express that.
- Outlines are not exported; the gadget's `outline` option replaces them. Parts
  the model prefab has are exported from their original meshes, not their
  `OL_` copies.
- Prop detection is inferred from clip bindings and, for meshes with no bone,
  from floor contact. A prop modelled at floor level is always kept.
- Mika's halo trail does not move. In the game it spins slowly, about 3° a
  second, and its mask scrolls by particle custom data. The halo's sparkle
  particles are missing.
- Props the game spawns on separate rigs (Shiroko＊Terror's pufferfish,
  Cherino's capybara, Utaha's robots) are effects and are not exported.
