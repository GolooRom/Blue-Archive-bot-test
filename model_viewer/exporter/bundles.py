"""Finding the game's asset bundles, and naming the models they hold.

Bundles name their dependencies by serialized file, so a header index says
which bundle holds which file.  Each model is named for the wiki's name for
the character rather than the bundle's dev name, through the wiki repo's
devname_map tables and the game's costume table.
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections import Counter
from pathlib import Path

from UnityPy.enums import ArchiveFlags, CompressionFlags
from UnityPy.environment import simplify_name
from UnityPy.helpers import CompressionHelper

REPO = Path(__file__).resolve().parents[2]

DEFAULT_BUNDLE_DIR = (
    REPO / "vendor" / "BlueArchive" / "BlueArchive_Data" / "StreamingAssets"
    / "PUB" / "Resource" / "GameData" / "Windows"
)

CHARACTER_BUNDLE_RE = re.compile(r"^assets-_mx-characters-(.+?)-_mxdependency-")

# The wiki's devname_map tables, kept current by update.py, give the wiki's
# name for each bundle's dev character code.
DEVNAME_MAPS = (REPO / "json" / "devname_map.json",
                REPO / "json" / "devname_map_aux.json")
# Some bundles are named for a model prefab code the maps lack (`ch0061`); the
# costume table ties each prefab to costume dev names the maps do know.
COSTUME_TABLE = REPO / "json" / "CostumeExcelTable.json"
QUALIFIERS = {"cutin": "Cut-in", "nonweapon": "No Weapon", "scenario": "Scenario",
              "carrier": "Carrier"}


def bundle_cabs(path: Path) -> list[str]:
    """The names of the serialized files a bundle holds, read from its header
    metadata rather than by decompressing the whole file."""
    with path.open("rb") as handle:
        head = handle.read(4096)
        end = head.index(b"\0")
        if head[:end] != b"UnityFS":
            return []
        offset = end + 1
        version = struct.unpack_from(">I", head, offset)[0]
        offset += 4
        for _ in range(2):  # the player and engine version strings
            offset = head.index(b"\0", offset) + 1
        _size, compressed, uncompressed, flags = struct.unpack_from(">qIII", head, offset)
        offset += 20
        if version >= 7:
            offset = (offset + 15) & ~15
        if flags & ArchiveFlags.BlocksInfoAtTheEnd:
            handle.seek(-compressed, os.SEEK_END)
            blob = handle.read(compressed)
        else:
            if offset + compressed > len(head):
                handle.seek(0)
                head = handle.read(offset + compressed)
            blob = head[offset:offset + compressed]
    compression = CompressionFlags(flags & ArchiveFlags.CompressionTypeMask)
    blob = CompressionHelper.DECOMPRESSION_MAP[compression](blob, uncompressed)

    offset = 16  # the hash of the uncompressed data
    blocks = struct.unpack_from(">i", blob, offset)[0]
    offset += 4 + blocks * 10  # (uncompressed size, compressed size, flags) each
    nodes = struct.unpack_from(">i", blob, offset)[0]
    offset += 4
    names = []
    for _ in range(nodes):
        offset += 20  # (offset, size, flags)
        end = blob.index(b"\0", offset)
        names.append(blob[offset:end].decode("utf-8", "replace"))
        offset = end + 1
    return names


_cab_index: dict[str, str] | None = None


def cab_index(bundle_dir: Path) -> dict[str, str]:
    """Which bundle holds which serialized file, built once per process."""
    global _cab_index
    if _cab_index is None:
        _cab_index = {}
        preload = bundle_dir.parent.parent / "Preload" / bundle_dir.name
        for directory in (bundle_dir, preload):
            for bundle in sorted(directory.glob("*.bundle")):
                try:
                    names = bundle_cabs(bundle)
                except Exception as e:
                    print(f"  skipped a bundle header: {bundle.name}: {type(e).__name__}: {e}")
                    continue
                for name in names:
                    _cab_index.setdefault(simplify_name(name), str(bundle))
    return _cab_index


def resolve_dependencies(env, bundle_dir: Path) -> None:
    """Let pointers follow into bundles beyond the character's own.

    Bundles name their dependencies by serialized file rather than by bundle,
    so the index says which bundle to open the moment a pointer needs it.  They
    are loaded as dependencies, which keeps their own prefabs, clips and
    avatars out of `env.objects`.
    """
    def find_file(name: str, is_dependency: bool = True):
        cab = env.get_cab(name)
        if cab is not None:
            return cab
        bundle = cab_index(bundle_dir).get(simplify_name(name))
        if bundle is None:
            raise FileNotFoundError(f"{name} is in no bundle under {bundle_dir}")
        env.load_file(bundle, is_dependency=True)
        cab = env.get_cab(name)
        if cab is None:
            raise FileNotFoundError(f"{name} not found in {Path(bundle).name}")
        return cab

    env.find_file = find_file


def list_characters(bundle_dir: Path) -> list[str]:
    names = set()
    for bundle in bundle_dir.glob("assets-_mx-characters-*.bundle"):
        match = CHARACTER_BUNDLE_RE.match(bundle.name)
        if match:
            names.add(match.group(1))
    return sorted(names)


def bundles_for(bundle_dir: Path, name: str) -> list[Path]:
    return sorted(bundle_dir.glob(f"assets-_mx-characters-{name}-_mxdependency-*.bundle"))


def prolog_bundles_for(bundle_dir: Path, name: str) -> list[Path]:
    """Preload bundles that can carry a legacy character's full Model prefab."""
    preload_dir = bundle_dir.parent.parent / "Preload" / bundle_dir.name
    return sorted(preload_dir.glob(
        f"prologdepengroup-assets-_mx-characters-{name}-_mxprolog-*.bundle"))


def read_devname_map() -> dict[str, str]:
    """Bundle name (lowercased) -> the wiki's name for that character."""
    names = {}
    for path in DEVNAME_MAPS:
        if not path.exists():
            print(f"  no {path.name}; run update.py to fetch it")
            continue
        for dev, entry in json.loads(path.read_text(encoding="utf-8")).items():
            variant = entry["variant"]
            names[dev.lower()] = entry["firstname"] + (f" ({variant})" if variant else "")
    if not COSTUME_TABLE.exists():
        print(f"  no {COSTUME_TABLE.name}; run update.py to fetch it")
        return names
    costumes = json.loads(COSTUME_TABLE.read_text(encoding="utf-8"))["DataList"]
    for costume in costumes:
        prefab = costume["ModelPrefabName"].lower()
        name = names.get(costume["DevName"].lower().removesuffix("_default"))
        # NPC and event copies share the prefab; the first costume the maps know wins
        if prefab and name:
            names.setdefault(prefab, name)
    return names


def qualifier(tail: str) -> str:
    """`carrier2`, `scenario_01` -> `Carrier 2`, `Scenario 1`."""
    parts = []
    for token in tail.split("_"):
        match = re.fullmatch(r"([a-z]*)([0-9]*)", token)
        if match is None:
            parts.append(token)
            continue
        word, digits = match.groups()
        label = QUALIFIERS.get(word, word.capitalize())
        parts.append(f"{label} {int(digits)}".strip() if digits else label)
    return " ".join(parts)


def model_names(bundles: list[str]) -> dict[str, str]:
    """Bundle name -> the file name its model is written under, minus `.glb`.

    `aru_newyear` becomes `Aru (New Year)`. A bundle whose name the map covers
    only up to a suffix carries that suffix as a second parenthesis —
    `ch0334_carrier` is `Arisu (Battle) (Carrier)` — and one the map does not
    know at all keeps its own name.  Bundles that land on the same name get
    their own trailing number appended.
    """
    devnames = read_devname_map()
    names = {}
    for bundle in bundles:
        stem = bundle.removesuffix("_original").split("_")
        names[bundle] = bundle
        for i in range(len(stem), 0, -1):
            name = devnames.get("_".join(stem[:i]))
            if name:
                tail = "_".join(stem[i:])
                names[bundle] = f"{name} ({qualifier(tail)})" if tail else name
                break
    # two bundles can share a name; their own trailing number tells them apart
    taken = Counter(names.values())
    for bundle, name in names.items():
        number = re.search(r"([0-9]+)$", bundle)
        if taken[name] > 1 and number:
            names[bundle] = f"{name} ({int(number.group(1))})"
    return names
