"""Export Blue Archive character models from the game's asset bundles to .glb.

Reads the Unity asset bundles shipped with the Steam build and writes one
self-contained glTF binary per character: skinned meshes with embedded
textures, the bone hierarchy, and every animation clip found alongside them.
Each file is named for the wiki's name for the character rather than the
bundle's dev name, through the wiki repo's devname_map tables and the game's
costume table.

Usage:
    python export_models.py --list
    python export_models.py airi_original aris_original
    python export_models.py "Aru (New Year)"
    python export_models.py --all
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from exporter.bundles import DEFAULT_BUNDLE_DIR, list_characters, model_names
from exporter.extract import export

HERE = Path(__file__).resolve().parent


def export_one(bundle_dir: Path, name: str, out: Path, animations: bool,
               overwrite: bool) -> Path | None:
    print(f"{name} -> {out.name}", flush=True)
    try:
        return export(bundle_dir, name, out, animations, overwrite)
    except Exception as e:  # a broken bundle should not stop the batch
        print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("characters", nargs="*",
                        help="bundle names (airi_original) or wiki names (Aru (New Year))")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, default=HERE / "models")
    parser.add_argument("--list", action="store_true", help="list available characters and exit")
    parser.add_argument("--all", action="store_true", help="export every character")
    parser.add_argument("--no-animations", action="store_true", help="export the bind pose only")
    parser.add_argument("--overwrite", action="store_true", help="replace existing model files")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 4),
                        help="concurrent export jobs (default: hardware threads minus 4, minimum 1)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    available = list_characters(args.bundle_dir)
    # the bundle name stays the handle the export works from: it names the
    # prefabs to walk and the prefix to strip off clip names
    names = model_names(available)
    if args.list:
        width = max(map(len, available), default=0)
        print("\n".join(f"{b:<{width}}  {names[b]}" for b in available))
        return

    by_wiki_name = {wiki.lower(): bundle for bundle, wiki in names.items()}
    targets = list(dict.fromkeys(
        available if args.all else
        [by_wiki_name.get(c.lower(), c) for c in args.characters]))
    if not targets:
        parser.error("give at least one character name, or --all / --list")

    args.out.mkdir(parents=True, exist_ok=True)
    exported = []
    skipped = []
    pending = []
    for name in targets:
        out = args.out / f"{names.get(name, name)}.glb"
        if out.exists() and not args.overwrite:
            print(f"  skipped {out.name}: already exists")
            skipped.append(out.name)
        else:
            pending.append((name, out))

    jobs = min(args.jobs, len(pending))
    if jobs == 1:
        for name, out in pending:
            out = export_one(args.bundle_dir, name, out, not args.no_animations,
                             args.overwrite)
            if out is not None:
                exported.append(out.name)
    elif jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(export_one, args.bundle_dir, name, out,
                            not args.no_animations, args.overwrite): name
                for name, out in pending
            }
            for future in as_completed(futures):
                try:
                    out = future.result()
                except Exception as e:
                    print(f"  {futures[future]} failed: {type(e).__name__}: {e}")
                    continue
                if out is not None:
                    exported.append(out.name)

    # the mtime is a cache-busting token: the viewer appends it to the URL
    manifest = args.out / "models.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else None
    models = {p.name: int(p.stat().st_mtime) for p in sorted(args.out.glob("*.glb"))}
    if models != existing:
        manifest.write_text(json.dumps(models, indent=2))
    print(f"\n{len(exported)} model(s) exported, {len(skipped)} skipped; manifest at {manifest}")


if __name__ == "__main__":
    main()
