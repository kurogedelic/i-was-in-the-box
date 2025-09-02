#!/usr/bin/env python3
"""
Cleanup helper to remove generated artifacts and temporary files.

Dry-run by default; pass --apply to actually delete.

Examples:
  - Preview deletions:  python scripts/cleanup.py
  - Delete audio/mix/backup only: python scripts/cleanup.py --apply
  - Delete also open_* CSVs and ID lists: python scripts/cleanup.py --apply --all
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def find_paths(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.exists():
                paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove generated files (dry-run by default)")
    ap.add_argument("--apply", action="store_true", help="Actually delete files")
    ap.add_argument("--all", action="store_true", help="Also remove open_* CSVs and *_ids.txt")
    ap.add_argument("--prune-dirs", action="store_true", help="Remove empty audio/mix dirs if empty")
    args = ap.parse_args()

    to_remove: List[Path] = []

    # Always consider these safe to remove
    patterns = [
        "audio/*.wav",
        "mix/*.wav",
        "data/*_backup.csv",
        "data/group2_*.csv",
        "**/.DS_Store",
        "scripts/__pycache__",
        "__pycache__",
    ]

    # Optionally remove data CSVs and ID lists
    if args.all:
        patterns += [
            "data/open_tle_data.csv",
            "data/open_positions.csv",
            "data/open_launch_timeline.csv",
            "data/*_ids.txt",
        ]

    # Collect paths
    for p in find_paths(patterns):
        to_remove.append(p)

    # Unique + stable order
    seen = set()
    uniq: List[Path] = []
    for p in sorted(to_remove):
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if not uniq:
        print("No targets found. Nothing to do.")
        return 0

    print("Cleanup targets:")
    for p in uniq:
        print(f"  - {p.relative_to(ROOT)}")

    if not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return 0

    # Delete
    for p in uniq:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            print(f"Failed to remove {p}: {e}")

    if args.prune_dirs:
        for d in [ROOT / "audio", ROOT / "mix", ROOT / "scripts/__pycache__"]:
            try:
                if d.exists() and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

