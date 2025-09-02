#!/usr/bin/env python3
"""
Export a minimal JSON for web visualization (p5.js/three.js).

Reads data/open_tle_data.csv and writes a compact satellites.json with:
  [{ id, name, altitude_km, inclination, raan, mean_motion }]

Usage:
  python scripts/export_satellite_json.py --out viz/p5/satellites.json --limit 800
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Export minimal satellite JSON for web viz")
    ap.add_argument("--csv", default=os.path.join("data", "open_tle_data.csv"))
    ap.add_argument("--timeline", default=os.path.join("data", "open_launch_timeline.csv"), help="Launch timeline CSV (satellite_id,launch_date)")
    ap.add_argument("--out", default=os.path.join("viz", "p5", "satellites.json"))
    ap.add_argument("--limit", type=int, default=800, help="Max satellites to include (0=all)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Try to join launch timeline if available
    launch_day_map = {}
    try:
        tl = pd.read_csv(args.timeline)
        if "launch_date" in tl.columns and "satellite_id" in tl.columns:
            # Parse date to ordinal days (float)
            ld = pd.to_datetime(tl["launch_date"], errors="coerce", utc=True)
            launch_day_map = dict(zip(tl["satellite_id"].astype(str), (ld.view('int64')/1_000_000_000/86400.0)))
    except Exception:
        pass
    # Keep essential fields, drop invalid
    cols = ["satellite_id", "name", "altitude_km", "inclination", "raan", "mean_motion"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["altitude_km", "mean_motion"]).copy()
    # Basic sanity ranges
    df = df[(df["mean_motion"] > 0.05) & (df["mean_motion"] < 25.0)]

    # Sample down if needed preserving altitude distribution
    limit = args.limit if args.limit and args.limit > 0 else None
    if limit is not None and len(df) > limit:
        df = (df
              .assign(_q=pd.qcut(df["altitude_km"], q=min(50, max(5, limit // 20)), duplicates="drop"))
              .groupby("_q", group_keys=False)
              .head(max(1, limit // 20))
              .drop(columns=["_q"]))
        if len(df) > limit:
            df = df.sample(limit, random_state=42)

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rec = {
            "id": str(r.get("satellite_id", "")),
            "name": str(r.get("name", "")),
            "altitude_km": float(r.get("altitude_km", 0.0)),
            "inclination": float(r.get("inclination", 0.0)) if not pd.isna(r.get("inclination")) else 0.0,
            "raan": float(r.get("raan", 0.0)) if not pd.isna(r.get("raan")) else 0.0,
            "mean_motion": float(r.get("mean_motion", 0.0)),
        }
        # Add launch_day if known
        ld = launch_day_map.get(rec["id"]) if launch_day_map else None
        if ld is not None and np.isfinite(ld):
            rec["launch_day"] = float(ld)
        out.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {len(out)} satellites -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
