#!/usr/bin/env python3
"""
Visualize satellites orbiting around an Earth-like circle (2D).

Reads data/open_tle_data.csv and animates points moving around a center
circle with radius scaled by altitude. Saves MP4 (ffmpeg) or GIF (Pillow)
depending on availability.

Examples:
  - MP4 (if ffmpeg available):
      python scripts/visualize_orbits.py --duration 20 --fps 30 --out viz/orbits.mp4
  - GIF fallback (no ffmpeg):
      python scripts/visualize_orbits.py --duration 10 --fps 20 --out viz/orbits.gif

Notes:
  - Angular speed is derived from mean_motion (rev/day).
  - Initial angle seeds use RAAN to provide variety.
  - For large N, consider --sat-limit to keep it responsive.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; enable --show to override
import matplotlib.pyplot as plt
from matplotlib import animation


SECONDS_PER_DAY = 86400.0


@dataclass
class VizConfig:
    width: int = 900
    height: int = 900
    fps: int = 30
    duration: float = 20.0
    bg: str = "#0b0f1a"
    earth_color: str = "#1b3b6f"
    orbit_color: str = "#234c8a"
    sat_color: str = "#e8f1ff"
    sat_alpha: float = 0.9
    earth_radius_px: float = 160.0
    min_alt_km: float | None = None
    max_alt_km: float | None = None
    sat_limit: int | None = 600
    show: bool = False


def load_satellites(csv_path: str, sat_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Sanity filters
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_motion", "altitude_km"])  # essential fields
    # Keep reasonable mean motion range (LEO/MEO-ish)
    df = df[(df["mean_motion"] > 0.1) & (df["mean_motion"] < 25.0)]
    if sat_limit is not None and len(df) > sat_limit:
        # Prefer a spread across altitude: stratified sample by quantiles
        df = (df
              .assign(_q=pd.qcut(df["altitude_km"], q=20, duplicates="drop"))
              .groupby("_q", group_keys=False)
              .head(max(1, sat_limit // 20))
              .drop(columns=["_q"]))
        if len(df) > sat_limit:
            df = df.sample(sat_limit, random_state=42)
    df = df.reset_index(drop=True)
    return df


def build_orbit_params(df: pd.DataFrame, cfg: VizConfig):
    # Altitude scaling
    alt_min = float(cfg.min_alt_km if cfg.min_alt_km is not None else max(200.0, df["altitude_km"].min()))
    alt_max = float(cfg.max_alt_km if cfg.max_alt_km is not None else df["altitude_km"].max())
    if alt_max <= alt_min:
        alt_max = alt_min + 1.0

    # Visual radius mapping: Earth disc + band for altitude
    band_px = min(cfg.width, cfg.height) * 0.5 - cfg.earth_radius_px - 20.0
    def r_of_alt(alt):
        return cfg.earth_radius_px + (float(alt) - alt_min) / (alt_max - alt_min) * band_px

    radii = df["altitude_km"].map(r_of_alt).to_numpy(dtype=np.float32)

    # Angular speed (rad/s) from mean_motion (rev/day)
    omegas = (df["mean_motion"].to_numpy(dtype=np.float64) * 2.0 * math.pi) / SECONDS_PER_DAY

    # Initial phase: use RAAN if present; otherwise derive from hash of ID for spread
    if "raan" in df.columns:
        phases0 = np.deg2rad(df["raan"].fillna(0.0).to_numpy(dtype=np.float64))
    else:
        # fallback: hash-based pseudo angle
        seeds = df["satellite_id"].astype(str).apply(lambda s: (hash(s) % 360) / 180.0 * math.pi)
        phases0 = seeds.to_numpy(dtype=np.float64)

    # Small random jitter to avoid perfect alignment
    rng = np.random.default_rng(123)
    phases0 = (phases0 + rng.uniform(-0.05, 0.05, size=len(phases0))) % (2 * math.pi)

    return radii, omegas, phases0, (alt_min, alt_max)


def animate_orbits(df: pd.DataFrame, cfg: VizConfig, out_path: str) -> None:
    n = len(df)
    total_frames = int(cfg.duration * cfg.fps)
    radii, omegas, phases0, alt_range = build_orbit_params(df, cfg)

    # Figure setup
    dpi = 100
    fig_w = cfg.width / dpi
    fig_h = cfg.height / dpi
    if cfg.show:
        # If showing, allow default backend (potentially interactive)
        import matplotlib
        matplotlib.use(matplotlib.get_backend())
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=cfg.bg)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor(cfg.bg)
    ax.set_xlim(-cfg.width/2, cfg.width/2)
    ax.set_ylim(-cfg.height/2, cfg.height/2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw Earth circle
    earth = plt.Circle((0, 0), radius=cfg.earth_radius_px, color=cfg.earth_color, zorder=1)
    ax.add_patch(earth)

    # Optional orbit band guide (min/max)
    band_min = plt.Circle((0, 0), radius=cfg.earth_radius_px, fill=False, lw=0.5, ec=cfg.orbit_color, alpha=0.4, zorder=2)
    band_max = plt.Circle((0, 0), radius=cfg.earth_radius_px + (min(cfg.width, cfg.height) * 0.5 - cfg.earth_radius_px - 20.0), fill=False, lw=0.5, ec=cfg.orbit_color, alpha=0.2, zorder=2)
    ax.add_patch(band_min)
    ax.add_patch(band_max)

    # Scatter for satellites
    # Start positions
    theta0 = phases0
    x = radii * np.cos(theta0)
    y = radii * np.sin(theta0)
    scat = ax.scatter(x, y, s=6, c=cfg.sat_color, alpha=cfg.sat_alpha, zorder=3, linewidths=0)

    title = ax.text(0, cfg.height*0.5*0.90 - cfg.earth_radius_px*0.0, "Satellite Orbits", color="#cfe0ff", ha='center', va='center', fontsize=12)
    subtitle = ax.text(0, -cfg.height*0.5*0.90 + cfg.earth_radius_px*0.0,
                       f"N={n} | alt {alt_range[0]:.0f}–{alt_range[1]:.0f} km | fps {cfg.fps}",
                       color="#8fb3ff", ha='center', va='center', fontsize=9)

    # Animation function
    dt = 1.0 / cfg.fps
    def update(frame_idx):
        t = frame_idx * dt
        theta = (phases0 + omegas * t) % (2 * math.pi)
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)
        scat.set_offsets(np.c_[x, y])
        return scat,

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000.0/cfg.fps, blit=True)

    # Ensure output dir
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Save with available writer
    ext = os.path.splitext(out_path)[1].lower()
    saved = False
    if ext in (".mp4", ".m4v"):
        try:
            writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=4000)
            anim.save(out_path, writer=writer, dpi=dpi, savefig_kwargs={'facecolor': cfg.bg})
            saved = True
        except Exception as e:
            print(f"FFmpeg not available or failed ({e}); will try GIF.")
            out_path = os.path.splitext(out_path)[0] + ".gif"

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=min(cfg.fps, 25))
            anim.save(out_path, writer=writer, dpi=dpi, savefig_kwargs={'facecolor': cfg.bg})
            saved = True
        except Exception as e:
            print(f"Pillow GIF writer failed: {e}")
            raise SystemExit("Failed to save animation (no suitable writer available). Install ffmpeg or pillow.")

    print(f"Saved visualization: {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Animate satellites around an Earth-like circle (2D)")
    ap.add_argument("--csv", default=os.path.join("data", "open_tle_data.csv"), help="Path to open_tle_data.csv")
    ap.add_argument("--out", default=os.path.join("viz", "orbits.mp4"), help="Output path (.mp4 or .gif)")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--duration", type=float, default=20.0, help="Duration seconds")
    ap.add_argument("--size", type=int, default=900, help="Canvas size (square px)")
    ap.add_argument("--earth-radius", type=float, default=160.0, help="Earth disc radius in pixels")
    ap.add_argument("--sat-limit", type=int, default=600, help="Limit number of satellites for performance (0=all)")
    ap.add_argument("--min-alt", type=float, default=None, help="Override min altitude km for scaling")
    ap.add_argument("--max-alt", type=float, default=None, help="Override max altitude km for scaling")
    ap.add_argument("--show", action="store_true", help="Try to show window (if backend allows)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    sat_limit = None if (args.sat_limit is not None and args.sat_limit <= 0) else args.sat_limit
    df = load_satellites(args.csv, sat_limit=sat_limit)
    if df.empty:
        print("No satellites to visualize (empty dataframe)")
        return 1

    cfg = VizConfig(
        width=args.size, height=args.size, fps=args.fps, duration=args.duration,
        earth_radius_px=args.earth_radius,
        min_alt_km=args.min_alt, max_alt_km=args.max_alt,
        sat_limit=sat_limit, show=args.show,
    )

    animate_orbits(df, cfg, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

