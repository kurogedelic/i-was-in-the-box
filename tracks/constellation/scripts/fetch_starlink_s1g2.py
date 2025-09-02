#!/usr/bin/env python3
"""
Convert Starlink S1G2 TLE data into this project's CSV format.

Usage:
  python scripts/fetch_starlink_s1g2.py --tle data/starlink_s1g2.tle

Inputs:
  - TLE file with standard 3-line sets (name, L1, L2) for S1G2 satellites.

Outputs (in ../data relative to this script):
  - open_tle_data.csv
  - open_positions.csv (24h pseudo-positions)
  - open_launch_timeline.csv (launch_date left as default unless provided)

Notes:
  - No external network required. Parses TLE locally.
  - Altitude is derived from mean motion using standard orbit relations.
  - Position samples are simplified (same method used by fetch_open_data.py).
"""
import argparse
import math
import os
from datetime import datetime, timedelta
import csv
from typing import Iterable, Optional, Set


MU_EARTH_KM3_S2 = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
EARTH_RADIUS_KM = 6378.137
SECONDS_PER_DAY = 86400.0


def parse_tle_triplets(lines: Iterable[str]):
    """Yield (name, line1, line2) triplets from a TLE file."""
    buf = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if len(buf) == 3:
            # Basic validation
            name, l1, l2 = buf
            if not (l1.startswith('1 ') and l2.startswith('2 ')):
                # Try to realign if file doesn't include name lines
                # In that case, lines come in pairs (L1, L2). We treat name as object number.
                if len(buf) >= 2 and buf[0].startswith('1 ') and buf[1].startswith('2 '):
                    # shift to pairs
                    yield ("OBJECT-" + buf[0][2:7].strip(), buf[0], buf[1])
                    buf = []
                    continue
                # Otherwise, skip malformed block
                buf = []
                continue
            yield (name, l1, l2)
            buf = []


def tle_field_float(s, start, end, scale=1.0):
    """Extract a float from fixed columns; for eccentricity digits, use scale=1e-7."""
    return float(s[start:end].strip()) * scale


def mean_motion_to_altitude_km(n_rev_per_day):
    """Approximate mean altitude from mean motion (rev/day)."""
    n = n_rev_per_day / 86400.0 * 2.0 * math.pi  # rad/s (mean angular rate)
    a = (MU_EARTH_KM3_S2 / (n * n)) ** (1.0 / 3.0)  # semi-major axis (km)
    altitude = a - EARTH_RADIUS_KM
    return max(0.0, altitude)


def build_csvs_from_tle_source(lines: Iterable[str], allowed_ids: Optional[Set[int]] = None) -> None:
    # Prepare outputs
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    pos_rows = []
    timeline_rows = []

    for name, l1, l2 in parse_tle_triplets(lines):
            # NORAD catalog ID from cols 3-7 on L1 or L2 (both contain it)
            try:
                norad_id = int(l1[2:7])
            except ValueError:
                try:
                    norad_id = int(l2[2:7])
                except Exception:
                    continue

            if allowed_ids is not None and norad_id not in allowed_ids:
                continue

            # Inclination (deg) cols 9-16 on line 2
            inclination = tle_field_float(l2, 8, 16)
            # RAAN (deg) cols 18-25 on line 2
            raan = tle_field_float(l2, 17, 25)
            # Eccentricity (decimal implied) cols 27-33 on line 2
            eccentricity = tle_field_float(l2, 26, 33, scale=1e-7)
            # Mean motion (rev/day) cols 53-63 on line 2
            mean_motion = tle_field_float(l2, 52, 63)

            altitude_km = mean_motion_to_altitude_km(mean_motion)

            sat_row = {
                'satellite_id': f'NORAD-{norad_id}',
                'name': name,
                'altitude_km': altitude_km,
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'mean_motion': mean_motion,
                'data_source': 'STARLINK_S1G2_TLE'
            }
            rows.append(sat_row)

    if not rows:
        raise RuntimeError('No TLE objects parsed. Check the input file.')

    # Build 24h pseudo-positions like existing pipeline
    now = datetime.utcnow().replace(microsecond=0)
    for sat in rows:
        mm = sat['mean_motion'] if sat['mean_motion'] else 15.0
        orbital_period_hours = 24.0 / mm
        for hour in range(24):
            phase = (hour / orbital_period_hours) * 2.0 * math.pi
            lat = sat['inclination'] * math.sin(phase)
            lon = (sat['raan'] + hour * 15.0) % 360.0 - 180.0
            pos_rows.append({
                'satellite_id': sat['satellite_id'],
                'timestamp': (now + timedelta(hours=hour)).isoformat() + 'Z',
                'latitude': lat,
                'longitude': lon,
                'altitude_km': sat['altitude_km'],
            })

    # Timeline rows (launch_date unknown here → default)
    default_date = '2020-01-01'
    for sat in rows:
        timeline_rows.append({
            'satellite_id': sat['satellite_id'],
            'launch_date': default_date,
            'data_source': sat['data_source'],
        })

    # Save CSVs without pandas
    tle_cols = ['satellite_id', 'name', 'altitude_km', 'inclination', 'raan', 'eccentricity', 'mean_motion', 'data_source']
    with open(os.path.join(out_dir, 'open_tle_data.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=tle_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in tle_cols})

    pos_cols = ['satellite_id', 'timestamp', 'latitude', 'longitude', 'altitude_km']
    with open(os.path.join(out_dir, 'open_positions.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=pos_cols)
        w.writeheader()
        for r in pos_rows:
            w.writerow({k: r.get(k, '') for k in pos_cols})

    tl_cols = ['satellite_id', 'launch_date', 'data_source']
    with open(os.path.join(out_dir, 'open_launch_timeline.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=tl_cols)
        w.writeheader()
        for r in timeline_rows:
            w.writerow({k: r.get(k, '') for k in tl_cols})

    print('Wrote:')
    print('  data/open_tle_data.csv')
    print('  data/open_positions.csv')
    print('  data/open_launch_timeline.csv')


def load_id_filter(path: str) -> Set[int]:
    ids: Set[int] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.replace(',', ' ').split():
                try:
                    ids.add(int(token))
                except ValueError:
                    pass
    return ids


def main():
    parser = argparse.ArgumentParser(description='Build open_* CSVs from Starlink S1G2 TLE file')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--tle', help='Path to TLE file for S1G2 (3-line sets)')
    src.add_argument('--url', help='CelesTrak TLE URL (e.g., Starlink group)')
    parser.add_argument('--filter-ids', help='Path to text with NORAD IDs to include (space/comma/newline separated)')
    args = parser.parse_args()

    allowed_ids = None
    if args.filter_ids:
        allowed_ids = load_id_filter(args.filter_ids)

    if args.tle:
        with open(args.tle, 'r', encoding='utf-8') as f:
            lines = list(f)
        build_csvs_from_tle_source(lines, allowed_ids)
    else:
        # Try requests, fallback to urllib
        text = None
        try:
            import requests  # type: ignore
            resp = requests.get(args.url, timeout=30)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            try:
                from urllib.request import urlopen
                with urlopen(args.url, timeout=30) as r:  # type: ignore
                    text = r.read().decode('utf-8', errors='replace')
            except Exception as e:
                raise RuntimeError(f"Failed to fetch URL: {e}")

        lines = text.splitlines()
        build_csvs_from_tle_source(lines, allowed_ids)


if __name__ == '__main__':
    main()
