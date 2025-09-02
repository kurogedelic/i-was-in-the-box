# Satellite Constellation Sonification 

This project sonifies satellite constellations. By default it operates on public-domain and generated data, and it can also work with user-provided CSVs derived from CelesTrak Starlink TLEs.

No satellite data is embedded in this repository. The program reads CSVs from `data/` and produces audio to `audio/` and mixes to `mix/`.

## Data Separation

- Input data lives only as CSV files under `data/` and is not distributed by default.
- You may redistribute the program without CSVs and without fetch scripts; users can generate CSVs themselves and place them in `data/`.
- When using CelesTrak or other sources, follow their terms of use. This repository does not redistribute their data.

## Required CSVs

Place the following files under `data/`:

- `open_tle_data.csv`: columns `satellite_id,name,altitude_km,inclination,raan,eccentricity,mean_motion,data_source`
- `open_positions.csv`: columns `satellite_id,timestamp,latitude,longitude,altitude_km`
- `open_launch_timeline.csv`: columns `satellite_id,launch_date,data_source`

These schemas match files produced by the helper script below.

## Getting Starlink CSVs from CelesTrak (for your own use)

This repo includes a helper to convert Starlink TLEs to the required CSVs. You do not need to distribute this script with your build — it’s only for users to generate their own CSVs.

- Helper: `scripts/fetch_starlink_s1g2.py`
  - Sources: either a local TLE file (`--tle`) or a CelesTrak URL (`--url`)
  - Optional filter: `--filter-ids` to restrict to a set of NORAD IDs

Examples:

1) Group 1-2 only (Starlink v1.0 L2, international designator `2020-001`):

```
MISSION=2020-001
python - <<'PY' "$MISSION"
import json,urllib.request,sys
MISSION=sys.argv[1]
u='https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json'
j=json.load(urllib.request.urlopen(u,timeout=30))
ids=sorted({o['NORAD_CAT_ID'] for o in j if o.get('OBJECT_ID','').startswith(MISSION) and abs(o.get('INCLINATION',0)-53.0)<1.0})
open(f'data/{MISSION}_ids.txt','w').write('\n'.join(map(str,ids)))
print(f'MISSION={MISSION} count={len(ids)} -> data/{MISSION}_ids.txt')
PY

python scripts/fetch_starlink_s1g2.py \
  --url "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" \
  --filter-ids data/${MISSION}_ids.txt
```

2) Season 1 (Shell 1, ~53° inclination) — all current objects:

```
python - <<'PY'
import json,urllib.request
u='https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json'
j=json.load(urllib.request.urlopen(u, timeout=30))
ids=sorted({o['NORAD_CAT_ID'] for o in j if abs(o.get('INCLINATION',0)-53.0)<=0.7})
open('data/shell1_ids.txt','w').write('\n'.join(map(str,ids)))
print('Shell1 IDs:', len(ids), '-> data/shell1_ids.txt')
PY

python scripts/fetch_starlink_s1g2.py \
  --url "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" \
  --filter-ids data/shell1_ids.txt
```

Outputs in both cases:

- `data/open_tle_data.csv`
- `data/open_positions.csv` (24h pseudo-positions)
- `data/open_launch_timeline.csv`

Notes:

- The helper estimates altitude from mean motion and synthesizes simple 24-hour position samples for panning; it does not perform high-precision propagation.
- Current CelesTrak GP provides only currently tracked objects. Historical “all satellites at launch time” requires archives or Space-Track.

## Generating Audio / Mix

- Generate audio per satellite: `python main.py --audio`
- Create a 2-min altitude sweep mix: `python main.py --mix`

Tips:

- For very large sets (e.g., Shell 1 with thousands of sats), `--mix` is faster and lighter than generating all per-satellite WAVs.
- Avoid `--full` when distributing non-public-domain data; it includes a data-fetch step aimed at public/open data.

## Visualization (2D Orbits)

You can render a simple 2D animation where satellites circle around an Earth-like disc. It uses `open_tle_data.csv` and maps mean motion to angular speed; altitude maps to ring radius.

Examples:

- MP4 (requires ffmpeg):

```
python scripts/visualize_orbits.py --duration 20 --fps 30 --out viz/orbits.mp4 --sat-limit 600
```

- GIF fallback (no ffmpeg):

```
python scripts/visualize_orbits.py --duration 12 --fps 20 --out viz/orbits.gif --sat-limit 400
```

Options:

- `--size`: canvas size (default 900 px square)
- `--earth-radius`: center disc radius in px (default 160)
- `--min-alt`/`--max-alt`: override altitude scaling range (km)
- `--sat-limit`: cap number of satellites for performance (0 means all)
- `--csv`: path to `open_tle_data.csv` (default `data/open_tle_data.csv`)

The animation is approximate and intended for aesthetic visualization, not precise orbital mechanics.

### Web (p5.js) — simple 2D ring

Export a compact JSON and open the p5.js viewer in a local server:

```
python scripts/export_satellite_json.py --out viz/p5/satellites.json --limit 600
python -m http.server -d viz/p5 8000
# open http://localhost:8000/
```

Controls:

- Speed (days/sec when timeline is present)
- Trails (enable/disable background clear)
- Green Screen (flat green bg for chroma key in editors)
- Ascend Days (time from launch to operational altitude)

Notes:

- Uses altitude→radius, mean_motion→angular velocity, RAAN→initial phase.
- For large datasets, increase `--limit` when exporting or tune the UI speed.
- If `open_launch_timeline.csv` is present, exporter includes `launch_day` and the viewer plays a launch→operational ascent over `Ascend Days`. Satellites are hidden pre-launch.

## Cleanup

Use the cleanup helper to remove generated artifacts before packaging:

- Preview (dry-run): `python scripts/cleanup.py`
- Delete audio/mix/backup files: `python scripts/cleanup.py --apply`
- Delete also `open_*` CSVs and ID lists: `python scripts/cleanup.py --apply --all`
- Optionally prune empty dirs: add `--prune-dirs`

## Licensing

- The code is MIT-like usage oriented for generated/public-domain data. Data obtained from CelesTrak or other providers is subject to their terms. Do not redistribute third-party data unless allowed.
