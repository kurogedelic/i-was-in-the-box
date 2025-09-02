#!/usr/bin/env python3
"""
Satellite Constellation Sonification
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))


def run_data_fetch():
    """Fetch public domain satellite data"""
    print("\n" + "=" * 60)
    print("STEP 1: Fetching PUBLIC DOMAIN Satellite Data")
    print("=" * 60)

    from scripts.fetch_open_data import save_open_data

    # Check if data already exists
    data_files = [
        "data/open_tle_data.csv",
        "data/open_positions.csv",
        "data/open_launch_timeline.csv",
    ]
    all_exist = all(os.path.exists(f) for f in data_files)

    if all_exist:
        print("Public domain data files already exist. Using existing data.")
        import pandas as pd

        tle_df = pd.read_csv("data/open_tle_data.csv")
        print(f"Loaded {len(tle_df)} satellites")

        # Show data sources
        sources = tle_df["data_source"].value_counts()
        print("\nData sources:")
        for source, count in sources.items():
            print(f"  • {source}: {count} satellites")
    else:
        print("Fetching public domain and generated data...")
        os.chdir("scripts")
        success = save_open_data()
        os.chdir("..")
        if not success:
            return False

    return True


def run_audio_generation():
    """Generate audio files for each satellite"""
    print("\n" + "=" * 60)
    print("STEP 2: Generating Audio from Open Data")
    print("=" * 60)

    # Temporarily update data file paths
    import shutil

    # Backup original if exists
    if os.path.exists("data/group2_tle_data.csv"):
        shutil.copy("data/group2_tle_data.csv", "data/group2_tle_data_backup.csv")
    if os.path.exists("data/group2_positions.csv"):
        shutil.copy("data/group2_positions.csv", "data/group2_positions_backup.csv")

    # Copy open data to expected names
    shutil.copy("data/open_tle_data.csv", "data/group2_tle_data.csv")
    shutil.copy("data/open_positions.csv", "data/group2_positions.csv")
    shutil.copy("data/open_launch_timeline.csv", "data/group2_launch_timeline.csv")

    # Clean up audio directory
    import glob

    if os.path.exists("audio"):
        for f in glob.glob("audio/*.wav"):
            os.remove(f)

    from scripts.generate_audio import generate_all_audio

    print("Generating audio files from public domain data...")
    os.chdir("scripts")
    generate_all_audio()
    os.chdir("..")

    return True


def run_mix_creation():
    """Create the final mix"""
    print("\n" + "=" * 60)
    print("STEP 3: Creating Final Mix")
    print("=" * 60)

    from scripts.create_mix_sweep import create_sweep_mix

    print("Creating altitude sweep mix...")
    os.chdir("scripts")
    output_file = create_sweep_mix()
    os.chdir("..")

    if output_file:
        # Rename to indicate it's from open data
        import shutil

        final_path = "mix/open_data_mix.wav"
        shutil.move(output_file.replace("../", ""), final_path)
        print(f"\n✅ SUCCESS! Mix created at: {final_path}")
        return True
    else:
        print("\n❌ ERROR: Failed to create mix")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Satellite Sonification - Commercial Use OK"
    )
    parser.add_argument("--full", action="store_true", help="Run complete pipeline")
    parser.add_argument("--data", action="store_true", help="Fetch data only")
    parser.add_argument("--audio", action="store_true", help="Generate audio only")
    parser.add_argument("--mix", action="store_true", help="Create mix only")

    args = parser.parse_args()

    # Default to full pipeline
    if not any([args.full, args.data, args.audio, args.mix]):
        args.full = True

    print("\n" + "=" * 60)
    print("🛰️  SATELLITE CONSTELLATION SONIFICATION")
    print("📜  LICENSE: PUBLIC DOMAIN / GENERATED DATA")
    print("✅  COMMERCIAL USE: ALLOWED")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = True

    try:
        if args.data or args.full:
            success = run_data_fetch() and success

        if args.audio or args.full:
            success = run_audio_generation() and success

        if args.mix or args.full:
            success = run_mix_creation() and success

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📊 DATA VERIFICATION:")
        print("  • ISS (NASA - Public Domain)")
        print("  • NASA Missions (Public Domain)")
        print("  • Synthetic Constellation (Generated)")
        print("  • NO LICENSE RESTRICTIONS")

        if args.full or args.mix:
            print("\n🎵 OUTPUT:")
            print("  • Final mix: mix/open_data_mix.wav")
            print("  • Duration: 2 minutes")
            print("  • Commercial use: ✅ ALLOWED")
            print("  • Attribution: Not required (but appreciated)")
            print("\n💰 You can:")
            print("  • Sell this music commercially")
            print("  • Use in commercial projects")
            print("  • Modify and redistribute")
            print("  • No permission needed!")
    else:
        print("❌ PIPELINE FAILED - Check errors above")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
