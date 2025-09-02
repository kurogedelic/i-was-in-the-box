#!/usr/bin/env python3

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import subprocess
import json
from datetime import datetime
import argparse
import sys
import soundfile as sf


class AlbumMastering:
    def __init__(self, input_dir="../bounces", output_dir="../dist"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_rate = 48000

        # Target loudness standards
        self.target_lufs = -14  # Streaming platforms standard
        self.target_true_peak = -1  # dBTP

        # Album metadata
        self.album_metadata = {
            "album": "箱の中に居たのは私だった。",
            "artist": "Leo Kuroshita",
            "year": "2025",
            "genre": "Experimental Electronic",
            "copyright": "(C) 2025 kurogedelic, CC BY-NC-SA 4.0",
            "comment": "Plain Music Album - DAWless Production",
        }

        # Track order and metadata
        self.track_list = [
            {
                "filename": "EMRSP.wav",
                "title": "感覚と感知の刹那的な認識",
                "track_num": 1,
                "original": "EMRSP.wav",
            },
            {
                "filename": "Your_Wrapped_Conversation.wav",
                "title": "ラッピングされた君の会話",
                "track_num": 2,
                "original": "Your_Wrapped_Conversation.wav",
            },
            {
                "filename": "drosera_with_piano.wav",
                "title": "Drosera's Song",
                "track_num": 3,
                "original": "drosera_with_piano.wav",
            },
            {
                "filename": "stress-strain_curve.wav",
                "title": "応力–ひずみ曲線",
                "track_num": 4,
                "original": "stress-strain_curve.wav",
            },
            {
                "filename": "constellation.wav",
                "title": "Constellation - Phase 1 Group 2",
                "track_num": 5,
                "original": "constellation.wav",
            },
            {
                "filename": "field.wav",
                "title": "不確定の庭",
                "track_num": 6,
                "original": "field.wav",
            },
            {
                "filename": "acid_new.wav",
                "title": "アシッド・テクノの印象",
                "track_num": 7,
                "original": "acid_new.wav",
            },
            {
                "filename": "shininglake.wav",
                "title": "輝く湖（Avonlea）",
                "track_num": 8,
                "original": "shininglake.wav",
            },
            {
                "filename": "scsi_disk_20250818_214022.wav",
                "title": "以下、SCSIディスクが回答します。",
                "track_num": 9,
                "original": "scsi_disk_20250818_214022.wav",
            },
        ]

        os.makedirs(self.output_dir, exist_ok=True)

    def measure_lufs(self, audio_file):
        """Measure LUFS using ffmpeg ebur128 filter"""
        cmd = [
            "ffmpeg",
            "-i",
            audio_file,
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            output = result.stdout

            # Parse LUFS values from output
            lufs_i = None
            true_peak = None

            for line in output.split("\n"):
                if "I:" in line and "LUFS" in line:
                    try:
                        lufs_i = float(line.split("I:")[1].split("LUFS")[0].strip())
                    except:
                        pass
                if "Peak:" in line:
                    try:
                        true_peak = float(
                            line.split("Peak:")[1].split("dBFS")[0].strip()
                        )
                    except:
                        pass

            return lufs_i, true_peak
        except Exception as e:
            print(f"Error measuring LUFS: {e}")
            return None, None

    def apply_mastering(self, input_file, output_file, track_metadata):
        """Apply mastering chain to a track"""
        print(f"Mastering: {track_metadata['title']}")

        # Read audio (supporting WAV/AIFF/others)
        data, sr = sf.read(input_file, always_2d=True, dtype="float32")
        # Ensure shape (num_samples, 2)
        if data.shape[1] == 1:
            audio = np.concatenate([data, data], axis=1)
        else:
            audio = data

        # Resample to 48kHz if needed
        target_sr = 48000
        if sr != target_sr:
            # Polyphase resampling per channel
            if len(audio.shape) == 1:
                audio = signal.resample_poly(audio, target_sr, sr)
            else:
                audio = np.stack(
                    [
                        signal.resample_poly(audio[:, ch], target_sr, sr)
                        for ch in range(audio.shape[1])
                    ],
                    axis=1,
                )
            sr = target_sr

        # 1. High-pass filter (remove sub-bass rumble)
        sos_hp = signal.butter(2, 20 / (sr / 2), "high", output="sos")
        audio = signal.sosfilt(sos_hp, audio, axis=0)

        # 2. Gentle EQ curve
        nyquist = sr / 2

        # Slight bass boost around 100Hz
        bass_low = min(80 / nyquist, 0.99)
        bass_high = min(120 / nyquist, 0.99)
        if bass_low < bass_high:
            sos_bass = signal.butter(2, [bass_low, bass_high], "band", output="sos")
            bass_boost = signal.sosfilt(sos_bass, audio, axis=0) * 0.15
        else:
            bass_boost = 0

        # Presence boost around 3-5kHz
        pres_low = min(3000 / nyquist, 0.99)
        pres_high = min(5000 / nyquist, 0.99)
        if pres_low < pres_high:
            sos_presence = signal.butter(2, [pres_low, pres_high], "band", output="sos")
            presence_boost = signal.sosfilt(sos_presence, audio, axis=0) * 0.1
        else:
            presence_boost = 0

        # Air boost above 10kHz (only if sample rate allows)
        air_freq = min(10000 / nyquist, 0.99)
        if air_freq < 0.99:
            sos_air = signal.butter(2, air_freq, "high", output="sos")
            air_boost = signal.sosfilt(sos_air, audio, axis=0) * 0.05
        else:
            air_boost = 0

        # Mix EQ
        audio_eq = audio + bass_boost + presence_boost + air_boost

        # 3. Multiband compression (simplified 3-band)
        bands = []

        # Low band (< 200 Hz)
        low_freq = min(200 / (sr / 2), 0.99)
        sos_low = signal.butter(2, low_freq, "low", output="sos")
        low_band = signal.sosfilt(sos_low, audio_eq, axis=0)
        low_compressed = self.compress_band(low_band, threshold=-18, ratio=2.5)
        bands.append(low_compressed)

        # Mid band (200 Hz - 4 kHz)
        mid_low = min(200 / (sr / 2), 0.99)
        mid_high = min(4000 / (sr / 2), 0.99)
        if mid_low < mid_high:
            sos_mid = signal.butter(2, [mid_low, mid_high], "band", output="sos")
            mid_band = signal.sosfilt(sos_mid, audio_eq, axis=0)
            mid_compressed = self.compress_band(mid_band, threshold=-20, ratio=2)
            bands.append(mid_compressed)

        # High band (> 4 kHz)
        high_freq = min(4000 / (sr / 2), 0.99)
        if high_freq < 0.99:
            sos_high = signal.butter(2, high_freq, "high", output="sos")
            high_band = signal.sosfilt(sos_high, audio_eq, axis=0)
            high_compressed = self.compress_band(high_band, threshold=-22, ratio=1.5)
            bands.append(high_compressed)

        # Sum bands
        audio_compressed = sum(bands) / len(bands)

        # 4. Stereo enhancement (subtle)
        if audio_compressed.shape[1] == 2:
            mid = (audio_compressed[:, 0] + audio_compressed[:, 1]) / 2
            side = (audio_compressed[:, 0] - audio_compressed[:, 1]) / 2

            # Enhance side signal slightly
            side *= 1.2

            # Convert back to L/R
            audio_compressed[:, 0] = mid + side
            audio_compressed[:, 1] = mid - side

        # 5. Final limiting (aim for -1 dBTP headroom)
        audio_limited = self.limiter(audio_compressed, threshold=-1.0)

        # 6. Apply fade in/out
        fade_samples = int(0.01 * sr)  # 10ms fades
        if len(audio_limited) > fade_samples * 2:
            audio_limited[:fade_samples] *= np.linspace(0, 1, fade_samples).reshape(
                -1, 1
            )
            audio_limited[-fade_samples:] *= np.linspace(1, 0, fade_samples).reshape(
                -1, 1
            )

        # Normalize to prevent clipping (sample peak safeguard)
        peak = np.max(np.abs(audio_limited))
        if peak > 0.891:  # ≈ -1 dBFS linear
            audio_limited = audio_limited * 0.891 / peak

        # Write 24-bit WAV @ 48kHz
        sf.write(output_file, audio_limited.astype(np.float32), sr, subtype="PCM_24")

        # Apply metadata using ffmpeg
        self.apply_metadata(output_file, track_metadata)

        # Measure final LUFS
        lufs, peak = self.measure_lufs(output_file)
        if lufs and peak:
            print(f"  Final LUFS: {lufs:.1f}, True Peak: {peak:.1f} dBTP")

        return output_file

    def compress_band(self, audio, threshold=-20, ratio=3, attack=0.005, release=0.1):
        """Simple compressor for a frequency band"""
        # RMS envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms window

        if len(audio.shape) == 2:
            envelope = np.sqrt(np.mean(audio**2, axis=1))
        else:
            envelope = np.abs(audio)

        # Smooth envelope
        from scipy.ndimage import uniform_filter1d

        envelope_smooth = uniform_filter1d(envelope, size=window_size)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope_smooth + 1e-10)

        # Calculate gain reduction
        gain_db = np.zeros_like(envelope_db)
        over_threshold = envelope_db > threshold
        gain_db[over_threshold] = (threshold - envelope_db[over_threshold]) * (
            1 - 1 / ratio
        )

        # Convert back to linear
        gain = 10 ** (gain_db / 20)

        # Apply attack/release smoothing
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        gain_smooth = np.copy(gain)
        for i in range(1, len(gain_smooth)):
            if gain_smooth[i] < gain_smooth[i - 1]:
                # Attack
                alpha = 1 - np.exp(-1 / attack_samples)
                gain_smooth[i] = gain_smooth[i - 1] + alpha * (
                    gain_smooth[i] - gain_smooth[i - 1]
                )
            else:
                # Release
                alpha = 1 - np.exp(-1 / release_samples)
                gain_smooth[i] = gain_smooth[i - 1] + alpha * (
                    gain_smooth[i] - gain_smooth[i - 1]
                )

        # Apply gain
        if len(audio.shape) == 2:
            return audio * gain_smooth.reshape(-1, 1)
        else:
            return audio * gain_smooth

    def limiter(self, audio, threshold=-0.3):
        """Brick-wall limiter"""
        threshold_linear = 10 ** (threshold / 20)

        # Look-ahead buffer
        lookahead_samples = int(0.005 * self.sample_rate)  # 5ms

        if len(audio.shape) == 2:
            peak = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))
        else:
            peak = np.abs(audio)

        # Calculate required gain reduction
        gain = np.ones_like(peak)
        over_threshold = peak > threshold_linear
        gain[over_threshold] = threshold_linear / peak[over_threshold]

        # Smooth gain changes
        from scipy.ndimage import minimum_filter1d

        gain_smooth = minimum_filter1d(gain, size=lookahead_samples)

        # Apply gain
        if len(audio.shape) == 2:
            return audio * gain_smooth.reshape(-1, 1)
        else:
            return audio * gain_smooth

    def apply_metadata(self, audio_file, track_metadata):
        """Apply metadata to audio file using ffmpeg"""
        temp_file = audio_file + ".temp.wav"

        cmd = [
            "ffmpeg",
            "-i",
            audio_file,
            "-metadata",
            f"title={track_metadata['title']}",
            "-metadata",
            f"artist={self.album_metadata['artist']}",
            "-metadata",
            f"album={self.album_metadata['album']}",
            "-metadata",
            f"track={track_metadata['track_num']}",
            "-metadata",
            f"date={self.album_metadata['year']}",
            "-metadata",
            f"genre={self.album_metadata['genre']}",
            "-metadata",
            f"copyright={self.album_metadata['copyright']}",
            "-metadata",
            f"comment={self.album_metadata['comment']}",
            "-c:a",
            "copy",
            "-y",
            temp_file,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(temp_file, audio_file)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not apply metadata: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def find_latest_file(self, pattern):
        """Find the latest file matching a pattern"""
        import glob

        files = glob.glob(os.path.join(self.input_dir, pattern))
        if files:
            return max(files, key=os.path.getctime)
        return None

    def process_album(self):
        """Process all tracks in the album"""
        print("=" * 60)
        print("Album Mastering Process")
        print(f"Album: {self.album_metadata['album']}")
        print(f"Artist: {self.album_metadata['artist']}")
        print("=" * 60)

        processed_tracks = []

        for track in self.track_list:
            print(f"\nTrack {track['track_num']}: {track['title']}")
            print("-" * 40)

            # Find input file
            if "*" in track["original"]:
                input_file = self.find_latest_file(track["original"])
            else:
                input_file = os.path.join(self.input_dir, track["original"])

            if not input_file or not os.path.exists(input_file):
                print(f"  WARNING: Source file not found for {track['title']}")
                continue

            # Generate output filename
            output_filename = f"{track['track_num']:02d}_{track['title'].replace(' ', '_').replace('/', '_')}.wav"
            output_file = os.path.join(self.output_dir, output_filename)

            # Apply mastering
            mastered_file = self.apply_mastering(input_file, output_file, track)
            processed_tracks.append(mastered_file)

            print(f"  Saved: {output_filename}")

        # Generate track listing
        self.generate_tracklist(processed_tracks)

        print("\n" + "=" * 60)
        print(f"Mastering complete! {len(processed_tracks)} tracks processed.")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

        return processed_tracks

    def generate_tracklist(self, tracks):
        """Generate a tracklist file"""
        tracklist_file = os.path.join(self.output_dir, "TRACKLIST.txt")

        with open(tracklist_file, "w", encoding="utf-8") as f:
            f.write(f"{self.album_metadata['album']}\n")
            f.write(f"{self.album_metadata['artist']}\n")
            f.write(f"{self.album_metadata['year']}\n")
            f.write("=" * 60 + "\n\n")

            for track in self.track_list:
                if any(str(track["track_num"]) in os.path.basename(t) for t in tracks):
                    f.write(f"{track['track_num']:02d}. {track['title']}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"License: {self.album_metadata['copyright']}\n")
            f.write(f"Genre: {self.album_metadata['genre']}\n")
            f.write(f"Production: {self.album_metadata['comment']}\n")

        print(f"\nTracklist saved to: {tracklist_file}")

    def generate_cue_sheet(self):
        """Generate a CUE sheet for CD burning"""
        cue_file = os.path.join(self.output_dir, "album.cue")

        with open(cue_file, "w", encoding="utf-8") as f:
            f.write(f'TITLE "{self.album_metadata["album"]}"\n')
            f.write(f'PERFORMER "{self.album_metadata["artist"]}"\n')
            f.write(f'REM DATE {self.album_metadata["year"]}\n')
            f.write(f'REM GENRE "{self.album_metadata["genre"]}"\n\n')

            for track in self.track_list:
                filename = f"{track['track_num']:02d}_{track['title'].replace(' ', '_').replace('/', '_')}.wav"
                if os.path.exists(os.path.join(self.output_dir, filename)):
                    f.write(f'FILE "{filename}" WAVE\n')
                    f.write(f'  TRACK {track["track_num"]:02d} AUDIO\n')
                    f.write(f'    TITLE "{track["title"]}"\n')
                    f.write(f'    PERFORMER "{self.album_metadata["artist"]}"\n')
                    f.write(f"    INDEX 01 00:00:00\n\n")

        print(f"CUE sheet saved to: {cue_file}")


def main():
    parser = argparse.ArgumentParser(description="Master album tracks")
    parser.add_argument(
        "--input-dir", default="../bounces", help="Input directory with source files"
    )
    parser.add_argument(
        "--output-dir", default="../dist", help="Output directory for mastered files"
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14,
        help="Target LUFS for streaming (default: -14)",
    )
    parser.add_argument(
        "--generate-cue", action="store_true", help="Generate CUE sheet for CD burning"
    )

    args = parser.parse_args()

    # Create mastering instance
    mastering = AlbumMastering(args.input_dir, args.output_dir)
    mastering.target_lufs = args.target_lufs

    # Process album
    tracks = mastering.process_album()

    # Generate CUE sheet if requested
    if args.generate_cue:
        mastering.generate_cue_sheet()

    return 0 if tracks else 1


if __name__ == "__main__":
    sys.exit(main())
