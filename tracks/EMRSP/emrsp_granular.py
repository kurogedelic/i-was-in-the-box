#!/usr/bin/env python3
"""
EMRSP
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter, filtfilt, hilbert
from scipy.fft import rfft, irfft, rfftfreq
import os
import tempfile
from pydub import AudioSegment
from datetime import datetime
import argparse

SR = 48000


def load_mono(path, sr=SR):
    """Load audio file and convert to mono"""
    if path.lower().endswith(".m4a"):
        audio = AudioSegment.from_file(path, format="m4a")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio.export(tmp_path, format="wav")
        try:
            x, s = sf.read(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        x, s = sf.read(path)

    if x.ndim == 2:
        x = x.mean(axis=1)
    if s != sr:
        t_old = np.linspace(0, len(x) / s, len(x), endpoint=False)
        t_new = np.linspace(0, len(x) / s, int(len(x) * sr / s), endpoint=False)
        x = np.interp(t_new, t_old, x)
    x = x.astype(np.float32)
    x /= max(1e-9, np.max(np.abs(x)))
    return x


class FXProcessor:
    """Dynamic effects processor for granular synthesis"""

    def __init__(self, sr=SR):
        self.sr = sr
        self.delay_buffer_l = np.zeros(sr * 2, dtype=np.float32)
        self.delay_buffer_r = np.zeros(sr * 2, dtype=np.float32)
        self.delay_pos = 0
        self.reverb_buffer_l = []
        self.reverb_buffer_r = []
        self.init_reverb()

    def init_reverb(self):
        delays = [0.013, 0.017, 0.023, 0.029, 0.037, 0.041]
        for d in delays:
            size = int(d * self.sr)
            self.reverb_buffer_l.append(np.zeros(size, dtype=np.float32))
            self.reverb_buffer_r.append(np.zeros(size, dtype=np.float32))

    def process_delay(self, audio_l, audio_r, delay_time, feedback, mix):
        delay_samples = int(delay_time * self.sr)
        output_l = np.zeros_like(audio_l)
        output_r = np.zeros_like(audio_r)

        for i in range(len(audio_l)):
            read_pos = (self.delay_pos - delay_samples) % len(self.delay_buffer_l)
            delayed_l = self.delay_buffer_l[read_pos]
            delayed_r = self.delay_buffer_r[read_pos]

            self.delay_buffer_l[self.delay_pos] = audio_l[i] + delayed_l * feedback
            self.delay_buffer_r[self.delay_pos] = audio_r[i] + delayed_r * feedback

            output_l[i] = audio_l[i] * (1 - mix) + delayed_l * mix
            output_r[i] = audio_r[i] * (1 - mix) + delayed_r * mix

            self.delay_pos = (self.delay_pos + 1) % len(self.delay_buffer_l)

        return output_l, output_r

    def process_reverb(self, audio_l, audio_r, room_size, damping, mix):
        output_l = audio_l.copy()
        output_r = audio_r.copy()

        for i, (buf_l, buf_r) in enumerate(
            zip(self.reverb_buffer_l, self.reverb_buffer_r)
        ):
            gain = 0.5 + room_size * 0.3

            for j in range(len(audio_l)):
                idx = j % len(buf_l)
                delayed_l = buf_l[idx]
                delayed_r = buf_r[idx]

                buf_l[idx] = output_l[j] + delayed_l * gain
                buf_r[idx] = output_r[j] + delayed_r * gain

                output_l[j] = delayed_l - output_l[j] * gain
                output_r[j] = delayed_r - output_r[j] * gain

        b, a = butter(1, 1000 * (1 - damping) / (self.sr / 2), btype="low")
        output_l = lfilter(b, a, output_l)
        output_r = lfilter(b, a, output_r)

        result_l = audio_l * (1 - mix) + output_l * mix
        result_r = audio_r * (1 - mix) + output_r * mix

        return result_l, result_r

    def process_filter(self, audio, freq, resonance, filter_type="bandpass"):
        nyquist = self.sr / 2
        freq = np.clip(freq, 20, nyquist * 0.95)

        if filter_type == "bandpass":
            low = freq * (1 - resonance * 0.5)
            high = freq * (1 + resonance * 0.5)
            low = np.clip(low, 20, nyquist * 0.95)
            high = np.clip(high, low + 10, nyquist * 0.95)
            b, a = butter(2, [low / nyquist, high / nyquist], btype="band")
        elif filter_type == "lowpass":
            b, a = butter(2, freq / nyquist, btype="low")
        elif filter_type == "highpass":
            b, a = butter(2, freq / nyquist, btype="high")

        return lfilter(b, a, audio)

    def spectral_freeze(self, audio, freeze_factor):
        spectrum = rfft(audio)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        phase_noise = np.random.uniform(-np.pi, np.pi, len(phase)) * freeze_factor
        new_phase = phase + phase_noise

        new_spectrum = magnitude * np.exp(1j * new_phase)
        return irfft(new_spectrum, n=len(audio)).real.astype(np.float32)


def make_emrsp_granular(sources, minutes=2.0, seed=None):
    """Create EMRSP granular synthesis with multiple sources"""
    rng = np.random.default_rng(seed)
    dur = int(minutes * 60 * SR)
    out = np.zeros((2, dur), np.float32)
    fx = FXProcessor(SR)
    n_sources = len(sources)

    n_segments = 20
    segment_size = dur // n_segments

    for seg in range(n_segments):
        start_pos = seg * segment_size
        end_pos = min((seg + 1) * segment_size, dur)

        progress = seg / n_segments

        # Dynamic grain density
        if progress < 0.3:
            grains_per_sec = 5 + progress * 50
        elif progress < 0.7:
            grains_per_sec = 20 + np.sin((progress - 0.3) * 10) * 15
        else:
            grains_per_sec = 10 + (1 - progress) * 20

        n_grains = int((end_pos - start_pos) / SR * grains_per_sec)

        # Dynamic FX parameters
        delay_time = 0.05 + progress * 0.3 + rng.uniform(-0.02, 0.02)
        delay_feedback = 0.3 + np.sin(progress * np.pi * 4) * 0.2
        delay_mix = 0.2 + progress * 0.3

        reverb_room = 0.3 + progress * 0.5
        reverb_damp = 0.5 - progress * 0.3
        reverb_mix = 0.1 + progress * 0.4

        filter_freq = 200 + np.exp(progress * 3) * 1000 + rng.uniform(-200, 200)
        filter_res = 0.3 + np.sin(progress * np.pi * 6) * 0.3

        freeze_factor = 0
        if 0.4 < progress < 0.6:
            freeze_factor = (progress - 0.4) * 5

        segment_audio = np.zeros((2, end_pos - start_pos), np.float32)

        for i in range(n_grains):
            g_min = 0.005 + progress * 0.02
            g_max = 0.05 + (1 - progress) * 0.2
            g_len = int(rng.uniform(g_min, g_max) * SR)

            # Select random sources for this grain
            src_idx1 = rng.integers(0, n_sources)
            src_idx2 = rng.integers(0, n_sources)

            src1 = sources[src_idx1]
            src2 = sources[src_idx2]

            chaos = np.sin(i * 0.1) * np.cos(i * 0.07) * np.sin(progress * np.pi * 2)
            pos_1 = int(np.abs(chaos) * (len(src1) - g_len))
            pos_2 = int(np.abs(chaos * 1.3) * (len(src2) - g_len))

            pos_1 = min(pos_1, len(src1) - g_len - 1)
            pos_2 = min(pos_2, len(src2) - g_len - 1)

            g1 = src1[pos_1 : pos_1 + g_len]
            g2 = src2[pos_2 : pos_2 + g_len]

            morph = progress**1.5
            grain = g1 * (1 - morph) + g2 * morph

            if rng.random() < 0.3:
                attack = int(g_len * rng.uniform(0.1, 0.3))
                w = np.ones(g_len, np.float32)
                w[:attack] = np.linspace(0, 1, attack)
                w[-attack:] = np.linspace(1, 0, attack)
            else:
                x = np.linspace(-3, 3, g_len)
                w = np.exp(-(x**2) / 2)

            grain *= w

            if rng.random() < 0.5:
                grain = fx.process_filter(
                    grain,
                    filter_freq,
                    filter_res,
                    rng.choice(["bandpass", "lowpass", "highpass"]),
                )

            if freeze_factor > 0 and rng.random() < 0.3:
                grain = fx.spectral_freeze(grain, freeze_factor)

            pitch_shift = 1.0 + np.sin(i * 0.2 + progress * np.pi * 4) * 0.1
            if pitch_shift != 1.0:
                new_len = int(g_len / pitch_shift)
                x_old = np.linspace(0, g_len - 1, g_len)
                x_new = np.linspace(0, g_len - 1, new_len)
                grain = np.interp(x_new, x_old, grain)
                g_len = new_len

            t_offset = int(rng.uniform(0, segment_size - g_len))

            pan = 0.5 + 0.4 * np.sin(2 * np.pi * i * 0.15 + progress * np.pi * 3)
            l_gain = np.cos(pan * np.pi / 2)
            r_gain = np.sin(pan * np.pi / 2)

            amp = 0.5 + 0.5 * np.sin(i * 0.3 + progress * np.pi * 2)

            if t_offset + g_len <= segment_size:
                segment_audio[0, t_offset : t_offset + g_len] += grain * l_gain * amp
                segment_audio[1, t_offset : t_offset + g_len] += grain * r_gain * amp

        segment_audio[0], segment_audio[1] = fx.process_delay(
            segment_audio[0], segment_audio[1], delay_time, delay_feedback, delay_mix
        )

        segment_audio[0], segment_audio[1] = fx.process_reverb(
            segment_audio[0], segment_audio[1], reverb_room, reverb_damp, reverb_mix
        )

        out[:, start_pos:end_pos] = segment_audio

    out = np.tanh(out * 0.7)
    out[0] -= np.mean(out[0])
    out[1] -= np.mean(out[1])

    return out


def main():
    parser = argparse.ArgumentParser(
        description="EMRSP - Experiments in Momentary Recognition of Sensation and Perception"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "sources/source_1.wav",
            "sources/source_2.wav",
            "sources/source_3.wav",
            "sources/source_4.wav",
            "sources/source_5.wav",
        ],
        help="Source audio files (default: all source files)",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, help="Duration in minutes"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-play", action="store_true", help="Skip playback after rendering"
    )
    parser.add_argument("--output", help="Custom output filename")

    args = parser.parse_args()

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/emrsp_{timestamp}.wav"

    print("=" * 60)
    print("EMRSP - Experiments in Momentary Recognition")
    print("        of Sensation and Perception")
    print("=" * 60)
    print()

    print("Loading audio sources...")
    sources = []
    for i, source_path in enumerate(args.sources, 1):
        src = load_mono(source_path)
        sources.append(src)
        print(f"Source {i}: {source_path} ({len(src)/SR:.1f} seconds)")

    print()
    print("Creating granular synthesis...")
    print(f"Using {len(sources)} source files")
    print(f"Duration: {args.duration} minutes")
    print(f"Random seed: {args.seed if args.seed else 'None (random)'}")
    print()

    mix = make_emrsp_granular(sources, minutes=args.duration, seed=args.seed)

    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mix.T, SR, subtype="PCM_24")
    print(f"Saved: {output_path}")

    if not args.no_play:
        print("\nPlaying...")
        sd.play(mix.T, SR, blocking=True)
        print("Done!")

    return output_path


if __name__ == "__main__":
    main()
