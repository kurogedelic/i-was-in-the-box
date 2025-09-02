#!/usr/bin/env python3
"""
Choral Induction Protocol
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def euclidean_rhythm(pulses, steps, rotation=0):
    """Generate Euclidean rhythm pattern with rotation"""
    if pulses > steps:
        pulses = steps
    if pulses == 0:
        return [0] * steps

    pattern = []
    for i in range(steps):
        if (i * pulses) % steps < pulses:
            pattern.append(1)
        else:
            pattern.append(0)

    # Rotate pattern
    if rotation > 0:
        pattern = pattern[rotation:] + pattern[:rotation]

    return pattern


def synthesize_vowel_formant(vowel, duration, pitch_hz, sr=44100, voice_character=1.0):
    """Enhanced vowel synthesis with strong formants"""

    t = np.arange(int(duration * sr)) / sr

    # Stronger glottal source with harmonics
    source = np.zeros_like(t)
    period_samples = int(sr / pitch_hz)

    # Add slight pitch variation for naturalness
    pitch_variation = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)

    for i in range(len(t)):
        phase = (i % period_samples) / period_samples
        # Asymmetric pulse for richer harmonics
        if phase < 0.35:
            source[i] = (phase / 0.35) ** 0.8
        elif phase < 0.45:
            source[i] = 1.0 - ((phase - 0.35) / 0.1) ** 0.5
        else:
            source[i] = 0

    # Add slight noise
    source += np.random.randn(len(source)) * 0.008

    # Non-Western formant settings for unique vocal character
    formants = {
        "a": [(650, 25, 1.3), (1100, 35, 0.9), (2500, 45, 0.6), (3500, 55, 0.4)],
        "e": [(500, 22, 1.3), (1750, 32, 1.0), (2450, 42, 0.6), (3400, 52, 0.4)],
        "i": [(280, 18, 1.2), (2250, 28, 1.1), (2900, 38, 0.7), (3700, 48, 0.5)],
        "o": [(550, 23, 1.3), (850, 33, 1.0), (2350, 43, 0.6), (3300, 53, 0.4)],
        "u": [(320, 18, 1.2), (800, 28, 0.9), (2200, 38, 0.6), (3100, 48, 0.4)],
        "ə": [
            (500, 30, 1.1),
            (1500, 40, 0.8),
            (2500, 50, 0.5),
            (3500, 60, 0.3),
        ],  # schwa
    }

    if vowel not in formants:
        vowel = "a"

    # Parallel formant synthesis
    output = np.zeros_like(source)

    for freq, bw, amp in formants[vowel]:
        if freq < sr / 2:
            # Adjust formant frequency by voice character
            freq = freq * voice_character

            # High-Q resonator
            Q = freq / bw
            w0 = 2 * np.pi * freq / sr
            r = np.exp(-np.pi * bw / sr)
            r = min(r * 1.08, 0.995)  # Strong resonance

            a1 = -2 * r * np.cos(w0)
            a2 = r * r

            # Compensate gain
            gain = (1 - r) * np.sqrt(1 + r * r - 2 * r * np.cos(w0))
            filtered = signal.lfilter([gain], [1, a1, a2], source)
            output += filtered * amp

    # Mix formants with source
    output = output * 0.9 + source * 0.1

    # Apply decay envelope
    envelope = np.ones_like(output)

    # Fast attack
    attack_samples = int(0.003 * sr)
    if attack_samples < len(envelope):
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2

    # Exponential decay
    decay_start = attack_samples
    if decay_start < len(envelope):
        decay_length = len(envelope) - decay_start
        decay_curve = np.exp(-2.5 * np.linspace(0, 1, decay_length))
        envelope[decay_start:] *= decay_curve

    output *= envelope

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.9

    return output


def synthesize_voice(syllable, duration, pitch_hz, sr=44100, voice_type="normal"):
    """Synthesize syllable with voice type variation"""

    # Voice type characteristics
    voice_types = {
        "normal": 1.0,
        "low": 0.8,
        "high": 1.2,
        "nasal": 1.1,
    }

    voice_char = voice_types.get(voice_type, 1.0)
    adjusted_pitch = pitch_hz * voice_char

    consonant = syllable[0] if len(syllable) > 1 else ""
    vowel = syllable[-1]

    samples = []

    # Quick consonant attacks
    if consonant in ["k", "c"]:
        # Short burst
        burst_dur = 0.008
        burst = np.random.randn(int(burst_dur * sr)) * 0.4
        b, a = signal.butter(2, [1200, 3500], btype="band", fs=sr)
        burst = signal.lfilter(b, a, burst)
        burst *= np.exp(-5 * np.linspace(0, 1, len(burst)))
        samples.append(burst)

    elif consonant == "t":
        burst_dur = 0.006
        burst = np.random.randn(int(burst_dur * sr)) * 0.35
        b, a = signal.butter(2, [2000, 5000], btype="band", fs=sr)
        burst = signal.lfilter(b, a, burst)
        burst *= np.exp(-5 * np.linspace(0, 1, len(burst)))
        samples.append(burst)

    # Vowel part
    consonant_duration = sum(len(s) / sr for s in samples)
    vowel_duration = max(duration - consonant_duration, 0.05)

    vowel_sound = synthesize_vowel_formant(
        vowel, vowel_duration, adjusted_pitch, sr, voice_char
    )
    samples.append(vowel_sound)

    if samples:
        output = np.concatenate(samples)
    else:
        output = synthesize_vowel_formant(
            vowel, duration, adjusted_pitch, sr, voice_char
        )

    return output


class VoicesSynthesizer:
    """Polyphonic voices synthesizer with interlocking patterns"""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.tempo = 180  # Fast interlocking tempo
        self.beat_duration = 60.0 / self.tempo / 4  # 16th notes

        # Pentatonic/modal pitches (not Western chords)
        # Using slendro-inspired intervals
        base_freq = 110  # A2
        self.pitch_set = [
            base_freq * 1.0,  # Root
            base_freq * 1.125,  # ~M2
            base_freq * 1.265,  # ~M3
            base_freq * 1.5,  # P5
            base_freq * 1.685,  # ~M6
            base_freq * 2.0,  # Octave
            base_freq * 2.25,  # M2 up
            base_freq * 2.53,  # M3 up
        ]

        # Complex interlocking patterns with wide stereo positioning
        self.voice_patterns = [
            # Sangsih (core pattern) - CENTER with slight movement
            {
                "name": "sangsih",
                "pattern": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "syllables": ["cak", "cak", "cak", "cak"],
                "pitch_indices": [0, 2, 0, 2],
                "voice_type": "low",
                "pan": 0.0,
                "pan_movement": 0.2,  # Slight stereo movement
                "volume": 1.0,
            },
            # Polos (interlocking with sangsih) - LEFT
            {
                "name": "polos",
                "pattern": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                "syllables": ["ke", "ke", "ke", "ke"],
                "pitch_indices": [3, 4, 3, 4],
                "voice_type": "normal",
                "pan": -0.8,  # Far left
                "pan_movement": 0.3,
                "volume": 0.9,
            },
            # Cecekan (fast pattern) - RIGHT
            {
                "name": "cecekan",
                "pattern": euclidean_rhythm(7, 16, rotation=2),
                "syllables": ["ce", "ce", "ce", "ce", "ce", "ce", "ce"],
                "pitch_indices": [5, 6, 5, 6, 5, 6, 5],
                "voice_type": "high",
                "pan": 0.8,  # Far right
                "pan_movement": 0.3,
                "volume": 0.8,
            },
            # Kotekan 1 - HARD LEFT
            {
                "name": "kotekan1",
                "pattern": euclidean_rhythm(5, 16, rotation=0),
                "syllables": ["ko", "te", "ko", "te", "kan"],
                "pitch_indices": [1, 3, 1, 3, 4],
                "voice_type": "normal",
                "pan": -0.95,  # Hard left
                "pan_movement": 0.1,
                "volume": 0.7,
            },
            # Kotekan 2 - HARD RIGHT
            {
                "name": "kotekan2",
                "pattern": euclidean_rhythm(5, 16, rotation=3),
                "syllables": ["te", "kan", "te", "kan", "te"],
                "pitch_indices": [2, 4, 2, 4, 5],
                "voice_type": "normal",
                "pan": 0.95,  # Hard right
                "pan_movement": 0.1,
                "volume": 0.7,
            },
            # Jegogan (bass) - CENTER WIDE (stereo bass)
            {
                "name": "jegogan",
                "pattern": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                "syllables": ["ja", "gong"],
                "pitch_indices": [0, 0],
                "voice_type": "low",
                "pan": 0.0,
                "pan_movement": 0.5,  # Wide stereo bass
                "volume": 1.1,
            },
            # Reyong pattern - MID LEFT
            {
                "name": "reyong",
                "pattern": euclidean_rhythm(9, 16),
                "syllables": [
                    "ri",
                    "yong",
                    "ri",
                    "yong",
                    "cak",
                    "ri",
                    "yong",
                    "cak",
                    "cak",
                ],
                "pitch_indices": [7, 6, 7, 6, 5, 7, 6, 5, 4],
                "voice_type": "high",
                "pan": -0.6,
                "pan_movement": 0.4,  # Moving around left
                "volume": 0.6,
            },
            # Cak pattern - MID RIGHT
            {
                "name": "cak_main",
                "pattern": euclidean_rhythm(11, 16),
                "syllables": [
                    "cak",
                    "a",
                    "cak",
                    "a",
                    "cak",
                    "cak",
                    "a",
                    "cak",
                    "a",
                    "cak",
                    "a",
                ],
                "pitch_indices": [3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 4],
                "voice_type": "nasal",
                "pan": 0.6,
                "pan_movement": 0.4,  # Moving around right
                "volume": 0.8,
            },
        ]

        # Dynamic sections for 2 minutes
        self.sections = [
            {"name": "Intro", "duration": 8, "voices": [0, 5], "dynamics": 0.6},
            {"name": "Build1", "duration": 16, "voices": [0, 1, 5], "dynamics": 0.7},
            {"name": "Build2", "duration": 16, "voices": [0, 1, 2, 5], "dynamics": 0.8},
            {
                "name": "Main1",
                "duration": 20,
                "voices": [0, 1, 2, 3, 4, 5, 7],
                "dynamics": 0.9,
            },
            {"name": "Peak", "duration": 24, "voices": list(range(8)), "dynamics": 1.0},
            {"name": "Break", "duration": 8, "voices": [0, 2, 5], "dynamics": 0.7},
            {
                "name": "Main2",
                "duration": 20,
                "voices": list(range(8)),
                "dynamics": 0.95,
            },
            {"name": "Outro", "duration": 8, "voices": [0, 1, 5], "dynamics": 0.6},
        ]

    def generate_sequence(self, duration=120):
        """Generate 2-minute polyphonic voice sequence"""
        print("Generating polyphonic voices with interlocking rhythms...")
        print(f"Duration: {duration} seconds")
        print(f"Tempo: {self.tempo} BPM")

        samples = int(duration * self.sr)
        output = np.zeros((samples, 2))  # Stereo

        current_time = 0

        # Process each section
        for section_idx, section in enumerate(self.sections):
            section_name = section["name"]
            section_duration = section["duration"]
            active_voices = section["voices"]
            dynamics = section["dynamics"]

            print(f"  Section {section_idx + 1}: {section_name} ({section_duration}s)")

            section_start = int(current_time * self.sr)
            section_samples = int(section_duration * self.sr)

            # Generate patterns for active voices
            for voice_idx in active_voices:
                voice = self.voice_patterns[voice_idx]
                pattern = voice["pattern"]
                syllables = voice["syllables"]
                pitch_indices = voice["pitch_indices"]
                voice_type = voice["voice_type"]
                pan = voice["pan"]
                pan_movement = voice.get("pan_movement", 0.1)
                volume = voice["volume"] * dynamics

                # Calculate pattern timing
                beat_samples = int(self.beat_duration * self.sr)
                pattern_duration = len(pattern) * beat_samples

                # Generate pattern audio
                pattern_audio = np.zeros(pattern_duration)
                syllable_idx = 0

                for beat_idx, hit in enumerate(pattern):
                    if hit:
                        # Get syllable and pitch
                        syl = syllables[syllable_idx % len(syllables)]
                        pitch_idx = pitch_indices[syllable_idx % len(pitch_indices)]
                        pitch = self.pitch_set[pitch_idx % len(self.pitch_set)]
                        syllable_idx += 1

                        # Duration with slight variation
                        syl_duration = self.beat_duration * np.random.uniform(0.7, 0.9)

                        # Synthesize
                        syl_audio = synthesize_voice(
                            syl, syl_duration, pitch, self.sr, voice_type
                        )

                        # Place in pattern
                        start = beat_idx * beat_samples
                        end = min(start + len(syl_audio), len(pattern_audio))
                        syl_len = end - start

                        if syl_len > 0:
                            pattern_audio[start:end] += syl_audio[:syl_len] * volume

                # Repeat pattern to fill section
                num_repeats = int(np.ceil(section_samples / pattern_duration))
                repeated = np.tile(pattern_audio, num_repeats)[:section_samples]

                # Apply dynamic panning with movement
                t_pan = np.arange(len(repeated)) / self.sr

                # Create moving pan position
                pan_freq = 0.3 + voice_idx * 0.1  # Different speed for each voice
                pan_dynamic = pan + pan_movement * np.sin(2 * np.pi * pan_freq * t_pan)
                pan_dynamic = np.clip(pan_dynamic, -1, 1)

                # Calculate stereo gains with Haas effect for width
                left_gain = np.sqrt((1 - pan_dynamic) / 2)
                right_gain = np.sqrt((1 + pan_dynamic) / 2)

                # Add micro delays for extra width (Haas effect)
                if pan < 0:  # Left-panned voices
                    delay_samples = int(0.0005 * self.sr)  # 0.5ms delay
                    if delay_samples < len(repeated):
                        delayed_right = np.zeros_like(repeated)
                        delayed_right[delay_samples:] = repeated[:-delay_samples]
                        repeated_right = delayed_right * 0.7 + repeated * 0.3
                        repeated_left = repeated
                    else:
                        repeated_right = repeated
                        repeated_left = repeated
                elif pan > 0:  # Right-panned voices
                    delay_samples = int(0.0005 * self.sr)
                    if delay_samples < len(repeated):
                        delayed_left = np.zeros_like(repeated)
                        delayed_left[delay_samples:] = repeated[:-delay_samples]
                        repeated_left = delayed_left * 0.7 + repeated * 0.3
                        repeated_right = repeated
                    else:
                        repeated_left = repeated
                        repeated_right = repeated
                else:  # Center voices
                    repeated_left = repeated
                    repeated_right = repeated

                # Add to output with processed stereo
                end_idx = min(section_start + len(repeated), samples)
                length = end_idx - section_start

                if length > 0:
                    output[section_start:end_idx, 0] += (
                        repeated_left[:length] * left_gain[:length]
                    )
                    output[section_start:end_idx, 1] += (
                        repeated_right[:length] * right_gain[:length]
                    )

            current_time += section_duration

        # Add room reverb with stereo width
        print("  Adding spatial acoustics with wide stereo field...")
        reverb = np.zeros_like(output)

        # Early reflections with wider stereo
        early_delays = [
            (0.012, 0.35, -0.7),
            (0.018, 0.30, 0.7),  # Wide left/right
            (0.025, 0.25, -0.9),
            (0.032, 0.20, 0.9),  # Very wide
            (0.038, 0.18, -0.5),
            (0.045, 0.15, 0.5),  # Mid width
        ]

        for delay_sec, gain, pan_offset in early_delays:
            delay_samples = int(delay_sec * self.sr)
            if delay_samples < len(output):
                delayed = np.zeros_like(output)
                delayed[delay_samples:] = output[:-delay_samples] * gain

                # Enhanced stereo spread
                if pan_offset != 0:
                    temp = delayed.copy()
                    # Stronger cross-feed for wider image
                    delayed[:, 0] = temp[:, 0] * (1 + pan_offset) + temp[:, 1] * (
                        -pan_offset * 0.3
                    )
                    delayed[:, 1] = temp[:, 1] * (1 - pan_offset) + temp[:, 0] * (
                        pan_offset * 0.3
                    )

                reverb += delayed

        # Late reverb with more stereo difference
        late_delays = [(0.067, 0.18), (0.089, 0.15), (0.113, 0.12), (0.147, 0.10)]

        for delay_sec, gain in late_delays:
            delay_samples_l = int(delay_sec * self.sr)
            delay_samples_r = int(delay_sec * 1.12 * self.sr)  # Bigger L/R difference

            if delay_samples_l < len(output):
                reverb[delay_samples_l:, 0] += output[:-delay_samples_l, 0] * gain
            if delay_samples_r < len(output):
                reverb[delay_samples_r:, 1] += output[:-delay_samples_r, 1] * gain

        # Mix
        output = output * 0.65 + reverb * 0.35

        # Soft compression
        output = np.tanh(output * 0.8) / 0.8

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9

        return output


def main():
    print("\n" + "=" * 60)
    print("VOICES - POLYPHONIC VOCAL SYNTHESIZER")
    print("Euclidean Rhythms with Harmonic Sequences")
    print("=" * 60)

    # Create synthesizer
    synth = VoicesSynthesizer(sample_rate=44100)

    # Generate 2-minute sequence
    duration = 120
    audio = synth.generate_sequence(duration)

    # Save
    output_filename = "voices_output.wav"
    wavfile.write(output_filename, synth.sr, np.int16(audio * 32767))
    print(f"\nSaved: {output_filename}")

    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Waveform
    t = np.arange(len(audio)) / synth.sr
    axes[0].plot(t, audio[:, 0], alpha=0.7, label="Left", linewidth=0.5)
    axes[0].plot(t, audio[:, 1], alpha=0.7, label="Right", linewidth=0.5)
    axes[0].set_title("Voices - Interlocking Polyphonic Rhythms")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark sections
    current_time = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(synth.sections)))
    for i, section in enumerate(synth.sections):
        start_time = current_time
        end_time = current_time + section["duration"]
        axes[0].axvspan(start_time, end_time, alpha=0.2, color=colors[i])
        axes[0].text(
            (start_time + end_time) / 2, 0.9, section["name"], ha="center", fontsize=8
        )
        current_time = end_time

    # Spectrogram
    mono = np.mean(audio, axis=1)
    f, t_spec, Sxx = signal.spectrogram(mono, synth.sr, nperseg=4096, noverlap=3072)
    freq_idx = np.where(f <= 2000)[0]

    im = axes[1].pcolormesh(
        t_spec,
        f[freq_idx],
        10 * np.log10(Sxx[freq_idx] + 1e-10),
        shading="gouraud",
        cmap="inferno",
        vmin=-60,
        vmax=-10,
    )
    axes[1].set_title("Spectrogram - Pentatonic Harmonics")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0, 2000)
    plt.colorbar(im, ax=axes[1], label="Power (dB)")

    # Pattern visualization
    axes[2].set_title("Interlocking Rhythm Patterns")
    axes[2].set_xlabel("Beat (16th notes)")
    axes[2].set_ylabel("Voice")

    for i, voice in enumerate(synth.voice_patterns[:5]):  # Show first 5 patterns
        pattern = voice["pattern"]
        y_pos = i
        for j, hit in enumerate(pattern):
            if hit:
                axes[2].scatter(j, y_pos, s=50, c="red", alpha=0.7)
            else:
                axes[2].scatter(j, y_pos, s=20, c="gray", alpha=0.3)
        axes[2].text(-1, y_pos, voice["name"], ha="right", fontsize=8)

    axes[2].set_xlim(-2, 16)
    axes[2].set_ylim(-0.5, 4.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("voices_analysis.png", dpi=150)
    print("Saved: voices_analysis.png")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Duration: {duration} seconds (2 minutes)")
    print(f"Tempo: {synth.tempo} BPM")
    print(f"Voices: {len(synth.voice_patterns)} interlocking patterns")
    print(f"Scale: Pentatonic/Modal (non-Western)")
    print("=" * 60)


if __name__ == "__main__":
    main()
