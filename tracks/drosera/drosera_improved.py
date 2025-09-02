#!/usr/bin/env python3
"""
Drosera's Song - Improved
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


# Simple and memorable melody (eliminated mountain patterns, more linear)
DROSERA_MELODY = [
    # "Little Drosera in the bog" - stepwise motion
    60,
    62,
    64,
    64,
    65,
    65,
    64,  # C D E E F F E
    # "Catching tiny bugs to eat" - sequential progression
    62,
    64,
    65,
    67,
    67,
    65,
    64,  # D E F G G F E
    # "Sticky droplets shine so bright" - descending line
    67,
    65,
    64,
    62,
    60,
    60,
    60,  # G F E D C C C
    # "That's how plants can grow and thrive" - resolution melody
    64,
    65,
    67,
    65,
    64,
    62,
    60,  # E F G F E D C
]

# Timing (even beats for easy memorization)
DROSERA_TIMING = [
    # Line 1
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    # Line 2
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    # Line 3
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    # Line 4
    2,
    2,
    2,
    2,
    2,
    2,
    4,
]

# Lyrics (broken down by syllable)
DROSERA_LYRICS = [
    # "Little Drosera in the bog"
    ("l", "i", "Li"),
    ("t", "o", "ttle"),
    ("d", "o", "Dro"),
    ("s", "e", "se"),
    ("r", "a", "ra"),
    ("vowel", "i", "in"),
    ("vowel", "a", "bog"),
    # "Catching tiny bugs to eat"
    ("k", "a", "Cat"),
    ("t", "i", "ching"),
    ("t", "a", "ti"),
    ("n", "i", "ny"),
    ("b", "a", "bugs"),
    ("t", "u", "to"),
    ("vowel", "i", "eat"),
    # "Sticky droplets shine so bright"
    ("s", "i", "Sti"),
    ("k", "i", "cky"),
    ("d", "o", "drop"),
    ("l", "e", "lets"),
    ("s", "a", "shine"),
    ("s", "o", "so"),
    ("b", "a", "bright"),
    # "That's how plants can grow and thrive"
    ("vowel", "a", "That's"),
    ("h", "a", "how"),
    ("p", "a", "plants"),
    ("k", "a", "can"),
    ("g", "o", "grow"),
    ("vowel", "a", "and"),
    ("vowel", "a", "thrive"),
]


def midi_to_freq(midi_note):
    """Convert MIDI note number to frequency"""
    A440 = 440.0
    return pow(2, (midi_note - 69) / 12.0) * A440


def generate_piano_tone(frequency, duration, sr=44100, harmonic_decay=0.8):
    """Generate piano tone (improved version)"""
    t = np.arange(int(duration * sr)) / sr

    # Generate fundamental and harmonics
    tone = np.zeros_like(t)
    for harmonic in range(1, 5):  # 4 harmonics
        amplitude = harmonic_decay ** (harmonic - 1)
        tone += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)

    # Piano-like envelope
    attack_time = 0.02
    decay_time = 0.15
    sustain_level = 0.4

    envelope = np.ones_like(t) * sustain_level

    # Attack
    attack_samples = int(attack_time * sr)
    if attack_samples < len(t):
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay
    decay_samples = int(decay_time * sr)
    if attack_samples + decay_samples < len(t):
        envelope[attack_samples : attack_samples + decay_samples] = np.linspace(
            1, sustain_level, decay_samples
        )

    # Overall decay
    envelope *= np.exp(-1.5 * t)

    tone *= envelope

    return tone * 0.3


def generate_chord(root_midi, chord_type, duration, sr=44100):
    """Generate chord (improved version)"""
    chord = np.zeros(int(duration * sr))

    if chord_type == "major":
        intervals = [0, 4, 7]  # Root, major 3rd, perfect 5th
    elif chord_type == "minor":
        intervals = [0, 3, 7]  # Root, minor 3rd, perfect 5th
    elif chord_type == "dom7":
        intervals = [0, 4, 7, 10]  # Dominant seventh
    else:
        intervals = [0]  # Single note

    for interval in intervals:
        freq = midi_to_freq(root_midi + interval)
        chord += generate_piano_tone(freq, duration, sr)

    return chord / len(intervals)


def generate_glottal_rosenberg(duration, f0, sr):
    """Rosenberg glottal model"""
    t = np.arange(int(duration * sr)) / sr
    phase = (t * f0) % 1.0

    glottal = np.zeros_like(phase)

    Tp = 0.4
    Tn = 0.16

    for i, p in enumerate(phase):
        if p < Tp:
            glottal[i] = 0.5 * (1 - np.cos(np.pi * p / Tp))
        elif p < Tp + Tn:
            t_close = (p - Tp) / Tn
            glottal[i] = 0.5 * (1 + np.cos(np.pi * t_close))
        else:
            glottal[i] = 0

    return glottal


def cascade_formant_synth(vowel, duration=1.0, f0=60, sr=44100):
    """Cascade formant synthesis"""

    formant_data = {
        "a": [(700, 130), (1220, 70), (2600, 160)],
        "i": [(300, 60), (2300, 100), (3000, 200)],
        "u": [(300, 60), (900, 70), (2500, 170)],
        "e": [(500, 80), (1800, 90), (2500, 150)],
        "o": [(500, 80), (1000, 80), (2500, 170)],
    }

    if vowel not in formant_data:
        vowel = "a"

    glottal = generate_glottal_rosenberg(duration, f0, sr)

    output = glottal
    for freq, bw in formant_data[vowel]:
        if freq < sr / 2:
            Q = freq / bw
            b, a = signal.iirpeak(freq, Q, sr)
            output = signal.lfilter(b, a, output)

    b, a = signal.butter(1, 200 / (sr / 2), "high")
    output = signal.lfilter(b, a, output)

    output = output / (np.max(np.abs(output)) + 1e-10) * 0.7

    return output


class ImprovedDroseraSong:
    """Improved Drosera nursery rhyme (musically correct with panning)"""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.quarter_note = 60.0 / 120.0  # BPM 120 (more natural tempo)

    def synthesize_vocal_note(self, consonant, vowel, midi_note, duration_units):
        """Synthesize vocal note"""
        duration = duration_units * self.quarter_note
        pitch = midi_to_freq(midi_note)

        segments = []

        # Process consonants
        if consonant == "vowel":
            pass
        elif consonant == "l":
            liquid = cascade_formant_synth("a", 0.02, pitch, self.sr) * 0.6
            segments.append(liquid)
        elif consonant == "t":
            segments.append(np.zeros(int(0.025 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.2
            b, a = signal.butter(2, [2000, 3000], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "d":
            segments.append(np.zeros(int(0.02 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.15
            segments.append(burst)
        elif consonant == "s":
            friction = np.random.randn(int(0.04 * self.sr)) * 0.25
            b, a = signal.butter(2, [3000, 5000], btype="band", fs=self.sr)
            friction = signal.lfilter(b, a, friction)
            segments.append(friction)
        elif consonant == "r":
            tap = np.random.randn(int(0.008 * self.sr)) * 0.12
            segments.append(tap)
        elif consonant == "k":
            segments.append(np.zeros(int(0.03 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.18
            b, a = signal.butter(2, [1000, 2000], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "n":
            nasal = cascade_formant_synth("a", 0.04, pitch, self.sr)
            b, a = signal.iirpeak(250, 5, self.sr)
            nasal = signal.lfilter(b, a, nasal)
            segments.append(nasal * 0.6)
        elif consonant == "b":
            segments.append(np.zeros(int(0.02 * self.sr)))
            voiced_burst = (
                cascade_formant_synth(vowel, 0.015, pitch * 0.8, self.sr) * 0.3
            )
            segments.append(voiced_burst)
        elif consonant == "h":
            aspiration = np.random.randn(int(0.02 * self.sr)) * 0.08
            segments.append(aspiration)
        elif consonant == "p":
            segments.append(np.zeros(int(0.04 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.18
            b, a = signal.butter(2, [500, 1500], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "g":
            segments.append(np.zeros(int(0.03 * self.sr)))
            burst = np.random.randn(int(0.01 * self.sr)) * 0.15
            b, a = signal.butter(2, [800, 1500], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)

        # Calculate vowel duration
        consonant_duration = sum(len(s) / self.sr for s in segments)
        vowel_duration = duration - consonant_duration

        if vowel_duration > 0:
            # Synthesize vowel
            vowel_sound = cascade_formant_synth(vowel, vowel_duration, pitch, self.sr)

            # Envelope
            attack_len = int(0.01 * self.sr)
            if attack_len < len(vowel_sound):
                vowel_sound[:attack_len] *= np.linspace(0, 1, attack_len)

            # Decay
            decay_factor = np.exp(
                -1.0 / duration * np.arange(len(vowel_sound)) / self.sr
            )
            vowel_sound *= decay_factor

            segments.append(vowel_sound)

        if segments:
            return np.concatenate(segments)
        else:
            return np.zeros(int(duration * self.sr))

    def create_piano_intro(self):
        """Create piano intro (short and simple)"""
        print("Creating piano intro...")

        intro = []

        # Short intro melody (8 bars only)
        intro_melody = [
            60,
            62,
            64,
            65,  # C D E F (ascending)
            67,
            65,
            64,
            60,  # G F E C (descending and resolving)
        ]
        intro_timing = [
            2,
            2,
            2,
            2,  # 2 beats each
            2,
            2,
            2,
            2,
        ]

        # Combine melody and harmony
        for i, (note, beats) in enumerate(zip(intro_melody, intro_timing)):
            duration = beats * self.quarter_note

            # Melody tone
            melody_tone = (
                generate_piano_tone(midi_to_freq(note), duration, self.sr) * 0.6
            )

            # Accompaniment chords (left hand)
            if i < 4:  # First 4 notes
                # C major chord
                chord_root = 48  # Low C
                harmony = (
                    generate_piano_tone(midi_to_freq(chord_root), duration, self.sr)
                    * 0.3
                )
            else:  # Next 4 notes
                # G7 chord (dominant)
                chord_root = 43  # Low G
                harmony = (
                    generate_piano_tone(midi_to_freq(chord_root), duration, self.sr)
                    * 0.3
                )

            # Layer melody and harmony
            combined = melody_tone + harmony
            intro.append(combined)

        # End with C major chord (prepare for vocals)
        final_chord = generate_chord(60, "major", self.quarter_note * 2, self.sr)
        intro.append(final_chord)

        return np.concatenate(intro)

    def create_piano_accompaniment(self):
        """Create piano accompaniment (simple chord progression)"""
        print("Creating piano accompaniment...")

        accompaniment = []

        # Chord progression for each bar (musically correct)
        chord_progression = [
            # Line 1: I - IV - V - I (C - F - G - C)
            (60, "major", 4),  # C
            (65, "major", 4),  # F
            (67, "major", 4),  # G
            (60, "major", 4),  # C
            # Line 2: I - vi - IV - V (C - Am - F - G)
            (60, "major", 4),  # C
            (57, "minor", 4),  # Am
            (65, "major", 4),  # F
            (67, "major", 4),  # G
            # Line 3: vi - IV - ii - V (Am - F - Dm - G)
            (57, "minor", 4),  # Am
            (65, "major", 4),  # F
            (62, "minor", 4),  # Dm
            (67, "major", 4),  # G
            # Line 4: I - ii - V - I (C - Dm - G - C)
            (60, "major", 4),  # C
            (62, "minor", 4),  # Dm
            (67, "dom7", 4),  # G7
            (60, "major", 4),  # C
        ]

        for root, chord_type, beats in chord_progression:
            chord = generate_chord(root, chord_type, beats * self.quarter_note, self.sr)
            accompaniment.append(chord)

        full_accompaniment = np.concatenate(accompaniment)

        # Light reverb
        reverb_delay = int(0.05 * self.sr)
        reverb = np.zeros_like(full_accompaniment)
        if reverb_delay < len(full_accompaniment):
            reverb[reverb_delay:] = full_accompaniment[:-reverb_delay] * 0.2
        full_accompaniment += reverb

        return full_accompaniment

    def create_bass_line(self):
        """Create bass line"""
        print("Creating bass line...")

        bass = []

        # Root notes for each bar (low register)
        bass_notes = [
            # Line 1
            48,
            48,
            53,
            53,
            55,
            55,
            48,
            48,  # C C F F G G C C
            # Line 2
            48,
            48,
            45,
            45,
            53,
            53,
            55,
            55,  # C C A A F F G G
            # Line 3
            45,
            45,
            53,
            53,
            50,
            50,
            55,
            55,  # A A F F D D G G
            # Line 4
            48,
            48,
            50,
            50,
            55,
            55,
            48,
            48,  # C C D D G G C C
        ]

        for note in bass_notes:
            bass_tone = (
                generate_piano_tone(midi_to_freq(note), self.quarter_note * 2, self.sr)
                * 0.4
            )
            bass.append(bass_tone)

        return np.concatenate(bass)

    def sing_vocals(self):
        """Generate vocal part"""
        print("Synthesizing vocals...")

        vocals = []

        for i in range(len(DROSERA_MELODY)):
            if i < len(DROSERA_LYRICS):
                consonant, vowel, word = DROSERA_LYRICS[i]
                print(f"  Note {i+1}: {word}")

                note = self.synthesize_vocal_note(
                    consonant, vowel, DROSERA_MELODY[i], DROSERA_TIMING[i]
                )
                vocals.append(note)

                # Short silence between notes
                if i < len(DROSERA_MELODY) - 1:
                    vocals.append(np.zeros(int(0.005 * self.sr)))

        vocal_audio = np.concatenate(vocals)

        # Light reverb
        reverb_delay = int(0.03 * self.sr)
        reverb = np.zeros_like(vocal_audio)
        if reverb_delay < len(vocal_audio):
            reverb[reverb_delay:] = vocal_audio[:-reverb_delay] * 0.15
        vocal_audio += reverb

        return vocal_audio

    def create_ending(self):
        """Create ending (resolve with tonic chord)"""
        print("Creating ending...")

        ending = []

        # Play final C major chord (tonic) for extended duration
        # Gradually decrease volume
        final_chord = generate_chord(60, "major", self.quarter_note * 8, self.sr)

        # Fade out
        fade_samples = len(final_chord)
        fade_envelope = np.linspace(1, 0, fade_samples)
        final_chord *= fade_envelope

        # Add low C bass note
        bass_note = (
            generate_piano_tone(midi_to_freq(36), self.quarter_note * 8, self.sr) * 0.4
        )
        bass_note *= fade_envelope

        # Add high C (3 octaves of C)
        high_c = (
            generate_piano_tone(midi_to_freq(72), self.quarter_note * 8, self.sr) * 0.2
        )
        high_c *= fade_envelope

        # Combine all
        ending_chord = final_chord + bass_note + high_c
        ending.append(ending_chord)

        return np.concatenate(ending)

    def mix_with_panning(self, intro, vocals, piano, bass, ending):
        """Mix with panning (including intro and ending)"""
        print("Mixing with panning...")

        # Get intro length
        intro_length = len(intro)

        # Adjust vocals to start after intro
        vocals = np.pad(vocals, (intro_length, 0))

        # Combine piano and bass with intro
        full_piano = np.concatenate([intro, piano])
        full_bass = np.pad(bass, (intro_length, 0))  # Bass starts after intro

        # Add ending
        full_piano = np.concatenate([full_piano, ending])

        # Match lengths
        max_length = max(len(vocals), len(full_piano), len(full_bass))

        if len(vocals) < max_length:
            vocals = np.pad(vocals, (0, max_length - len(vocals)))
        if len(full_piano) < max_length:
            full_piano = np.pad(full_piano, (0, max_length - len(full_piano)))
        if len(full_bass) < max_length:
            full_bass = np.pad(full_bass, (0, max_length - len(full_bass)))

        # Create stereo track
        stereo = np.zeros((max_length, 2))

        # Piano: left channel biased (70% left, 30% right)
        stereo[:, 0] += (full_piano + full_bass) * 0.7  # Left channel
        stereo[:, 1] += (full_piano + full_bass) * 0.3  # Right channel

        # Vocals: right channel biased (30% left, 70% right)
        stereo[:, 0] += vocals * 0.3  # Left channel
        stereo[:, 1] += vocals * 0.7  # Right channel

        # Normalize
        max_val = np.max(np.abs(stereo))
        if max_val > 0:
            stereo = stereo / max_val * 0.8

        return stereo


def main():
    print("\n" + "=" * 60)
    print("IMPROVED DROSERA NURSERY RHYME")
    print("Improved Drosera Nursery Rhyme")
    print("=" * 60)

    song = ImprovedDroseraSong(sample_rate=44100)

    # Generate each part
    print("\nGenerating song parts...")
    intro = song.create_piano_intro()
    vocals = song.sing_vocals()
    piano = song.create_piano_accompaniment()
    bass = song.create_bass_line()
    ending = song.create_ending()

    # Mix with panning (including intro and ending)
    print("\nMixing with panning (piano left, vocals right)...")
    mixed_stereo = song.mix_with_panning(intro, vocals, piano, bass, ending)

    # Save
    wavfile.write("drosera_improved.wav", song.sr, np.int16(mixed_stereo * 32767))
    print("\nSaved: drosera_improved.wav (stereo with panning)")

    # Also create mono version
    mixed_mono = np.mean(mixed_stereo, axis=1)
    wavfile.write("drosera_improved_mono.wav", song.sr, np.int16(mixed_mono * 32767))
    print("Saved: drosera_improved_mono.wav (mono version)")

    # Analysis plots
    plt.figure(figsize=(16, 12))

    # Left channel (piano-focused)
    plt.subplot(4, 1, 1)
    t = np.arange(len(mixed_stereo)) / song.sr
    plt.plot(t, mixed_stereo[:, 0], color="blue", alpha=0.7)
    plt.title("Left Channel (Piano-focused)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Right channel (vocal-focused)
    plt.subplot(4, 1, 2)
    plt.plot(t, mixed_stereo[:, 1], color="red", alpha=0.7)
    plt.title("Right Channel (Vocal-focused)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Stereo image
    plt.subplot(4, 1, 3)
    plt.plot(t, mixed_stereo[:, 0] - mixed_stereo[:, 1], color="purple", alpha=0.5)
    plt.title("Stereo Difference (L-R)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Spectrogram (mono)
    plt.subplot(4, 1, 4)
    f, t, Sxx = signal.spectrogram(mixed_mono, song.sr, nperseg=1024)
    freq_idx = np.where(f <= 4000)[0]
    plt.pcolormesh(
        t,
        f[freq_idx],
        10 * np.log10(Sxx[freq_idx, :] + 1e-10),
        shading="gouraud",
        cmap="viridis",
    )
    plt.title("Spectrogram - Full Mix")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Power (dB)")

    plt.tight_layout()
    plt.savefig("drosera_improved_analysis.png")
    print("Saved: drosera_improved_analysis.png")

    # Song information
    print("\n" + "=" * 60)
    print("Composition Details:")
    print("  BPM:     120 (natural tempo)")
    print("  Key:     C Major")
    print("  Panning: Piano (Left), Vocals (Right)")
    print("  Intro:   8-bar simple piano introduction")
    print("  Ending:  8-bar tonic chord (C major) with fade out")
    print("\nChord Progression:")
    print("  Line 1: C - F - G - C   (I - IV - V - I)")
    print("  Line 2: C - Am - F - G  (I - vi - IV - V)")
    print("  Line 3: Am - F - Dm - G (vi - IV - ii - V)")
    print("  Line 4: C - Dm - G7 - C (I - ii - V7 - I)")
    print("\nMelody: Simple, memorable patterns in C major scale")
    print("=" * 60)


if __name__ == "__main__":
    main()
