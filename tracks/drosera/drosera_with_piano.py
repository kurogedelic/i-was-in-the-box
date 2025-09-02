#!/usr/bin/env python3
"""
Drosera's Song - Piano
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def midi_to_freq(midi_note):
    """Convert MIDI note to frequency"""
    A440 = 440.0
    return pow(2, (midi_note - 69) / 12.0) * A440


def generate_piano_tone(frequency, duration, sr=16000, harmonic_decay=0.7):
    """Generate piano tone (simplified)"""
    t = np.arange(int(duration * sr)) / sr

    # Generate fundamental and harmonics
    tone = np.zeros_like(t)
    for harmonic in range(1, 6):  # 5 harmonics
        amplitude = harmonic_decay ** (harmonic - 1)
        tone += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)

    # Piano-like envelope (sharp attack, gradual decay)
    attack_time = 0.01
    decay_time = 0.1
    sustain_level = 0.3

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
    envelope *= np.exp(-2.0 * t)

    tone *= envelope

    # Low-pass filter to remove high frequencies
    if frequency < sr / 4:
        b, a = signal.butter(2, frequency * 8 / (sr / 2), "low")
        tone = signal.lfilter(b, a, tone)

    return tone * 0.3  # Volume adjustment


def generate_chord(root_midi, chord_type, duration, sr=16000):
    """Generate chord"""
    chord = np.zeros(int(duration * sr))

    if chord_type == "major":
        intervals = [0, 4, 7]  # Root, major 3rd, perfect 5th
    elif chord_type == "minor":
        intervals = [0, 3, 7]  # Root, minor 3rd, perfect 5th
    elif chord_type == "dim":
        intervals = [0, 3, 6]  # Diminished triad
    elif chord_type == "sus4":
        intervals = [0, 5, 7]  # Sus4
    elif chord_type == "augmented":
        intervals = [0, 4, 8]  # Augmented triad (scary)
    else:
        intervals = [0]  # Single note

    for interval in intervals:
        freq = midi_to_freq(root_midi + interval)
        chord += generate_piano_tone(freq, duration, sr)

    return chord / len(intervals)  # Normalize


class DroseraAccompaniment:
    """Piano accompaniment for Drosera nursery rhyme"""

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.quarter_note = 60.0 / 135.0  # BPM 135 (quarter note = 0.444s)

    def create_intro(self):
        """Create intro (nursery rhyme style)"""
        print("  Creating intro...")
        intro = []

        # Arpeggio-style intro (C major) - 8 beats
        intro_notes = [
            60,
            64,
            67,
            72,  # C E G C (ascending arpeggio)
            72,
            67,
            64,
            60,  # C G E C (descending arpeggio)
        ]

        for note in intro_notes:
            tone = generate_piano_tone(
                midi_to_freq(note), self.quarter_note * 0.5, self.sr
            )
            intro.append(tone)
            # Short interval
            intro.append(np.zeros(int(0.01 * self.sr)))

        # End with chord (C major) - 4 beats
        intro.append(generate_chord(60, "major", self.quarter_note * 4, self.sr))

        # Total intro = 8 beats (arpeggio) + 4 beats (chord) = 12 beats = 3 bars
        print(f"  Intro duration: {(8 * 0.5 + 4) * self.quarter_note:.3f} seconds")

        return np.concatenate(intro)

    def create_accompaniment(self):
        """Create accompaniment part"""
        print("Creating piano accompaniment...")

        accompaniment = []

        # === Intro ===
        accompaniment.append(self.create_intro())

        # === Verse 1: "Little Drosera in the bog" ===
        # C major - Am - F - G (1 octave higher: +12)
        print("  Verse 1: C - Am - F - G")
        accompaniment.append(
            generate_chord(60, "major", self.quarter_note * 4, self.sr)
        )  # C (48→60)
        accompaniment.append(
            generate_chord(57, "minor", self.quarter_note * 4, self.sr)
        )  # Am (45→57)
        accompaniment.append(
            generate_chord(65, "major", self.quarter_note * 4, self.sr)
        )  # F (53→65)
        accompaniment.append(
            generate_chord(55, "major", self.quarter_note * 4, self.sr)
        )  # G (43→55)

        # === Verse 2: "Catching tiny bugs to eat" ===
        # C - Em - F - G
        print("  Verse 2: C - Em - F - G")
        accompaniment.append(
            generate_chord(60, "major", self.quarter_note * 4, self.sr)
        )  # C
        accompaniment.append(
            generate_chord(64, "minor", self.quarter_note * 4, self.sr)
        )  # Em (52→64)
        accompaniment.append(
            generate_chord(65, "major", self.quarter_note * 4, self.sr)
        )  # F
        accompaniment.append(
            generate_chord(55, "major", self.quarter_note * 4, self.sr)
        )  # G

        # === Verse 3: "Sticky droplets shine so bright" ===
        # G - F - Em - C
        print("  Verse 3: G - F - Em - C")
        accompaniment.append(
            generate_chord(55, "major", self.quarter_note * 4, self.sr)
        )  # G
        accompaniment.append(
            generate_chord(65, "major", self.quarter_note * 4, self.sr)
        )  # F
        accompaniment.append(
            generate_chord(64, "minor", self.quarter_note * 4, self.sr)
        )  # Em
        accompaniment.append(
            generate_chord(60, "major", self.quarter_note * 4, self.sr)
        )  # C

        # === Verse 4: "That's how plants can grow and thrive" ===
        # C - Dm - Bb - C
        print("  Verse 4: C - Dm - Bb - C")
        accompaniment.append(
            generate_chord(60, "major", self.quarter_note * 4, self.sr)
        )  # C
        accompaniment.append(
            generate_chord(62, "minor", self.quarter_note * 4, self.sr)
        )  # Dm (50→62)
        accompaniment.append(
            generate_chord(58, "major", self.quarter_note * 4, self.sr)
        )  # Bb (46→58)
        accompaniment.append(
            generate_chord(60, "major", self.quarter_note * 4, self.sr)
        )  # C

        # === Scary part: "I am hungry... Come closer..." ===
        print("  Scary part: Dissonant chords")

        # Dissonant and unstable chord progression (1 octave higher)
        # Bb dim - Ab augmented
        accompaniment.append(
            generate_chord(58, "dim", self.quarter_note * 8, self.sr)
        )  # Bb dim (46→58)
        accompaniment.append(
            generate_chord(56, "augmented", self.quarter_note * 8, self.sr)
        )  # Ab aug (44→56)

        # Descending dissonant notes (single note tremolo) - 1 octave higher
        scary_notes = [55, 53, 51, 49]  # G, F, Eb, Db (each +12)
        for note in scary_notes:
            # Tremolo effect (trembling sound)
            tone = generate_piano_tone(
                midi_to_freq(note), self.quarter_note * 4, self.sr
            )
            tremolo = 1 + 0.5 * np.sin(2 * np.pi * 8 * np.arange(len(tone)) / self.sr)
            tone *= tremolo
            accompaniment.append(tone)

        # Final long dissonant chord (cluster) - 1 octave higher
        final_cluster = np.zeros(int(self.quarter_note * 8 * self.sr))
        for note in [
            49,
            50,
            51,
            52,
            53,
        ]:  # Db, D, Eb, E, F (semitone cluster, each +12)
            final_cluster += (
                generate_piano_tone(midi_to_freq(note), self.quarter_note * 8, self.sr)
                * 0.2
            )
        accompaniment.append(final_cluster)

        # Concatenate all parts
        full_accompaniment = np.concatenate(accompaniment)

        # Add reverb
        reverb_delay = int(0.1 * self.sr)
        reverb = np.zeros_like(full_accompaniment)
        if reverb_delay < len(full_accompaniment):
            reverb[reverb_delay:] = full_accompaniment[:-reverb_delay] * 0.3
        full_accompaniment += reverb

        # Fade out
        fade_out = int(0.5 * self.sr)
        if fade_out < len(full_accompaniment):
            full_accompaniment[-fade_out:] *= np.linspace(1, 0, fade_out)

        # Normalize
        full_accompaniment = (
            full_accompaniment / (np.max(np.abs(full_accompaniment)) + 1e-10) * 0.5
        )

        return full_accompaniment

    def create_bass_line(self):
        """Create bass line (left hand part)"""
        print("Creating bass line...")

        bass = []

        # Bass for intro (matching arpeggio)
        intro_bass = (
            generate_piano_tone(midi_to_freq(36), self.quarter_note * 4, self.sr) * 0.3
        )
        intro_bass += (
            generate_piano_tone(midi_to_freq(48), self.quarter_note * 4, self.sr) * 0.2
        )
        bass.append(intro_bass)

        # Root notes for each bar (1 octave lower)
        bass_pattern = [
            # Verse 1
            36,
            36,
            33,
            33,
            41,
            41,
            31,
            31,  # C, C, A, A, F, F, G, G
            # Verse 2
            36,
            36,
            40,
            40,
            41,
            41,
            31,
            31,  # C, C, E, E, F, F, G, G
            # Verse 3
            31,
            31,
            41,
            41,
            40,
            40,
            36,
            36,  # G, G, F, F, E, E, C, C
            # Verse 4
            36,
            36,
            38,
            38,
            34,
            34,
            36,
            36,  # C, C, D, D, Bb, Bb, C, C
        ]

        # Bass for normal parts
        for note in bass_pattern:
            # Quarter note walking bass style
            bass_tone = generate_piano_tone(
                midi_to_freq(note), self.quarter_note * 2, self.sr
            )
            bass.append(bass_tone)

        # Bass for scary part (low drone)
        print("  Creating scary bass drone...")

        # Low Bb drone
        drone_freq = midi_to_freq(34)  # Low Bb
        drone_duration = self.quarter_note * 16
        t = np.arange(int(drone_duration * self.sr)) / self.sr

        # Layer multiple low notes for eerie sound
        drone = np.sin(2 * np.pi * drone_freq * t) * 0.3
        drone += np.sin(2 * np.pi * drone_freq * 0.5 * t) * 0.2  # 1 octave down
        drone += np.sin(2 * np.pi * drone_freq * 1.5 * t) * 0.1  # 5th up

        # Add fluctuation
        lfo = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        drone *= lfo

        bass.append(drone)

        # Descending bass notes
        for note in [31, 29, 27, 25]:  # G, F, Eb, Db
            bass_tone = (
                generate_piano_tone(midi_to_freq(note), self.quarter_note * 4, self.sr)
                * 0.5
            )
            bass.append(bass_tone)

        # Final low cluster
        final_bass = np.zeros(int(self.quarter_note * 8 * self.sr))
        for note in [25, 26, 27]:  # Db, D, Eb
            final_bass += (
                generate_piano_tone(midi_to_freq(note), self.quarter_note * 8, self.sr)
                * 0.2
            )
        bass.append(final_bass)

        # Concatenate
        full_bass = np.concatenate(bass)

        # Normalize
        full_bass = full_bass / (np.max(np.abs(full_bass)) + 1e-10) * 0.3

        return full_bass


def mix_with_vocals(vocal_file, accompaniment, bass, sr=16000, intro_duration=None):
    """Mix vocals with accompaniment (with intro)"""
    # Load vocals
    vocal_sr, vocals = wavfile.read(vocal_file)

    # Convert if sample rate differs
    if vocal_sr != sr:
        vocals = signal.resample(vocals, int(len(vocals) * sr / vocal_sr))

    # Convert to float32 and normalize
    if vocals.dtype == np.int16:
        vocals = vocals.astype(np.float32) / 32767.0

    # Delay vocals by intro length
    # BPM 135: 8 beat arpeggio + 4 beat chord = 8 beats (simplified)
    if intro_duration is None:
        quarter_note = 60.0 / 135.0
        intro_duration = 8 * quarter_note  # 8 beats = ~3.556 seconds
    intro_samples = int(intro_duration * sr)
    vocals = np.pad(vocals, (intro_samples, 0))

    # Match lengths
    max_length = max(len(vocals), len(accompaniment), len(bass))

    if len(vocals) < max_length:
        vocals = np.pad(vocals, (0, max_length - len(vocals)))
    if len(accompaniment) < max_length:
        accompaniment = np.pad(accompaniment, (0, max_length - len(accompaniment)))
    if len(bass) < max_length:
        bass = np.pad(bass, (0, max_length - len(bass)))

    # Mix (vocals subdued)
    mixed = vocals * 0.3 + accompaniment * 0.5 + bass * 0.4

    # Prevent clipping
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-10) * 0.9

    return mixed


def main():
    print("\n" + "=" * 60)
    print("DROSERA NURSERY RHYME WITH PIANO ACCOMPANIMENT")
    print("Drosera Nursery Rhyme - With Piano Accompaniment")
    print("=" * 60)

    # Generate accompaniment
    accompanist = DroseraAccompaniment(sample_rate=16000)

    print("\nGenerating accompaniment parts...")
    accompaniment = accompanist.create_accompaniment()
    bass = accompanist.create_bass_line()

    # Save accompaniment only (match lengths)
    max_len = max(len(accompaniment), len(bass))
    if len(accompaniment) < max_len:
        accompaniment = np.pad(accompaniment, (0, max_len - len(accompaniment)))
    if len(bass) < max_len:
        bass = np.pad(bass, (0, max_len - len(bass)))

    piano_only = accompaniment + bass
    piano_only = piano_only / (np.max(np.abs(piano_only)) + 1e-10) * 0.8
    wavfile.write("drosera_piano_only.wav", 16000, np.int16(piano_only * 32767))
    print("\nSaved: drosera_piano_only.wav (piano accompaniment only)")

    # Mix with vocals
    print("\nMixing with vocals...")
    mixed = mix_with_vocals("drosera_nursery_rhyme.wav", accompaniment, bass, 16000)

    # Save
    wavfile.write("drosera_with_piano.wav", 16000, np.int16(mixed * 32767))
    print("Saved: drosera_with_piano.wav (full song with accompaniment)")

    # Analysis plot
    plt.figure(figsize=(16, 10))

    # Mixed waveform
    plt.subplot(3, 1, 1)
    t = np.arange(len(mixed)) / 16000
    plt.plot(t, mixed, color="purple", alpha=0.7)
    plt.title("Drosera Nursery Rhyme with Piano - Full Mix")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Piano only waveform
    plt.subplot(3, 1, 2)
    t_piano = np.arange(len(piano_only)) / 16000
    plt.plot(t_piano, piano_only, color="blue", alpha=0.7)
    plt.title("Piano Accompaniment Only")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Spectrogram
    plt.subplot(3, 1, 3)
    f, t, Sxx = signal.spectrogram(mixed, 16000, nperseg=1024)
    freq_idx = np.where(f <= 2000)[0]  # Focus on piano range
    plt.pcolormesh(
        t,
        f[freq_idx],
        10 * np.log10(Sxx[freq_idx, :] + 1e-10),
        shading="gouraud",
        cmap="magma",
    )
    plt.title("Spectrogram - Full Mix with Piano")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Power (dB)")

    plt.tight_layout()
    plt.savefig("drosera_with_piano_analysis.png")
    print("Saved: drosera_with_piano_analysis.png")

    # Song information
    print("\n" + "=" * 60)
    print("Composition Details:")
    print("  BPM:     135")
    print("  Intro:   C major arpeggio (3.556 seconds / 8 beats)")
    print("  Vocal Track: Educational nursery rhyme about Drosera")
    print("  Piano: Simple chord progression with scary ending (1 octave higher)")
    print("  Bass: Walking bass pattern + drone in scary part")
    print("\nChord Progression:")
    print("  Intro:   C arpeggio")
    print("  Verse 1: C - Am - F - G")
    print("  Verse 2: C - Em - F - G")
    print("  Verse 3: G - F - Em - C")
    print("  Verse 4: C - Dm - Bb - C")
    print("  Scary:   Bb dim - Ab aug - Cluster")
    print("=" * 60)


if __name__ == "__main__":
    main()
