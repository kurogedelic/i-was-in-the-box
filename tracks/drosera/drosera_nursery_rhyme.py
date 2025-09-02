#!/usr/bin/env python3
"""
Drosera's Song - Rhyme
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


# Simple nursery rhyme lyrics and melody
# "Little Drosera" - Educational plant song
DROSERA_MELODY = [
    # "Little Drosera in the bog"
    60,
    62,
    64,
    64,
    62,
    60,
    58,  # C D E E D C Bb
    # "Catching tiny bugs to eat"
    60,
    62,
    64,
    65,
    67,
    65,
    64,  # C D E F G F E
    # "Sticky droplets shine so bright"
    67,
    67,
    65,
    65,
    64,
    62,
    60,  # G G F F E D C
    # "That's how plants can grow and thrive"
    60,
    62,
    64,
    62,
    60,
    58,
    60,  # C D E D C Bb C
    # === Scary part (like original) ===
    # "I am hungry... Come closer..."
    58,
    56,
    58,
    56,  # Bb Ab Bb Ab (unstable intervals)
    55,
    53,
    51,
    49,  # G F Eb Db (descending dissonance)
]

# Timing (beats)
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
    # Scary part (slow and eerie)
    4,
    4,
    4,
    4,  # I am hungry
    4,
    4,
    6,
    8,  # Come closer... (last note held long)
]

# Lyrics (syllable breakdown)
DROSERA_LYRICS = [
    # "Little Drosera in the bog"
    ("l", "i", "Li"),  # Li-
    ("t", "o", "ttle"),  # -ttle
    ("d", "o", "Dro"),  # Dro-
    ("s", "e", "se"),  # -se-
    ("r", "a", "ra"),  # -ra
    ("vowel", "i", "in"),  # in
    ("vowel", "a", "bog"),  # bog (simplified)
    # "Catching tiny bugs to eat"
    ("k", "a", "Cat"),  # Cat-
    ("t", "i", "ching"),  # -ching (simplified)
    ("t", "a", "ti"),  # ti-
    ("n", "i", "ny"),  # -ny
    ("b", "a", "bugs"),  # bugs
    ("t", "u", "to"),  # to
    ("vowel", "i", "eat"),  # eat
    # "Sticky droplets shine so bright"
    ("s", "i", "Sti"),  # Sti-
    ("k", "i", "cky"),  # -cky
    ("d", "o", "drop"),  # drop-
    ("l", "e", "lets"),  # -lets
    ("s", "a", "shine"),  # shine (simplified)
    ("s", "o", "so"),  # so
    ("b", "a", "bright"),  # bright (simplified)
    # "That's how plants can grow and thrive"
    ("vowel", "a", "That's"),  # That's (simplified)
    ("h", "a", "how"),  # how
    ("p", "a", "plants"),  # plants (simplified)
    ("k", "a", "can"),  # can
    ("g", "o", "grow"),  # grow
    ("vowel", "a", "and"),  # and
    ("vowel", "a", "thrive"),  # thrive (simplified)
    # === Scary part ===
    # "I am hungry"
    ("vowel", "a", "I"),  # I (whispered)
    ("vowel", "a", "am"),  # am (whispered)
    ("h", "a", "hun"),  # hun- (breathy)
    ("g", "i", "gry"),  # -gry (low)
    # "Come closer"
    ("k", "a", "Come"),  # Come (dark)
    ("k", "o", "clo"),  # clo- (darker)
    ("s", "e", "ser"),  # -ser (hissing)
    ("vowel", "e", "..."),  # ... (fading whisper)
]


def midi_to_freq(midi_note):
    """Convert MIDI note to frequency (1 octave lower)"""
    A440 = 440.0
    original_freq = pow(2, (midi_note - 69) / 12.0) * A440
    return original_freq / 2.0  # Lower by 1 octave


def generate_glottal_rosenberg(duration, f0, sr):
    """Rosenberg glottal model (proven Daisy Bell version)"""
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


def cascade_formant_synth(vowel, duration=1.0, f0=60, sr=16000):
    """Cascade formant synthesis (proven version)"""

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


class DroseraSinger:
    """Drosera nursery rhyme synthesizer (using Daisy Bell techniques)"""

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.quarter_note = 60.0 / 135.0  # BPM 135 (quarter note = 0.444s)

    def synthesize_note(
        self, consonant, vowel, midi_note, duration_units, is_scary=False
    ):
        """Synthesize one note (same technique as Daisy Bell)"""
        duration = duration_units * self.quarter_note
        pitch = midi_to_freq(midi_note)

        # In scary part, vary pitch
        if is_scary:
            pitch = pitch * 0.8  # Even lower voice

        segments = []

        # Consonant processing (successful Daisy Bell settings)
        if consonant == "vowel":
            # Vowel only
            pass
        elif consonant == "l":
            # Liquid (English l)
            liquid = cascade_formant_synth("a", 0.02, pitch, self.sr) * 0.6
            segments.append(liquid)
        elif consonant == "t":
            # T sound
            segments.append(np.zeros(int(0.025 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.2
            b, a = signal.butter(2, [2000, 3000], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "d":
            # D sound
            segments.append(np.zeros(int(0.02 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.15
            segments.append(burst)
        elif consonant == "s":
            # S sound
            friction = np.random.randn(int(0.04 * self.sr)) * 0.25
            b, a = signal.butter(2, [3000, 5000], btype="band", fs=self.sr)
            friction = signal.lfilter(b, a, friction)
            segments.append(friction)
        elif consonant == "r":
            # R sound
            tap = np.random.randn(int(0.008 * self.sr)) * 0.12
            segments.append(tap)
        elif consonant == "k":
            # K sound
            segments.append(np.zeros(int(0.03 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.18
            b, a = signal.butter(2, [1000, 2000], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "n":
            # N sound
            nasal = cascade_formant_synth("a", 0.04, pitch, self.sr)
            b, a = signal.iirpeak(250, 5, self.sr)
            nasal = signal.lfilter(b, a, nasal)
            segments.append(nasal * 0.6)
        elif consonant == "b":
            # B sound
            segments.append(np.zeros(int(0.02 * self.sr)))
            voiced_burst = (
                cascade_formant_synth(vowel, 0.015, pitch * 0.8, self.sr) * 0.3
            )
            segments.append(voiced_burst)
        elif consonant == "h":
            # H sound
            aspiration = np.random.randn(int(0.02 * self.sr)) * 0.08
            segments.append(aspiration)
        elif consonant == "p":
            # P sound
            segments.append(np.zeros(int(0.04 * self.sr)))
            burst = np.random.randn(int(0.008 * self.sr)) * 0.18
            b, a = signal.butter(2, [500, 1500], btype="band", fs=self.sr)
            burst = signal.lfilter(b, a, burst)
            segments.append(burst)
        elif consonant == "g":
            # G sound
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

            # Make whisper-like in scary part
            if is_scary:
                # Add noise for whisper effect
                whisper_noise = np.random.randn(len(vowel_sound)) * 0.05
                vowel_sound = vowel_sound * 0.5 + whisper_noise
                # Vary amplitude (unstable voice)
                tremolo = 1 + 0.3 * np.sin(
                    2 * np.pi * 4 * np.arange(len(vowel_sound)) / self.sr
                )
                vowel_sound *= tremolo

            # Envelope (ADSR-style)
            attack_len = int(0.01 * self.sr)
            if attack_len < len(vowel_sound):
                vowel_sound[:attack_len] *= np.linspace(0, 1, attack_len)

            # Decay (gentle for nursery rhyme, sharp for scary part)
            if is_scary:
                decay_factor = np.exp(
                    -3.0 / duration * np.arange(len(vowel_sound)) / self.sr
                )
            else:
                decay_factor = np.exp(
                    -1.0 / duration * np.arange(len(vowel_sound)) / self.sr
                )
            vowel_sound *= decay_factor

            segments.append(vowel_sound)

        if segments:
            return np.concatenate(segments)
        else:
            return np.zeros(int(duration * self.sr))

    def sing_drosera_rhyme(self):
        """Sing the Drosera nursery rhyme"""
        print("Synthesizing Drosera Nursery Rhyme...")
        print("Educational song about carnivorous plants for children\n")

        song = []

        # Process each note
        for i in range(len(DROSERA_MELODY)):
            if i < len(DROSERA_LYRICS):
                consonant, vowel, word = DROSERA_LYRICS[i]
                print(
                    f"  Note {i+1}: {word} (MIDI {DROSERA_MELODY[i]}, {DROSERA_TIMING[i]} beats)"
                )

                # Determine scary part (last 8 notes)
                is_scary_part = i >= len(DROSERA_LYRICS) - 8

                note = self.synthesize_note(
                    consonant,
                    vowel,
                    DROSERA_MELODY[i],
                    DROSERA_TIMING[i],
                    is_scary=is_scary_part,
                )
                song.append(note)

                # Short silence between notes (longer in scary part)
                if i < len(DROSERA_MELODY) - 1:
                    if is_scary_part:
                        song.append(
                            np.zeros(int(0.02 * self.sr))
                        )  # Longer gaps in scary part
                    else:
                        song.append(np.zeros(int(0.005 * self.sr)))

        # Concatenate all parts
        song_audio = np.concatenate(song)

        # Add reverb effect (emphasized in scary part)
        reverb_delay1 = int(0.05 * self.sr)
        reverb_delay2 = int(0.1 * self.sr)
        reverb = np.zeros_like(song_audio)
        if reverb_delay1 < len(song_audio):
            reverb[reverb_delay1:] += song_audio[:-reverb_delay1] * 0.3
        if reverb_delay2 < len(song_audio):
            reverb[reverb_delay2:] += song_audio[:-reverb_delay2] * 0.2

        # Apply strong reverb only to scary part
        scary_start = int(len(song_audio) * 0.7)  # Last 30%
        song_audio[scary_start:] = song_audio[scary_start:] + reverb[scary_start:] * 0.5

        # Overall fade out (slow fade at end)
        fade_out = int(0.5 * self.sr)  # Long fade out
        if fade_out < len(song_audio):
            song_audio[-fade_out:] *= np.linspace(1, 0, fade_out)

        # Normalize
        song_audio = song_audio / (np.max(np.abs(song_audio)) + 1e-10) * 0.8

        return song_audio


def main():
    print("\n" + "=" * 60)
    print("DROSERA NURSERY RHYME")
    print("Educational Children's Song about Carnivorous Plants")
    print("Drosera Nursery Rhyme - Educational song about carnivorous plants")
    print("=" * 60)

    singer = DroseraSinger(sample_rate=16000)

    # Sing the Drosera nursery rhyme
    song_audio = singer.sing_drosera_rhyme()

    # Save
    wavfile.write("drosera_nursery_rhyme.wav", singer.sr, np.int16(song_audio * 32767))
    print("\nSaved: drosera_nursery_rhyme.wav")

    # Analysis plot
    plt.figure(figsize=(16, 10))

    # Waveform
    plt.subplot(3, 1, 1)
    t = np.arange(len(song_audio)) / singer.sr
    plt.plot(t, song_audio, color="green", alpha=0.8)
    plt.title("Drosera Nursery Rhyme - Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Spectrogram
    plt.subplot(3, 1, 2)
    f, t, Sxx = signal.spectrogram(song_audio, singer.sr, nperseg=512)
    freq_idx = np.where(f <= 4000)[0]
    plt.pcolormesh(
        t,
        f[freq_idx],
        10 * np.log10(Sxx[freq_idx, :] + 1e-10),
        shading="gouraud",
        cmap="YlGn",
    )
    plt.title("Spectrogram - Educational Song")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Power (dB)")

    # First few bars
    plt.subplot(3, 1, 3)
    detail_samples = int(4.0 * singer.sr)  # First 4 seconds
    t_detail = np.arange(min(detail_samples, len(song_audio))) / singer.sr
    plt.plot(t_detail, song_audio[: len(t_detail)], color="darkgreen", alpha=0.8)
    plt.title('First Line: "Little Drosera in the bog"')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("drosera_nursery_rhyme_analysis.png")
    print("Saved: drosera_nursery_rhyme_analysis.png")

    # Song information
    total_duration = sum(DROSERA_TIMING) * singer.quarter_note
    print(f"\nSong Statistics:")
    print(f"  Total notes: {len(DROSERA_MELODY)}")
    print(f"  Duration: {total_duration:.2f} seconds")
    print(f"  Tempo: {60.0 / singer.quarter_note:.1f} BPM")

    print("\n" + "=" * 60)
    print("Complete!")
    print("\nEducational Content:")
    print("- Little Drosera in the bog")
    print("- Catching tiny bugs to eat")
    print("- Sticky droplets shine so bright")
    print("- That's how plants can grow and thrive")
    print("\nThis nursery rhyme teaches children about")
    print("carnivorous plants in a fun, educational way!")
    print("=" * 60)


if __name__ == "__main__":
    main()
