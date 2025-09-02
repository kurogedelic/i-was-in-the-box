#!/usr/bin/env python3
"""
Algorithmic Music Generator
Cellular automaton-based music generation
"""

import numpy as np
import wave
import time
import sys
from scipy import signal
from dataclasses import dataclass
from typing import List, Tuple

# Constants
SR = 48000              # Sample rate
DURATION_SEC = 150
BPM = 120
BEAT_SAMPLES = int(SR * 60 / BPM / 4)  # 16th note samples
STEREO = True

@dataclass
class Note:
    """Note data"""
    pitch: int      # MIDI note number
    velocity: float # 0.0-1.0
    duration: int   # in 16th notes
    start: int      # start position in 16th notes

class CellularSequencer:
    """Cellular automaton sequencer"""
    
    def __init__(self, size=16):
        self.size = size
        self.grid = np.random.randint(0, 2, (size, size))
        self.generation = 0
        
    def step(self):
        """Update with Game of Life rules"""
        new_grid = np.zeros_like(self.grid)
        
        for i in range(self.size):
            for j in range(self.size):
                # 8-neighbor count
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % self.size
                        nj = (j + dj) % self.size
                        neighbors += self.grid[ni, nj]
                
                # Rules (musically adjusted)
                if self.grid[i, j] == 1:
                    # Survival: 2-4 neighbors (broader than Game of Life)
                    if neighbors in [2, 3, 4]:
                        new_grid[i, j] = 1
                else:
                    # Birth: 3 neighbors
                    if neighbors == 3:
                        new_grid[i, j] = 1
        
        self.grid = new_grid
        self.generation += 1
        
        # Reinitialize if dying
        if np.sum(self.grid) < 3:
            self.grid = np.random.randint(0, 2, (self.size, self.size))
        
        return self.grid

class MusicGenerator:
    """Music generator"""
    
    def __init__(self):
        self.sequencer = CellularSequencer()
        
        # Scales
        self.scales = {
            'intro': [60, 63, 65, 67, 70],        # C minor pentatonic
            'verse': [60, 62, 63, 65, 67, 68, 70], # C dorian
            'chorus': [60, 63, 65, 67, 70, 72],    # C minor with octave
            'bridge': [58, 60, 63, 65, 67, 70],    # Bb relative
            'outro': [60, 63, 65, 67],             # Simple ending
            'ending': [60]                          # Final note
        }
        
        # Chord progressions
        self.chord_progressions = {
            'intro': [(60, 63, 67), (58, 62, 65), (60, 63, 67), (55, 58, 62)],  # Cm - Bb - Cm - G
            'verse': [(60, 63, 67), (65, 68, 72), (58, 62, 65), (60, 63, 67)],  # Cm - F - Bb - Cm
            'chorus': [(60, 63, 67), (53, 56, 60), (58, 62, 65), (55, 58, 62)], # Cm - Ab - Bb - G
            'bridge': [(58, 62, 65), (55, 58, 62), (53, 56, 60), (60, 63, 67)], # Bb - G - Ab - Cm
            'outro': [(60, 63, 67), (58, 62, 65), (60, 63, 67)],                # Cm - Bb - Cm
            'ending': [(60, 63, 67)]                                            # Cm (final)
        }
        
        # Drum patterns
        self.drum_patterns = {
            'kick': [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,1,0],
            'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            'hihat': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            'ride': [0,0,0,0, 0,0,0,0, 1,0,0,1, 0,0,1,0]
        }
        
        self.section_lengths = {
            'intro': 8,
            'verse': 16,
            'chorus': 16,
            'verse2': 16,
            'chorus2': 16,
            'bridge': 16,
            'chorus3': 16,
            'outro': 12,
            'ending': 8
        }
        
    def grid_to_melody(self, grid, scale, length=16):
        """Generate melody from grid"""
        notes = []
        
        # Each row as time axis
        for step in range(min(length, grid.shape[0])):
            row = grid[step, :]
            active_cells = np.where(row == 1)[0]
            
            if len(active_cells) > 0:
                # Select pitch
                cell_idx = active_cells[0] % len(scale)
                pitch = scale[cell_idx]
                
                # Octave variation
                if len(active_cells) > 1:
                    octave = (active_cells[1] % 3) - 1  # -1, 0, 1
                    pitch += octave * 12
                
                # Velocity from active cells
                velocity = min(0.3 + len(active_cells) * 0.1, 1.0)
                
                notes.append(Note(pitch, velocity, 1, step))
        
        return notes
    
    def grid_to_rhythm(self, grid):
        """Generate rhythm from grid"""
        rhythm = {}
        
        # Kick: top-left
        kick_region = grid[:4, :4]
        rhythm['kick'] = [1 if np.sum(kick_region[:, i % 4]) > 1 else 0 for i in range(16)]
        
        # Snare: top-right
        snare_region = grid[:4, -4:]
        rhythm['snare'] = [0] * 16
        # 2nd and 4th beats
        for i in [4, 12]:  # 2nd and 4th beats
            if np.sum(snare_region[:, i % 4]) > 1:
                rhythm['snare'][i] = 1
        
        # HiHat: bottom
        hihat_region = grid[-4:, :]
        rhythm['hihat'] = [1 if np.sum(hihat_region[i % 4, :]) > 3 else 0 for i in range(16)]
        
        return rhythm
    
    def generate_section(self, section_name, bars):
        """Generate section data"""
        # Strip number suffixes
        base_name = section_name.replace('2', '').replace('3', '')
        scale = self.scales.get(base_name, self.scales['chorus'])
        chords = self.chord_progressions.get(base_name, self.chord_progressions['chorus'])
        
        section_data = {
            'melody': [],
            'bass': [],
            'chords': [],
            'drums': []
        }
        
        for bar in range(bars):
            # Update CA
            grid = self.sequencer.step()
            
            # Generate melody
            melody_notes = self.grid_to_melody(grid, scale)
            for note in melody_notes:
                note.start += bar * 16
            section_data['melody'].extend(melody_notes)
            
            # Bass
            chord_idx = bar % len(chords)
            root = chords[chord_idx][0]
            bass_pattern = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            for i, hit in enumerate(bass_pattern):
                if hit:
                    section_data['bass'].append(
                        Note(root - 24, 0.9, 2, bar * 16 + i)
                    )
            
            # Chords
            if bar % 2 == 0:
                for i in range(0, 16, 4):
                    chord = chords[chord_idx]
                    for pitch in chord:
                        section_data['chords'].append(
                            Note(pitch, 0.4, 4, bar * 16 + i)
                        )
            
            # Drums
            rhythm = self.grid_to_rhythm(grid)
            section_data['drums'].append(rhythm)
        
        return section_data
    
    def synthesize_note(self, note: Note, samples_per_16th, is_bass=False):
        """Synthesize note"""
        duration_samples = note.duration * samples_per_16th
        t = np.arange(duration_samples) / SR
        
        # Base frequency
        freq = 440 * 2**((note.pitch - 69) / 12)
        
        if is_bass:
            # Bass waveform
            wave = np.sin(2 * np.pi * freq * t) * 1.2
            wave += np.sin(2 * np.pi * freq * 0.5 * t) * 0.3
        else:
            # Normal waveform
            wave = np.sin(2 * np.pi * freq * t)
            wave += np.sin(2 * np.pi * freq * 2 * t) * 0.3
            wave += np.sin(2 * np.pi * freq * 3 * t) * 0.1
        
        # Envelope
        attack = int(0.01 * SR)
        decay = int(0.1 * SR)
        sustain_level = 0.7
        release = int(0.2 * SR)
        
        env = np.ones(duration_samples)
        
        # Attack
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        
        # Decay
        if decay > 0 and attack + decay < duration_samples:
            env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
            env[attack+decay:] = sustain_level
        
        # Release
        if release > 0 and duration_samples > release:
            env[-release:] = np.linspace(sustain_level, 0, release)
        
        return wave * env * note.velocity * 0.3
    
    def synthesize_drums(self, pattern, samples_per_16th):
        """Synthesize drums"""
        # Stereo arrays
        if STEREO:
            output_left = np.zeros(16 * samples_per_16th)
            output_right = np.zeros(16 * samples_per_16th)
        else:
            output = np.zeros(16 * samples_per_16th)
        
        for step in range(16):
            pos = step * samples_per_16th
            
            # Kick
            if pattern['kick'][step]:
                t = np.arange(samples_per_16th * 2) / SR
                # Low frequency emphasis
                kick = np.sin(2 * np.pi * 50 * np.exp(-t * 25) * t) * 1.5
                kick += np.sin(2 * np.pi * 100 * t) * 0.3 * np.exp(-t * 40)
                kick *= np.exp(-t * 30)
                
                if STEREO:
                    # Center
                    kick_len = min(len(kick), samples_per_16th)
                    output_left[pos:pos+kick_len] += kick[:kick_len] * 1.2
                    output_right[pos:pos+kick_len] += kick[:kick_len] * 1.2
                else:
                    kick_len = min(len(kick), samples_per_16th)
                    output[pos:pos+kick_len] += kick[:kick_len] * 1.2
            
            # Snare
            if pattern['snare'][step]:
                t = np.arange(samples_per_16th // 2) / SR
                snare = np.random.randn(len(t)) * 0.3
                snare += np.sin(2 * np.pi * 200 * t) * 0.5
                snare *= np.exp(-t * 20)
                
                if STEREO:
                    # Slightly left
                    output_left[pos:pos+len(snare)] += snare * 0.7
                    output_right[pos:pos+len(snare)] += snare * 0.5
                else:
                    output[pos:pos+len(snare)] += snare * 0.6
            
            # HiHat
            if pattern['hihat'][step]:
                t = np.arange(samples_per_16th // 4) / SR
                hihat = np.random.randn(len(t))
                hihat *= np.exp(-t * 100)
                
                if STEREO:
                    # Pan hihat
                    pan = 0.5 + 0.4 * np.sin(step * np.pi / 4)
                    output_left[pos:pos+len(hihat)] += hihat * 0.3 * (1 - pan)
                    output_right[pos:pos+len(hihat)] += hihat * 0.3 * pan
                else:
                    output[pos:pos+len(hihat)] += hihat * 0.2
        
        if STEREO:
            return np.stack([output_left, output_right])
        else:
            return output
    
    def generate_music(self):
        """Generate complete music"""
        print("Generating algorithmic music structure...")
        
        # Generate sections
        sections = {}
        section_order = ['intro', 'verse', 'chorus', 'verse2', 'chorus2', 
                        'bridge', 'chorus3', 'outro', 'ending']
        
        for section in section_order:
            bars = self.section_lengths[section]
            print(f"  - Generating {section}: {bars} bars")
            sections[section] = self.generate_section(section, bars)
        
        # Convert to audio
        total_16ths = sum(self.section_lengths[s] * 16 for s in section_order)
        total_samples = total_16ths * BEAT_SAMPLES
        
        if STEREO:
            output = np.zeros((2, total_samples))
        else:
            output = np.zeros(total_samples)
        samples_per_16th = BEAT_SAMPLES
        
        current_pos = 0
        
        for section_name in section_order:
            section = sections[section_name]
            section_bars = self.section_lengths[section_name]
            section_samples = section_bars * 16 * samples_per_16th
            
            print(f"  - Synthesizing {section_name}...")
            
            # Section dynamics
            section_volume = 1.0
            drums_enabled = True
            
            if section_name == 'intro':
                section_volume = 0.7
                drums_enabled = False
            elif section_name == 'bridge':
                section_volume = 0.8
            elif section_name in ['outro', 'ending']:
                section_volume = 0.6
            
            # Melody
            for note in section['melody']:
                start_sample = current_pos + note.start * samples_per_16th
                note_audio = self.synthesize_note(note, samples_per_16th) * section_volume
                end_sample = min(start_sample + len(note_audio), total_samples)
                if start_sample < total_samples:
                    if STEREO:
                        # Slightly right
                        output[0, start_sample:end_sample] += note_audio[:end_sample-start_sample] * 0.4
                        output[1, start_sample:end_sample] += note_audio[:end_sample-start_sample] * 0.6
                    else:
                        output[start_sample:end_sample] += note_audio[:end_sample-start_sample]
            
            # Bass
            for note in section['bass']:
                start_sample = current_pos + note.start * samples_per_16th
                note_audio = self.synthesize_note(note, samples_per_16th, is_bass=True) * 1.5
                end_sample = min(start_sample + len(note_audio), total_samples)
                if start_sample < total_samples:
                    if STEREO:
                        # Center
                        output[0, start_sample:end_sample] += note_audio[:end_sample-start_sample]
                        output[1, start_sample:end_sample] += note_audio[:end_sample-start_sample]
                    else:
                        output[start_sample:end_sample] += note_audio[:end_sample-start_sample]
            
            # Chords
            for note in section['chords']:
                start_sample = current_pos + note.start * samples_per_16th
                note_audio = self.synthesize_note(note, samples_per_16th)
                end_sample = min(start_sample + len(note_audio), total_samples)
                if start_sample < total_samples:
                    if STEREO:
                        # Wide
                        output[0, start_sample:end_sample] += note_audio[:end_sample-start_sample] * 0.3
                        output[1, start_sample:end_sample] += note_audio[:end_sample-start_sample] * 0.3
                    else:
                        output[start_sample:end_sample] += note_audio[:end_sample-start_sample] * 0.5
            
            # Drums
            if drums_enabled:
                for bar_idx, drum_pattern in enumerate(section['drums']):
                    drum_start = current_pos + bar_idx * 16 * samples_per_16th
                    drum_audio = self.synthesize_drums(drum_pattern, samples_per_16th)
                    
                    if STEREO:
                        drum_end = min(drum_start + drum_audio.shape[1], total_samples)
                        if drum_start < total_samples:
                            output[:, drum_start:drum_end] += drum_audio[:, :drum_end-drum_start]
                    else:
                        drum_end = min(drum_start + len(drum_audio), total_samples)
                        if drum_start < total_samples:
                            output[drum_start:drum_end] += drum_audio[:drum_end-drum_start]
            
            current_pos += section_samples
        
        # Normalize and limit
        output = np.tanh(output * 0.5)
        
        # Fade in
        fade_in_samples = int(2.0 * SR)
        if STEREO:
            for ch in range(2):
                output[ch, :fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
        else:
            output[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
        
        # Fade out
        fade_out_samples = int(15.0 * SR)
        fade_curve = np.exp(-np.linspace(0, 5, fade_out_samples))
        
        if STEREO:
            if output.shape[1] > fade_out_samples:
                for ch in range(2):
                    output[ch, -fade_out_samples:] *= fade_curve
        else:
            if len(output) > fade_out_samples:
                output[-fade_out_samples:] *= fade_curve
        
        # Reverb
        if STEREO:
            # Stereo reverb
            reverb = np.zeros_like(output)
            # Different delays per channel
            delays_l = [int(0.037 * SR), int(0.067 * SR), int(0.089 * SR)]
            delays_r = [int(0.041 * SR), int(0.071 * SR), int(0.097 * SR)]
            gains = [0.25, 0.15, 0.08]
            
            for delay, gain in zip(delays_l, gains):
                reverb[0, delay:] += output[0, :-delay] * gain
            for delay, gain in zip(delays_r, gains):
                reverb[1, delay:] += output[1, :-delay] * gain
                
            # Cross-feedback
            reverb[0, delays_l[0]:] += output[1, :-delays_l[0]] * 0.1
            reverb[1, delays_r[0]:] += output[0, :-delays_r[0]] * 0.1
            
            output += reverb
        else:
            # Mono reverb
            delay_samples = int(0.05 * SR)
            reverb = np.zeros_like(output)
            reverb[delay_samples:] = output[:-delay_samples] * 0.3
            reverb[delay_samples*2:] += output[:-delay_samples*2] * 0.15
            reverb[delay_samples*3:] += output[:-delay_samples*3] * 0.08
            output += reverb
        
        # Final limiting
        output = np.tanh(output * 0.5)
        
        if STEREO:
            # Transpose for stereo
            return output.T[:int(SR * DURATION_SEC)]
        else:
            return output[:int(SR * DURATION_SEC)]

def main():
    """Main"""
    print("=== Algorithmic Music Generator ===")
    print(f"Duration: {DURATION_SEC} seconds")
    print(f"Tempo: {BPM} BPM")
    print("")
    
    # Generate music
    generator = MusicGenerator()
    start_time = time.time()
    
    audio = generator.generate_music()
    
    # Save WAV
    wav_path = "field.wav"
    print(f"\nSaving to {wav_path}...")
    
    wf = wave.open(wav_path, 'wb')
    wf.setnchannels(2 if STEREO else 1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(SR)
    
    audio_int = (audio * 32767).astype(np.int16)
    wf.writeframes(audio_int.tobytes())
    wf.close()
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Completed!")
    print(f"Output: {wav_path}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"File size: {len(audio) * 2 / 1024 / 1024:.2f} MB")
    
    print(f"\nMusic Structure:")
    print("  Intro (8 bars) → Verse (16 bars) → Chorus (16 bars)")
    print("  → Verse2 (16 bars) → Chorus2 (16 bars) → Bridge (16 bars)")
    print("  → Chorus3 (16 bars) → Outro (12 bars) → Ending (8 bars)")
    print(f"\nTotal: 128 bars at {BPM} BPM = 2:30 with natural fade")

if __name__ == "__main__":
    main()