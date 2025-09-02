#!/usr/bin/env python3

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import os
from datetime import datetime
import random

class SCSIDiskSynth:
    def __init__(self, sample_rate=48000):
        self.sr = sample_rate
        
    def generate_motor_sound(self, duration, motor_speed, wobble, harmonics=8):
        """Generate motor spinning sound with harmonics"""
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Wobble LFO
        wobble_lfo = 1 + 0.1 * np.sin(2 * np.pi * wobble * t)
        
        # Fundamental frequency
        fundamental = np.sin(2 * np.pi * motor_speed * wobble_lfo * t)
        
        # Harmonics
        harmonics_signal = np.zeros_like(t)
        for i in range(1, harmonics + 1):
            harmonics_signal += np.sin(2 * np.pi * motor_speed * i * wobble_lfo * t) / i
        harmonics_signal /= harmonics
        
        # Mix fundamental and harmonics
        motor = fundamental * 0.6 + harmonics_signal * 0.4
        return motor * 0.5  # motor_level (reduced for 10000 RPM)
    
    def generate_motor_sound_sweep(self, duration, start_freq, end_freq, wobble, harmonics=8):
        """Generate motor sound with continuous frequency sweep"""
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Smooth frequency interpolation
        freq_sweep = np.linspace(start_freq, end_freq, len(t))
        
        # Wobble LFO
        wobble_lfo = 1 + 0.1 * np.sin(2 * np.pi * wobble * t)
        
        # Calculate phase with continuous frequency change
        phase = 2 * np.pi * np.cumsum(freq_sweep * wobble_lfo) / self.sr
        fundamental = np.sin(phase)
        
        # Harmonics with swept frequency
        harmonics_signal = np.zeros_like(t)
        for i in range(1, harmonics + 1):
            harmonic_phase = 2 * np.pi * np.cumsum(freq_sweep * i * wobble_lfo) / self.sr
            harmonics_signal += np.sin(harmonic_phase) / i
        harmonics_signal /= harmonics
        
        # Mix fundamental and harmonics
        motor = fundamental * 0.6 + harmonics_signal * 0.4
        
        # Fade out completely at the end if stopping
        if end_freq == 0:
            fade_samples = int(0.8 * self.sr)  # 0.8 second fade
            fade = np.ones_like(motor)
            fade[-fade_samples:] = np.linspace(1, 0, fade_samples) ** 2  # Exponential fade
            motor *= fade
        
        return motor * 0.5  # motor_level (reduced for 10000 RPM)
    
    def generate_bearing_noise(self, duration, freq, resonance, noise_amount):
        """Generate bearing noise with bandpass filter"""
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Generate white noise
        noise = np.random.normal(0, noise_amount, len(t))
        
        # Design bandpass filter for bearing noise (2-8 kHz range)
        nyquist = self.sr / 2
        low = max((freq - freq/resonance) / nyquist, 0.01)
        high = min((freq + freq/resonance) / nyquist, 0.99)
        
        if low >= high:
            low = 0.01
            high = 0.99
            
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, noise)
        
        # Add some high-frequency metallic resonance
        high_freq = np.random.normal(0, noise_amount * 0.3, len(t))
        sos_high = signal.butter(2, [4000/nyquist, min(8000/nyquist, 0.99)], btype='band', output='sos')
        high_filtered = signal.sosfilt(sos_high, high_freq)
        
        return (filtered + high_filtered * 0.5) * 0.4  # bearing_level
    
    def generate_cogging(self, duration, cogging_freq, depth, sharpness, motor_speed):
        """Generate cogging/clicking sounds"""
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Generate pulse train
        pulse_phase = 2 * np.pi * cogging_freq * t
        pulses = np.where(np.sin(pulse_phase) > (1 - sharpness * 2), 1, 0)
        
        # Low-pass filter the pulses
        sos = signal.butter(2, 800 / (self.sr/2), output='sos')
        envelope = signal.sosfilt(sos, pulses)
        
        # Modulate with higher frequency (motor speed harmonics)
        modulated = envelope * np.sin(2 * np.pi * motor_speed * 3 * t) * depth
        
        # Add mechanical clicking
        click_noise = pulses * np.random.normal(0, 0.05, len(t))
        
        return (modulated + click_noise) * 0.15  # cogging_level
    
    def generate_seek(self, duration, freq_start, freq_end):
        """Generate seek chirp sound"""
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Envelope
        attack = 0.001
        attack_samples = int(attack * self.sr)
        env = np.ones_like(t)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        env[-attack_samples:] = np.linspace(1, 0, attack_samples)
        
        # Frequency sweep
        freq_sweep = np.linspace(freq_start, freq_end, len(t))
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.sr
        chirp = np.sin(phase)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(t))
        sos = signal.butter(2, 1000 / (self.sr/2), 'high', output='sos')
        noise_filtered = signal.sosfilt(sos, noise)
        
        # Mix chirp and noise
        seek = (chirp * 0.7 + noise_filtered * 0.3) * env
        
        return seek * 0.5  # seek_level
    
    def apply_compression(self, audio, threshold=-20, ratio=3, attack=0.002, release=0.1):
        """Simple compression"""
        # Convert to dB
        eps = 1e-10
        audio_db = 20 * np.log10(np.abs(audio) + eps)
        
        # Simple compression curve
        compressed_db = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        gain_db = compressed_db - audio_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Smooth the gain changes
        attack_samples = int(attack * self.sr)
        release_samples = int(release * self.sr)
        
        smoothed_gain = np.copy(gain_linear)
        for i in range(1, len(smoothed_gain)):
            if smoothed_gain[i] < smoothed_gain[i-1]:
                # Attack
                smoothed_gain[i] = smoothed_gain[i-1] * 0.9 + smoothed_gain[i] * 0.1
            else:
                # Release
                smoothed_gain[i] = smoothed_gain[i-1] * 0.99 + smoothed_gain[i] * 0.01
        
        return audio * smoothed_gain
    
    def render_segment(self, duration, params):
        """Render a segment with given parameters"""
        # Generate each component
        motor = self.generate_motor_sound(
            duration, 
            params['motor_speed'],
            params['motor_wobble']
        )
        
        bearing = self.generate_bearing_noise(
            duration,
            params['bearing_freq'],
            params['bearing_resonance'],
            params['bearing_noise']
        )
        
        cogging = self.generate_cogging(
            duration,
            params['cogging_freq'],
            params['cogging_depth'],
            params['cogging_sharpness'],
            params['motor_speed']
        )
        
        # Mix components
        mix = motor + bearing + cogging
        
        # Add seek events if present
        if 'seek_events' in params and params['seek_events']:
            for event in params['seek_events']:
                seek_time = event['offset'] * duration
                seek_duration = event['duration'] / 1000  # Convert ms to s
                
                if seek_time + seek_duration <= duration:
                    seek_sound = self.generate_seek(
                        seek_duration,
                        event['freq_start'],
                        event['freq_end']
                    )
                    
                    # Insert seek sound at the right position
                    start_idx = int(seek_time * self.sr)
                    end_idx = start_idx + len(seek_sound)
                    if end_idx <= len(mix):
                        mix[start_idx:end_idx] += seek_sound
        
        # Apply compression
        compressed = self.apply_compression(mix)
        
        # Limit to prevent clipping
        limited = np.clip(compressed, -0.95, 0.95)
        
        return limited
    
    def render_segment_with_sweep(self, duration, params, start_freq, end_freq):
        """Render a segment with frequency sweep for motor"""
        # Generate motor with sweep
        motor = self.generate_motor_sound_sweep(
            duration,
            start_freq,
            end_freq,
            params['motor_wobble']
        )
        
        # Generate other components (they fade with motor if stopping)
        bearing = self.generate_bearing_noise(
            duration,
            params['bearing_freq'],
            params['bearing_resonance'],
            params['bearing_noise']
        )
        
        cogging = self.generate_cogging(
            duration,
            params['cogging_freq'],
            params['cogging_depth'],
            params['cogging_sharpness'],
            start_freq if start_freq > 0 else 60  # Use start freq for cogging reference
        )
        
        # If stopping, fade everything
        if end_freq == 0:
            fade_samples = int(0.5 * self.sr)
            fade = np.ones(len(bearing))
            fade[-fade_samples:] = np.linspace(1, 0, fade_samples)
            bearing *= fade
            cogging *= fade
        
        # Mix components
        mix = motor + bearing + cogging
        
        # Apply compression
        compressed = self.apply_compression(mix)
        
        # Limit to prevent clipping
        limited = np.clip(compressed, -0.95, 0.95)
        
        return limited

class SCSIDiskAutomation:
    def __init__(self, duration=180, sample_rate=48000):
        self.duration = duration
        self.sample_rate = sample_rate
        self.synth = SCSIDiskSynth(sample_rate)
        self.output_dir = "../bounces"
        
        # SCSI disk behavioral patterns
        self.states = ['idle', 'spinning_up', 'seeking', 'reading', 'writing', 'spinning_down']
        
        # Realistic SCSI disk parameters (10000 RPM drive)
        # 10000 RPM = 166.7 Hz base frequency
        self.state_params = {
            'idle': {
                'motor_speed': (166, 167),  # 10000 RPM
                'motor_wobble': (0.05, 0.1),
                'bearing_noise': (0.15, 0.2),
                'cogging_depth': (0.03, 0.05),
            },
            'spinning_up': {
                'motor_speed': (0, 167),  # 0 to 10000 RPM
                'motor_wobble': (0.1, 0.2),
                'bearing_noise': (0.1, 0.3),
                'cogging_depth': (0.1, 0.2),
            },
            'seeking': {
                'motor_speed': (166, 168),
                'motor_wobble': (0.08, 0.12),
                'bearing_noise': (0.25, 0.35),
                'cogging_depth': (0.05, 0.08),
                'seek_pattern': True
            },
            'reading': {
                'motor_speed': (166.5, 167.5),
                'motor_wobble': (0.05, 0.08),
                'bearing_noise': (0.2, 0.25),
                'cogging_depth': (0.04, 0.06),
            },
            'writing': {
                'motor_speed': (166.7, 167.3),
                'motor_wobble': (0.04, 0.06),
                'bearing_noise': (0.22, 0.28),
                'cogging_depth': (0.04, 0.06),
            },
            'spinning_down': {
                'motor_speed': (167, 0),  # 10000 RPM to stop
                'motor_wobble': (0.08, 0.15),
                'bearing_noise': (0.1, 0.2),
                'cogging_depth': (0.05, 0.1),
            }
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_timeline(self):
        """Generate a timeline of disk operations"""
        timeline = []
        current_time = 0
        
        # More complex and continuous timeline with more segments
        # Start with spin-up (3 seconds)
        timeline.append({
            'start': 0,
            'duration': 3,  # Longer spin-up for 10000 RPM
            'state': 'spinning_up',
            'params': self.generate_state_params('spinning_up')
        })
        current_time = 3
        
        # Generate many short, varied operations for more continuous sound
        operations = [
            ('idle', 0.5),
            ('seeking', 0.3),
            ('reading', 1.2),
            ('seeking', 0.2),
            ('writing', 0.8),
            ('seeking', 0.4),
            ('reading', 0.6),
            ('idle', 0.3),
            ('seeking', 0.5),
            ('writing', 1.5),
            ('seeking', 0.3),
            ('reading', 0.7),
            ('seeking', 0.2),
            ('writing', 0.9),
            ('idle', 0.4),
            ('seeking', 0.6),
            ('reading', 1.1),
            ('seeking', 0.3),
            ('writing', 0.7),
            ('seeking', 0.4),
            ('reading', 0.5),
            ('idle', 0.2),
            ('seeking', 0.3),
            ('writing', 1.0),
            ('seeking', 0.5),
            ('reading', 0.8),
            ('seeking', 0.2),
            ('writing', 0.6),
            ('idle', 0.3),
            ('seeking', 0.4),
            ('reading', 0.9),
            ('seeking', 0.3),
            ('writing', 1.2),
            ('seeking', 0.2),
            ('reading', 0.4),
            ('idle', 0.5),
            ('seeking', 0.3),
            ('writing', 0.7),
            ('seeking', 0.4),
            ('reading', 1.0),
            # Continue with more operations to fill 50 seconds
            ('seeking', 0.3),
            ('writing', 0.9),
            ('idle', 0.4),
            ('reading', 1.3),
            ('seeking', 0.2),
            ('writing', 0.8),
            ('seeking', 0.5),
            ('reading', 0.6),
            ('idle', 0.3),
            ('seeking', 0.4),
            ('writing', 1.1),
            ('seeking', 0.3),
            ('reading', 0.9),
            ('seeking', 0.2),
            ('writing', 0.7),
            ('idle', 0.5),
            ('seeking', 0.3),
            ('reading', 1.0),
            ('seeking', 0.4),
            ('writing', 1.2),
            ('seeking', 0.2),
            ('reading', 0.5),
            ('idle', 0.3),
            ('seeking', 0.5),
            ('writing', 0.8),
            ('seeking', 0.3),
            ('reading', 1.1),
            ('seeking', 0.2),
            ('writing', 0.6),
            ('idle', 0.4),
            ('seeking', 0.3),
            ('reading', 0.9),
            ('seeking', 0.4),
            ('writing', 1.0),
            ('seeking', 0.3),
            ('reading', 0.7),
            ('idle', 0.2),
            ('seeking', 0.5),
            ('writing', 1.3),
            ('seeking', 0.2),
            ('reading', 0.8),
        ]
        
        # Add operations to timeline with slight random variations
        for state, base_duration in operations:
            # Add small random variation to duration (±20%)
            duration = base_duration * random.uniform(0.8, 1.2)
            
            timeline.append({
                'start': current_time,
                'duration': duration,
                'state': state,
                'params': self.generate_state_params(state)
            })
            current_time += duration
            
            # Stop if we're getting close to the end (leave room for spin-down)
            if current_time >= 47:  # Fixed at 47 seconds to leave 3 seconds for spin-down
                break
        
        # End with spin-down to complete stop (3 seconds)
        timeline.append({
            'start': current_time,
            'duration': 3,
            'state': 'spinning_down',
            'params': self.generate_state_params('spinning_down')
        })
        
        return timeline
    
    def generate_state_params(self, state):
        """Generate parameters for a given state"""
        params = {}
        state_def = self.state_params[state]
        
        for param, range_vals in state_def.items():
            if param == 'seek_pattern':
                params['seek_events'] = self.generate_seek_pattern()
            elif isinstance(range_vals, tuple):
                params[param] = random.uniform(range_vals[0], range_vals[1])
        
        # Add realistic SCSI parameters
        params['bearing_freq'] = random.uniform(3000, 6000)  # Higher freq for bearings
        params['bearing_resonance'] = random.uniform(8, 20)
        params['cogging_freq'] = random.uniform(30, 60)  # Higher for faster rotation
        params['cogging_sharpness'] = random.uniform(0.4, 0.7)
        
        return params
    
    def generate_seek_pattern(self):
        """Generate seek events within a time period"""
        events = []
        num_seeks = random.randint(2, 8)
        for _ in range(num_seeks):
            events.append({
                'offset': random.random(),  # 0-1 normalized time
                'duration': random.uniform(10, 50),  # ms - faster seeks
                'freq_start': random.uniform(500, 1200),
                'freq_end': random.uniform(800, 3000)
            })
        return sorted(events, key=lambda x: x['offset'])
    
    def render_timeline(self, timeline):
        """Render the entire timeline with smooth transitions"""
        print("Rendering SCSI disk timeline...")
        
        # Calculate total samples
        total_samples = int(self.duration * self.sample_rate)
        output = np.zeros(total_samples)
        
        for i, segment in enumerate(timeline):
            if i % 10 == 0:  # Print progress every 10 segments
                print(f"  Segment {i+1}/{len(timeline)}: {segment['state']} "
                      f"({segment['duration']:.1f}s)")
            
            # Handle special cases for spinning up/down
            if segment['state'] == 'spinning_up':
                # Smooth continuous frequency sweep
                segment_audio = self.synth.render_segment_with_sweep(
                    segment['duration'],
                    segment['params'],
                    0,    # start frequency (from stop)
                    167   # end frequency (10000 RPM)
                )
                
                # Add to output at the right position
                start_idx = int(segment['start'] * self.sample_rate)
                end_idx = start_idx + len(segment_audio)
                if end_idx <= len(output):
                    # Apply crossfade for smooth transition
                    if i > 0:
                        fade_samples = int(0.01 * self.sample_rate)  # 10ms crossfade
                        if fade_samples > 0 and start_idx >= fade_samples:
                            fade_in = np.linspace(0, 1, fade_samples)
                            segment_audio[:fade_samples] *= fade_in
                    output[start_idx:end_idx] += segment_audio
                        
            elif segment['state'] == 'spinning_down':
                # Smooth continuous frequency sweep
                segment_audio = self.synth.render_segment_with_sweep(
                    segment['duration'],
                    segment['params'],
                    167,  # start frequency (10000 RPM)
                    0     # end frequency (complete stop)
                )
                
                # Add to output at the right position
                start_idx = int(segment['start'] * self.sample_rate)
                end_idx = start_idx + len(segment_audio)
                if end_idx <= len(output):
                    output[start_idx:end_idx] += segment_audio
                        
            else:
                # Normal segment
                segment_audio = self.synth.render_segment(
                    segment['duration'],
                    segment['params']
                )
                
                # Apply short crossfade between segments for continuity
                fade_samples = int(0.005 * self.sample_rate)  # 5ms crossfade
                if i > 0 and fade_samples > 0:
                    # Fade in at start
                    fade_in = np.linspace(0, 1, min(fade_samples, len(segment_audio)))
                    segment_audio[:len(fade_in)] *= fade_in
                
                if i < len(timeline) - 1 and fade_samples > 0:
                    # Fade out at end
                    fade_out = np.linspace(1, 0, min(fade_samples, len(segment_audio)))
                    segment_audio[-len(fade_out):] *= fade_out
                
                # Add to output at the right position
                start_idx = int(segment['start'] * self.sample_rate)
                end_idx = start_idx + len(segment_audio)
                if end_idx <= len(output):
                    output[start_idx:end_idx] += segment_audio
        
        # Apply gentle overall compression for smoother dynamics
        output = self.synth.apply_compression(output, threshold=-15, ratio=2.5)
        
        # Final limiting
        output = np.clip(output, -0.95, 0.95)
        
        return output
    
    def generate(self, seed=None):
        """Main generation function"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Generating SCSI disk sounds ({self.duration}s)...")
        
        # Generate timeline
        timeline = self.generate_timeline()
        
        # Print timeline summary
        print("\nTimeline summary:")
        for segment in timeline:
            print(f"  {segment['start']:.1f}s - "
                  f"{segment['start'] + segment['duration']:.1f}s: "
                  f"{segment['state']}")
        
        # Render timeline
        output = self.render_timeline(timeline)
        
        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"scsi_disk_{timestamp}.wav")
        
        # Convert to 16-bit PCM
        output_int = np.int16(output * 32767)
        wavfile.write(output_file, self.sample_rate, output_int)
        
        print(f"\nSuccess! Output saved to: {output_file}")
        
        return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SCSI Disk Sound Synthesis')
    parser.add_argument('--duration', type=float, default=180,
                        help='Duration in seconds (default: 180)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Sample rate (default: 48000)')
    
    args = parser.parse_args()
    
    automation = SCSIDiskAutomation(
        duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    output = automation.generate(seed=args.seed)

if __name__ == "__main__":
    main()