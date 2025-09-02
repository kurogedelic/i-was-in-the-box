#!/usr/bin/env python3

import subprocess
import numpy as np
import os
from datetime import datetime
import json
import random
import sys

class SCSIDiskAutomation:
    def __init__(self, duration=180, sample_rate=48000):
        self.duration = duration
        self.sample_rate = sample_rate
        self.temp_dir = "temp_renders"
        self.output_dir = "../bounces"
        
        # SCSI disk behavioral patterns
        self.states = ['idle', 'spinning_up', 'seeking', 'reading', 'writing', 'spinning_down']
        
        # Parameter ranges for different states
        self.state_params = {
            'idle': {
                'motor_speed': (90, 100),
                'motor_wobble': (0.01, 0.02),
                'bearing_noise': (0.05, 0.1),
                'cogging_depth': (0.05, 0.1),
            },
            'spinning_up': {
                'motor_speed': (60, 120),
                'motor_wobble': (0.03, 0.05),
                'bearing_noise': (0.1, 0.2),
                'cogging_depth': (0.15, 0.25),
            },
            'seeking': {
                'motor_speed': (118, 122),
                'motor_wobble': (0.02, 0.03),
                'bearing_noise': (0.15, 0.25),
                'cogging_depth': (0.1, 0.15),
                'seek_pattern': True
            },
            'reading': {
                'motor_speed': (119, 121),
                'motor_wobble': (0.015, 0.025),
                'bearing_noise': (0.12, 0.18),
                'cogging_depth': (0.08, 0.12),
            },
            'writing': {
                'motor_speed': (119.5, 120.5),
                'motor_wobble': (0.01, 0.015),
                'bearing_noise': (0.13, 0.19),
                'cogging_depth': (0.09, 0.13),
            },
            'spinning_down': {
                'motor_speed': (120, 60),
                'motor_wobble': (0.025, 0.04),
                'bearing_noise': (0.08, 0.15),
                'cogging_depth': (0.1, 0.2),
            }
        }
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_timeline(self):
        """Generate a timeline of disk operations"""
        timeline = []
        current_time = 0
        
        # Start with spin-up
        timeline.append({
            'start': 0,
            'duration': 3,
            'state': 'spinning_up',
            'params': self.generate_state_params('spinning_up')
        })
        current_time = 3
        
        while current_time < self.duration - 5:  # Leave time for spin-down
            # Choose next operation
            if random.random() < 0.4:  # 40% chance of seek
                state = 'seeking'
                duration = random.uniform(0.5, 2)
            elif random.random() < 0.3:  # 30% chance of read
                state = 'reading'
                duration = random.uniform(2, 8)
            elif random.random() < 0.2:  # 20% chance of write
                state = 'writing'
                duration = random.uniform(3, 10)
            else:  # 10% idle
                state = 'idle'
                duration = random.uniform(1, 3)
            
            timeline.append({
                'start': current_time,
                'duration': duration,
                'state': state,
                'params': self.generate_state_params(state)
            })
            
            current_time += duration
        
        # End with spin-down
        timeline.append({
            'start': current_time,
            'duration': 5,
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
        
        # Add some randomization to other parameters
        params['bearing_freq'] = random.uniform(1500, 3000)
        params['bearing_resonance'] = random.uniform(5, 15)
        params['cogging_freq'] = random.uniform(15, 30)
        params['cogging_sharpness'] = random.uniform(0.2, 0.5)
        
        return params
    
    def generate_seek_pattern(self):
        """Generate seek events within a time period"""
        events = []
        num_seeks = random.randint(2, 8)
        for _ in range(num_seeks):
            events.append({
                'offset': random.random(),  # 0-1 normalized time
                'duration': random.uniform(20, 100),  # ms
                'freq_start': random.uniform(300, 800),
                'freq_end': random.uniform(1000, 2500)
            })
        return sorted(events, key=lambda x: x['offset'])
    
    def interpolate_params(self, timeline, time):
        """Get interpolated parameters at a specific time"""
        for i, segment in enumerate(timeline):
            if segment['start'] <= time < segment['start'] + segment['duration']:
                # We're in this segment
                progress = (time - segment['start']) / segment['duration']
                
                # Handle special cases for spinning up/down
                if segment['state'] == 'spinning_up':
                    motor_speed = 60 + (120 - 60) * progress
                elif segment['state'] == 'spinning_down':
                    motor_speed = 120 - (120 - 60) * progress
                else:
                    motor_speed = segment['params']['motor_speed']
                
                return {
                    'motor_speed': motor_speed,
                    'motor_wobble': segment['params']['motor_wobble'],
                    'bearing_freq': segment['params']['bearing_freq'],
                    'bearing_resonance': segment['params']['bearing_resonance'],
                    'bearing_noise': segment['params']['bearing_noise'],
                    'cogging_freq': segment['params']['cogging_freq'],
                    'cogging_depth': segment['params']['cogging_depth'],
                    'cogging_sharpness': segment['params']['cogging_sharpness'],
                    'state': segment['state'],
                    'seek_events': segment['params'].get('seek_events', [])
                }
        
        # Default idle state
        return self.generate_state_params('idle')
    
    def render_segment(self, params, duration, segment_index):
        """Render a segment with given parameters"""
        output_file = os.path.join(self.temp_dir, f"segment_{segment_index:04d}.wav")
        cpp_file = os.path.join(self.temp_dir, f"segment_{segment_index:04d}.cpp")
        exec_file = os.path.join(self.temp_dir, f"segment_{segment_index:04d}")
        
        # Create a modified DSP file with parameter values hardcoded
        modified_dsp = f"""import("stdfaust.lib");

// SCSI Disk Sound Synthesis - Automated segment {segment_index}

// Motor Parameters (hardcoded for this segment)
motor_speed = {params['motor_speed']:.2f};
motor_wobble = {params['motor_wobble']:.4f};

// Bearing Noise Parameters  
bearing_freq = {params['bearing_freq']:.1f};
bearing_resonance = {params['bearing_resonance']:.1f};
bearing_noise_amount = {params['bearing_noise']:.3f};

// Cogging Parameters
cogging_freq = {params['cogging_freq']:.1f};
cogging_depth = {params['cogging_depth']:.3f};
cogging_sharpness = {params['cogging_sharpness']:.3f};

// Seek Sound Parameters (disabled for now)
seek_trigger = 0;
seek_duration = 50;
seek_freq_start = 500;
seek_freq_end = 1500;

// Output Mix
motor_level = 0.7;
bearing_level = 0.3;
cogging_level = 0.2;
seek_level = 0.5;

// Motor synthesis with harmonics and wobble
motor_base = os.osc(motor_speed) * 0.5;
motor_wobble_lfo = os.osc(motor_wobble) * 0.1 + 1;
motor_fundamental = os.osc(motor_speed * motor_wobble_lfo);

// Fixed number of harmonics (8)
motor_harmonics_gen = par(i, 8, 
    os.osc(motor_speed * (i+1) * motor_wobble_lfo) * (1.0 / (i+1))
) :> _ / 8;

motor_sound = (motor_fundamental * 0.6 + motor_harmonics_gen * 0.4) * motor_level;

// Bearing noise - bandpassed noise with resonance
bearing_noise = no.noise * bearing_noise_amount;
bearing_filtered = bearing_noise : fi.resonbp(bearing_freq, bearing_resonance, 1);
bearing_sound = bearing_filtered * bearing_level;

// Cogging - periodic clicking/ticking
cogging_pulse = os.lf_pulsetrain(cogging_freq, cogging_sharpness);
cogging_envelope = cogging_pulse : fi.lowpass(2, 500);
cogging_modulated = cogging_envelope * os.osc(motor_speed * 3) * cogging_depth;
cogging_sound = cogging_modulated * cogging_level;

// Seek sound - chirp with envelope
seek_env = seek_trigger : en.ar(0.001, seek_duration/1000);
seek_freq_sweep = seek_freq_start + (seek_freq_end - seek_freq_start) * seek_env;
seek_chirp = os.osc(seek_freq_sweep) * seek_env;
seek_noise_burst = no.noise * seek_env : fi.highpass(2, 1000);
seek_sound = (seek_chirp * 0.7 + seek_noise_burst * 0.3) * seek_level;

// Mix all components
output = motor_sound + bearing_sound + cogging_sound + seek_sound;

// Subtle compression and limiting
compressed = output : co.compressor_mono(3, -20, 0.002, 0.1);
limited = compressed : co.limiter_1176_R4_mono;

process = limited <: _,_;
"""
        
        # Write modified DSP file
        temp_dsp = os.path.join(self.temp_dir, f"temp_{segment_index:04d}.dsp")
        with open(temp_dsp, 'w') as f:
            f.write(modified_dsp)
        
        try:
            # Compile to C++ with console architecture (outputs WAV to stdout)
            compile_cmd = f"faust -a sndfile.cpp -o {cpp_file} {temp_dsp}"
            subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
            
            # Compile C++ to executable
            compile_cpp = f"c++ -O3 {cpp_file} -o {exec_file} `pkg-config --cflags --libs sndfile`"
            subprocess.run(compile_cpp, shell=True, check=True, capture_output=True)
            
            # Run executable to generate WAV
            num_samples = int(duration * self.sample_rate)
            run_cmd = f"{exec_file} -n {num_samples} > {output_file}"
            subprocess.run(run_cmd, shell=True, check=True, capture_output=True)
            
            # Clean up temp files
            os.remove(temp_dsp)
            os.remove(cpp_file)
            os.remove(exec_file)
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"Error rendering segment {segment_index}: {e}")
            print(f"Command output: {e.stderr.decode() if e.stderr else 'No error output'}")
            return None
    
    def render_timeline(self, timeline):
        """Render the entire timeline"""
        segments = []
        
        print("Rendering SCSI disk timeline...")
        for i, segment in enumerate(timeline):
            print(f"  Segment {i+1}/{len(timeline)}: {segment['state']} "
                  f"({segment['duration']:.1f}s)")
            
            rendered = self.render_segment(
                segment['params'],
                segment['duration'],
                i
            )
            if rendered:
                segments.append(rendered)
        
        return segments
    
    def concatenate_segments(self, segments):
        """Concatenate all segments into final output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"scsi_disk_{timestamp}.wav")
        
        if len(segments) == 0:
            print("No segments to concatenate")
            return None
        
        # Use sox to concatenate
        sox_cmd = f"sox {' '.join(segments)} {output_file}"
        
        try:
            subprocess.run(sox_cmd, shell=True, check=True)
            print(f"Final output: {output_file}")
            
            # Clean up temp files
            for segment in segments:
                os.remove(segment)
            
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating segments: {e}")
            return None
    
    def generate(self, seed=None):
        """Main generation function"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Generating SCSI disk sounds ({self.duration}s)...")
        
        # Generate timeline
        timeline = self.generate_timeline()
        
        # Save timeline for reference
        timeline_file = os.path.join(self.temp_dir, "timeline.json")
        with open(timeline_file, 'w') as f:
            json.dump(timeline, f, indent=2, default=str)
        print(f"Timeline saved to {timeline_file}")
        
        # Render segments
        segments = self.render_timeline(timeline)
        
        # Concatenate into final output
        output = self.concatenate_segments(segments)
        
        if output:
            print(f"\nSuccess! Output saved to: {output}")
            
            # Print timeline summary
            print("\nTimeline summary:")
            for segment in timeline:
                print(f"  {segment['start']:.1f}s - "
                      f"{segment['start'] + segment['duration']:.1f}s: "
                      f"{segment['state']}")
        
        return output

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SCSI Disk Sound Automation')
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
    
    if not output:
        sys.exit(1)

if __name__ == "__main__":
    main()