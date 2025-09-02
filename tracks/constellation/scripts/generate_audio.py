#!/usr/bin/env python3
"""
Generate audio from PUBLIC DOMAIN satellite data
Commercial use allowed - no restrictions!
"""
import numpy as np
import pandas as pd
import wave
import os

def map_value(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    if in_max == in_min:
        return (out_min + out_max) / 2
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Generate a sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def generate_satellite_audio(sat_data, duration=2.0, sample_rate=44100):
    """Generate audio for a single satellite based on orbital parameters"""
    
    # Map altitude to pitch (280-1500km → 150-1000Hz)
    altitude = sat_data['altitude_km']
    frequency = map_value(altitude, 280, 1500, 150, 1000)
    
    # Map inclination to waveform modulation
    inclination = abs(sat_data['inclination'])
    mod_depth = map_value(inclination, 0, 100, 0.05, 0.4)
    
    # Generate base tone
    base_wave = generate_sine_wave(frequency, duration, sample_rate)
    
    # Add second harmonic
    harmonic_wave = generate_sine_wave(frequency * 2, duration, sample_rate, amplitude=0.2)
    
    # Add modulation
    mod_freq = 3 + (hash(sat_data['name']) % 5)  # 3-7Hz modulation
    t = np.linspace(0, duration, int(sample_rate * duration))
    modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    
    # Combine waves
    combined_wave = (base_wave + harmonic_wave) * modulation
    
    # Apply envelope
    envelope = np.ones_like(combined_wave)
    attack = int(0.01 * sample_rate)
    release = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    combined_wave *= envelope
    
    # Normalize
    max_val = np.max(np.abs(combined_wave))
    if max_val > 0:
        combined_wave = combined_wave / max_val * 0.8
    
    return combined_wave

def create_stereo_pan(mono_signal, pan_position):
    """Create stereo signal with panning"""
    pan_position = np.clip(pan_position, -1, 1)
    left_gain = np.sqrt(0.5 * (1.0 - pan_position))
    right_gain = np.sqrt(0.5 * (1.0 + pan_position))
    
    stereo = np.zeros((len(mono_signal), 2))
    stereo[:, 0] = mono_signal * left_gain
    stereo[:, 1] = mono_signal * right_gain
    
    return stereo

def save_wav(filename, audio_data, sample_rate=44100):
    """Save audio data to WAV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        nchannels = 2 if len(audio_data.shape) > 1 else 1
        sampwidth = 2
        framerate = sample_rate
        nframes = len(audio_data)
        
        wav_file.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'NONE'))
        
        if nchannels == 2:
            interleaved = np.empty((audio_data.shape[0] * 2,), dtype=np.int16)
            interleaved[0::2] = audio_data[:, 0]
            interleaved[1::2] = audio_data[:, 1]
            wav_file.writeframes(interleaved.tobytes())
        else:
            wav_file.writeframes(audio_data.tobytes())

def generate_all_audio():
    """Generate audio files for all satellites"""
    
    print("Loading open data...")
    tle_df = pd.read_csv('../data/open_tle_data.csv')
    pos_df = pd.read_csv('../data/open_positions.csv')
    
    os.makedirs('../audio', exist_ok=True)
    
    print(f"Generating audio for {len(tle_df)} satellites...")
    print(f"Data sources: {tle_df['data_source'].unique()}")
    print(f"Altitude range: {tle_df['altitude_km'].min():.1f} - {tle_df['altitude_km'].max():.1f} km")
    
    for idx, sat in tle_df.iterrows():
        sat_name = sat['name']
        
        # Generate audio
        audio_mono = generate_satellite_audio(sat, duration=2.0)
        
        # Get position for panning
        sat_positions = pos_df[pos_df['satellite_id'] == sat['satellite_id']]
        if not sat_positions.empty:
            avg_lat = sat_positions['latitude'].mean()
            pan = map_value(avg_lat, -100, 100, -1, 1)
        else:
            # Use RAAN for panning
            raan = sat.get('raan', 0)
            pan = map_value(raan % 180, 0, 180, -1, 1)
        
        # Create stereo
        audio_stereo = create_stereo_pan(audio_mono, pan)
        
        # Save with safe filename
        safe_name = sat['satellite_id'].replace(' ', '_').replace('/', '_')
        filename = f"../audio/{safe_name}.wav"
        save_wav(filename, audio_stereo)
        
        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{len(tle_df)} audio files...")
    
    print(f"Audio generation complete! Generated {len(tle_df)} WAV files")
    print("All data is PUBLIC DOMAIN - Commercial use allowed!")

if __name__ == "__main__":
    generate_all_audio()