#!/usr/bin/env python3
"""
Simplified altitude sweep mix - each satellite sweeps from 280km to operational altitude
Optimized for faster generation
"""
import numpy as np
import pandas as pd
import wave
import os

def create_sweep_mix():
    """Create mix with altitude sweeps"""
    print("="*60)
    print("CREATING ALTITUDE SWEEP MIX")
    print("280km (deployment) → operational altitude")
    print("="*60)
    
    # Load data
    tle_df = pd.read_csv('../data/group2_tle_data.csv')
    
    # Constants
    DEPLOYMENT_ALT = 280  # km
    mix_duration = 120  # seconds
    sample_rate = 44100
    total_samples = mix_duration * sample_rate
    
    # Prepare data - handle different column names
    if 'norad_id' in tle_df.columns:
        tle_df['id_numeric'] = tle_df['norad_id'].astype(int)
    elif 'satellite_id' in tle_df.columns:
        # For open data, extract numeric part from satellite_id
        tle_df['id_numeric'] = tle_df['satellite_id'].str.extract('(\d+)').astype(float).fillna(0).astype(int)
    else:
        # Fallback: use index
        tle_df['id_numeric'] = range(len(tle_df))
    
    min_id = tle_df['id_numeric'].min()
    max_id = tle_df['id_numeric'].max()
    tle_df['timeline_pos'] = ((tle_df['id_numeric'] - min_id) / (max_id - min_id)) * mix_duration if max_id > min_id else 0
    
    # Sort by timeline
    tle_df_sorted = tle_df.sort_values('timeline_pos')
    
    # Initialize mix
    mix_buffer = np.zeros((total_samples, 2), dtype=np.float32)
    
    print(f"Processing {len(tle_df)} satellites...")
    print(f"Deployment: {DEPLOYMENT_ALT}km → Operational: {tle_df['altitude_km'].min():.0f}-{tle_df['altitude_km'].max():.0f}km")
    
    processed = 0
    
    # Process only every Nth satellite for speed (but all eventually)
    for idx, sat in tle_df_sorted.iterrows():
        timeline_pos = sat['timeline_pos']
        operational_alt = sat['altitude_km']
        sat_id = sat['id_numeric']
        
        # Duration for this satellite
        remaining_time = mix_duration - timeline_pos
        if remaining_time <= 1:
            continue
        
        # Samples for this satellite
        num_samples = int(remaining_time * sample_rate)
        t = np.linspace(0, remaining_time, num_samples)
        
        # Altitude profile: exponential rise from deployment to operational
        # Fast initial rise, then gradual approach
        rise_factor = 1 - np.exp(-3 * t / remaining_time)
        altitude_profile = DEPLOYMENT_ALT + (operational_alt - DEPLOYMENT_ALT) * rise_factor
        
        # Convert altitude to frequency (280-600km → 150-800Hz)
        frequencies = 150 + (altitude_profile - 280) * (650 / 320)
        
        # Generate swept frequency sine wave (simplified)
        phase = np.cumsum(2 * np.pi * frequencies / sample_rate)
        audio = np.sin(phase) * 0.3
        
        # Add simple harmonic
        audio += np.sin(2 * phase) * 0.1
        
        # Envelope
        envelope = np.ones_like(audio)
        fade_in = int(0.5 * sample_rate)
        fade_out = int(1.0 * sample_rate)
        
        if fade_in < len(envelope):
            envelope[:fade_in] *= np.linspace(0, 1, fade_in)
        if fade_out < len(envelope):
            envelope[-fade_out:] *= np.linspace(1, 0, fade_out)
        
        audio *= envelope
        
        # Simple stereo (based on sat ID for variety)
        pan = (sat_id % 200 - 100) / 100.0
        left_gain = np.sqrt(0.5 * (1.0 - pan))
        right_gain = np.sqrt(0.5 * (1.0 + pan))
        
        stereo = np.zeros((len(audio), 2))
        stereo[:, 0] = audio * left_gain
        stereo[:, 1] = audio * right_gain
        
        # Add to mix
        start_sample = int(timeline_pos * sample_rate)
        end_sample = min(start_sample + len(stereo), total_samples)
        actual_length = end_sample - start_sample
        
        if actual_length > 0:
            # Reduce amplitude as more satellites join
            amplitude = 0.05 / np.sqrt(1 + processed * 0.01)
            mix_buffer[start_sample:end_sample] += stereo[:actual_length] * amplitude
            processed += 1
        
        if processed % 100 == 0:
            print(f"  Processed {processed} satellites...")
    
    print(f"Total processed: {processed} satellites")
    
    # Normalize
    max_val = np.max(np.abs(mix_buffer))
    if max_val > 0:
        mix_buffer = mix_buffer / max_val * 0.9
    
    # Save
    os.makedirs('../mix', exist_ok=True)
    output_file = '../mix/final_mix_sweep.wav'
    
    # Convert to 16-bit
    mix_buffer = np.clip(mix_buffer, -1.0, 1.0)
    mix_buffer = (mix_buffer * 32767).astype(np.int16)
    
    # Write WAV
    with wave.open(output_file, 'wb') as wav:
        wav.setparams((2, 2, sample_rate, total_samples, 'NONE', 'NONE'))
        interleaved = np.empty((mix_buffer.shape[0] * 2,), dtype=np.int16)
        interleaved[0::2] = mix_buffer[:, 0]
        interleaved[1::2] = mix_buffer[:, 1]
        wav.writeframes(interleaved.tobytes())
    
    print("\n" + "="*60)
    print("✅ ALTITUDE SWEEP MIX COMPLETE")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"Duration: {mix_duration}s")
    print(f"Satellites: {processed}")
    print("\nEach satellite:")
    print(f"  • Starts at {DEPLOYMENT_ALT}km (low pitch ~150Hz)")
    print("  • Sweeps UP to operational altitude (higher pitch)")
    print("  • Continuous sweep (no gaps)")
    
    return output_file

if __name__ == "__main__":
    create_sweep_mix()