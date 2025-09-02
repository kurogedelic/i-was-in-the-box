# EMRSP - Experiments in Momentary Recognition of Sensation and Perception

感覚と感知の刹那的な認識における実験

## Overview

EMRSP is an experimental audio project exploring the boundaries of granular synthesis and dynamic effects processing. The system creates evolving soundscapes by morphing between two audio sources through time-varying granular synthesis with multiple layers of effects.

## Features

- **Dynamic Granular Synthesis**: Time-varying grain density and morphing
- **Evolving Effects**: Real-time modulation of reverb, delay, and filtering
- **Spectral Processing**: Frequency-domain manipulation and spectral freezing
- **Chaotic Grain Positioning**: Non-linear grain extraction patterns
- **Spatial Audio**: Dynamic stereo panning with rotating sound fields

## Installation

```bash
pip install numpy scipy soundfile sounddevice pydub
```

## Usage

### Basic Usage

```bash
python emrsp_granular.py
```

### With Custom Parameters

```bash
python emrsp_granular.py --source1 sources/sound1.wav --source2 sources/sound2.wav --duration 3.0 --seed 42
```

### Command Line Options

- `--source1`: Path to first source audio file (default: sources/source_1.wav)
- `--source2`: Path to second source audio file (default: sources/source_2.wav)
- `--duration`: Output duration in minutes (default: 2.0)
- `--seed`: Random seed for reproducible results (optional)
- `--no-play`: Skip playback after rendering
- `--output`: Custom output filename (default: auto-generated with timestamp)

## Project Structure

```
EMRSP/
├── sources/           # Input audio files
│   ├── source_1.wav
│   └── source_2.wav
├── outputs/           # Generated audio files
│   └── emrsp_*.wav
├── emrsp_granular.py  # Main synthesis engine
└── README.md
```

## Technical Details

### Granular Engine

The synthesis engine operates in segments, each with evolving parameters:

1. **Grain Density**: Varies from sparse to dense throughout the composition
2. **Morphing**: Non-linear transition between source materials
3. **Window Functions**: Adaptive windowing (asymmetric, Gaussian, Hann)
4. **Pitch Modulation**: Micro-variations in grain playback speed

### Effects Processing

- **Delay**: Time-varying delay with feedback modulation
- **Reverb**: Allpass filter chain with dynamic room size and damping
- **Filtering**: Bandpass, lowpass, and highpass with resonance control
- **Spectral Freeze**: Phase randomization for textural effects

### Audio Processing

- Automatic conversion from M4A to WAV format
- Resampling to 48kHz for consistent processing
- 24-bit PCM output for high quality
- Soft limiting and DC offset removal

## Artistic Concept

EMRSP explores the momentary recognition patterns in human perception, creating a sonic environment where familiar sounds dissolve and reconstitute in unexpected ways. The system mimics the fleeting nature of sensory processing, where conscious recognition occurs in discrete moments rather than continuous streams.

## License

This project is for experimental and artistic purposes.

## Author

Created as part of the InTheBox/S experimental audio series.