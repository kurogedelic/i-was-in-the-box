# Stress-Strain Curve Sonification - 応力歪曲線のソニフィケーション

An experimental music project that sonifies metal deformation and fracture through stress-strain curves.
金属の応力歪曲線を音響化するエクスペリメンタル音楽プロジェクト

## Overview

This project converts material stress-strain data into audio, creating an artistic representation of metal deformation and fracture. The sonification follows the actual physics of metal behavior through different regions: elastic deformation, yield point, plastic deformation, necking, and final fracture.

## Files

### Core Audio Files
- `stress-strain_curve.wav` - 30-second sonification output
  - 0-20s: Data-driven sonification from CSV
  - 20-30s: Fracture aftermath and resonance decay
- `stress_strain_serrations.csv` - Real stress-strain data with serrations (Portevin-Le Chatelier effect)

### Processing Scripts
- `stress-strain_curve.py` - Converts CSV data to Csound with interpolation
  - Cubic interpolation to 1000 data points
  - Region detection (elastic, yield, plastic, necking, fracture)
  - Serration event detection
  - Generates layered synthesis with multiple instruments
  - Self-contained orchestra definition

### Csound Files
- `stress-strain_curve.csd` - Generated Csound file with 6 instruments:
  1. Main metal stress sonification
  2. Creaking and cracking texture
  3. Low frequency rumble
  4. Impact/dislocation events
  98. Delay effect
  99. Global reverb

### Web Interface
- `index.html` - Interactive web UI with real-time stress-strain curve visualization
- `steel-data.js` - Material properties database (7 metal types)
- `worklet-processor.js` - Web Audio worklet for real-time synthesis
- `ssc.js` - Compiled Faust processor

### Faust DSP
- `ssc.dsp` - Original Faust DSP code with:
  - 5 material types (Steel, Aluminum, Titanium, Cast Iron, Copper)
  - 4 auto-play patterns
  - Region-specific synthesis

## Usage

### Generate Sonification from CSV
```bash
python3 stress-strain_curve.py
csound stress-strain_curve.csd
```

### Web Interface
Open `index.html` in a browser to interact with the real-time stress-strain curve visualization and audio synthesis.

## Technical Details

### Sonification Mapping
- **Stress (MPa)** → Frequency (60-180 Hz base + harmonics)
- **Strain** → Modulation depth and texture density
- **Region** → Synthesis method and filtering
  - Elastic: Clean metallic tone
  - Yield: Bright resonance with impact
  - Plastic: Band-passed with creaking
  - Necking: Resonant filter with instability
  - Fracture: Explosive with heavy distortion

### Audio Synthesis Layers
1. Fundamental metallic tone with vibrato
2. Harmonic resonances (2.76x, 5.40x, 8.93x)
3. Inharmonic partials (1.73x, 3.89x, 7.23x)
4. Noise component (filtered by region)
5. Stress-dependent distortion (tanh waveshaping)

### Effects Processing
- Stereo spatialization based on deformation
- Hall reverb simulation
- Delay with feedback
- Dynamic filtering per material state

## Material Data

The project uses real stress-strain data including:
- Young's Modulus
- Yield Strength
- Ultimate Tensile Strength (UTS)
- Fracture point
- Serrations (discontinuous yielding)

## Credits

Created as an experimental exploration of material science through sound, combining engineering data with artistic interpretation.