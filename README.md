# 箱の中に居たのは私だった。
*I Was the One in the Box*

**Leo Kuroshita**  

An experimental music album produced without a DAW. 

## Tracklist

1. 偏頭痛／刹那的な感覚の再現 - *Experiments in Momentary Recognition of Sensation and Perception*  
   — Dynamic granular synthesis of 5 sound sources. Particle positions driven by chaotic functions, 20-segment progression, FFT spectrum freezing  
2. ラッピングされた君の会話 - *Your Wrapped Conversation*  
   — Csound Euclidean rhythm generation. Zap synthesis (12kHz→20Hz), tempo acceleration from 280 to 400 BPM, sudden stop at 164 seconds  
3. 不確定の庭 - *Indeterminacy Field*  
   — 16×16 cellular automaton music generation. Conway variant rules, pentatonic mapping, C minor/Dorian harmony progression  
4. Drosera's Song - *Drosera's Song*  
   — Rosenberg glottal pulse + cascade formant synthesis. Scientifically accurate carnivorous plant educational nursery rhyme  
5. 応力–ひずみ曲線 - *Stress–Strain Curve*  
   — Sonification of material test CSV data. Resonant frequency changes from elastic to yield to plastic to fracture, strain above 300 MPa  
6. Choral Induction Protocol - *Choral Induction Protocol*  
   — 8-voice polyphonic formant synthesis. Slendro tuning, gamelan kotekan structure, Haas effect  
7. Constellation - Phase 1 Group 2 - *Constellation - Phase 1 Group 2*  
   — Sonification of synthetic satellite data + NASA public data. Altitude mapped to frequency, orbital inclination mapped to modulation  
8. アシッド・テクノの印象 - *Impression of Acid Techno*  
   — Web Audio API TB-303 modeling. Resonant filter sweep, real-time pattern generation.
   -> Webapp [acid-test500.vercel.app](acid-test500.vercel.app)  
9. 夜の輝く湖水 (Avonlea) - *The Lake of Shining Waters (Avonlea)*  
   — norns environment-responsive ambient. Moon phase and light condition reactions, inspired by *Anne of Green Gables* by LM Montgomery
   -> norns script [Avonlea](github.com/kurogedelic/avonlea)  
10. 以下、SCSIディスクが回答します。 - *SCSI Disk is Answering*  
    — 10000 RPM physical model. Fundamental frequency at 166.7 Hz, bearing resonances 2-8 kHz, seek operation 500-3000 Hz chirp  

## Project Structure

```
InTheBox/
├── tracks/           # Source code for each track production
│   ├── EMRSP/        # Track 1: 偏頭痛／刹那的な感覚の再現
│   ├── Your_Wrapped_Conversation/ # Track 2: ラッピングされた君の会話
│   ├── indeterminacy_field/        # Track 3: 不確定の庭
│   ├── drosera/      # Track 4: Drosera's Song
│   ├── stress-strain_curve/ # Track 5: 応力–ひずみ曲線
│   ├── choral_induction_protocol/       # Track 6: Choral Induction Protocol
│   ├── constellation/ # Track 7: Constellation - Phase 1 Group 2
│   └── scsi_disk/    # Track 10: SCSIディスク
└── mastering/        # Mastering scripts
```

## File Formats

### Release Version (`release/`)  
- **WAV**: 24bit @ 48kHz (standard format for distribution/delivery)  
- **Metadata**: Embedding song titles, artist info, etc. (via ffmpeg)  
- **Loudness Targets**:  
  - Apple Music: −16 LUFS / True Peak ≤ −1 dBTP (for Sound Check)  
  - Bandcamp: −14 LUFS / True Peak ≤ −1 dBTP (no normalization assumed)  

### Master Version (`masters/wav/`)  
- **WAV**: 24bit @ 48kHz  
- **Purpose**: Archive and base for remastering  

## Production & Reproduction

### Dependencies  
- **Python 3.8+**  
- Python packages: see `requirements.txt`  
- **FFmpeg** (pydub/masters 用)  
- **SuperCollider** (Stress–Strain Curve, Avonlea)  
- **FAUST** (SCSI Disk)  
- **norns** (Avonlea)  

### Setup  
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`  
- Install deps: `pip install -U pip && pip install -r requirements.txt`  
- FFmpeg: macOS `brew install ffmpeg` / Ubuntu `sudo apt-get install ffmpeg`

### How to Run (examples)  
- Track 7 (Constellation): `python tracks/constellation/main.py`  
- Track 10 (SCSI Disk): `python tracks/scsi_disk/scsi_disk_synth.py`  
- Track 1 (EMRSP granular): requires audio source; see script `--help`  

## License

- **Author**: Leo Kuroshita (2025)  
- **Code**: GPL-3.0 — see `LICENSE`  
- **Audio & Art**: CC BY-NC-SA 4.0 — see `LICENSE-CC-BY-NC-SA-4.0.txt`  
