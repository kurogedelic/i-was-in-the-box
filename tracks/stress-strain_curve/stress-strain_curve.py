#!/usr/bin/env python3
"""
Stress-Strain Curve Sonification
Converts CSV stress-strain data to Csound for audio synthesis
"""

import csv
import numpy as np
from scipy import interpolate

def create_orchestra_section():
    """Create the Csound orchestra section"""
    return """<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

; Global reverb sends
ga_reverb_l init 0
ga_reverb_r init 0
ga_delay_l init 0
ga_delay_r init 0

; Instrument 1: Main Metal Stress Sonification
instr 1
    istart_stress = p4
    iend_stress = p5
    istart_strain = p6
    iend_strain = p7
    iregion = p8
    idur = p3
    
    kstress line istart_stress, idur, iend_stress
    kstrain line istart_strain, idur, iend_strain
    kbase_freq = 60 + kstress * 0.3
    
    if (iregion == 0) then
        kamp linseg 0, 0.01, 0.15, idur-0.02, 0.15, 0.01, 0
    elseif (iregion == 1) then
        kamp linseg 0, 0.001, 0.35, idur*0.3, 0.4, idur*0.7-0.01, 0.25, 0.01, 0
    elseif (iregion == 2) then
        kamp linseg 0, 0.01, 0.3, idur-0.02, 0.3, 0.01, 0
        kfluc oscili 0.05, 3 + kstrain * 10, 1
        kamp = kamp + kfluc
    elseif (iregion == 3) then
        kamp linseg 0, 0.01, 0.25, idur*0.5, 0.2, idur*0.5-0.01, 0.15, 0.01, 0
        kchaos randh 0.08, 20
        kamp = kamp + kchaos
    else
        kamp expon 0.35, idur, 0.001
    endif
    
    kfund_mod oscili 0.02, 0.7 + kstrain * 2, 1
    afund oscili kamp * 0.4, kbase_freq * (1 + kfund_mod), 1
    ares1 oscili kamp * 0.2, kbase_freq * 2.76, 1
    ares2 oscili kamp * 0.1, kbase_freq * 5.40, 1
    ares3 oscili kamp * 0.05, kbase_freq * 8.93, 1
    ainhar1 oscili kamp * 0.15, kbase_freq * 1.73, 1
    ainhar2 oscili kamp * 0.08, kbase_freq * 3.89, 1
    ainhar3 oscili kamp * 0.04, kbase_freq * 7.23, 1
    
    if (iregion <= 1) then
        anoise rand kamp * 0.02
        anoise butterlp anoise, 2000
    elseif (iregion == 2) then
        anoise rand kamp * 0.08
        anoise butterbp anoise, kbase_freq * 4, kbase_freq
    else
        anoise rand kamp * 0.15
        anoise butterhp anoise, 1000
    endif
    
    ametal_sum = afund + ares1 + ares2 + ares3 + ainhar1 + ainhar2 + ainhar3 + anoise
    
    if (kstress > 300) then
        kdist = (kstress - 300) / 100
        ametal_sum tanh ametal_sum * (1 + kdist)
    endif
    
    if (iregion == 0) then
        aout = ametal_sum
        aout butterlp aout, 8000
    elseif (iregion == 1) then
        aout = ametal_sum
        aout butterhp aout, 200
    elseif (iregion == 2) then
        aout = ametal_sum
        aout butterbp aout, kbase_freq * 2, kbase_freq * 4
    elseif (iregion == 3) then
        aout = ametal_sum
        aout reson aout, kbase_freq * 3, kbase_freq * 0.1, 1
    else
        aout = ametal_sum
    endif
    
    kpan = 0.5 + sin(kstress * 0.01) * 0.3 * (1 + kstrain)
    aleft = aout * sqrt(1 - kpan)
    aright = aout * sqrt(kpan)
    
    outs aleft, aright
    ga_reverb_l = ga_reverb_l + aleft * 0.3
    ga_reverb_r = ga_reverb_r + aright * 0.3
    ga_delay_l = ga_delay_l + aleft * 0.1
    ga_delay_r = ga_delay_r + aright * 0.1
endin

; Instrument 2: Creaking and Cracking Texture
instr 2
    iintensity = p4
    ifreq = p5
    idensity = p6
    kcrack_trig metro idensity
    
    if (kcrack_trig == 1) then
        kcrack_amp random 0.1, 0.5
        kcrack_freq random ifreq * 0.8, ifreq * 1.2
    endif
    
    acrack_env expseg 0.001, 0.001, 1, 0.02, 0.1, 0.1, 0.001
    acrack oscili kcrack_amp * iintensity, kcrack_freq, 1
    acrack = acrack * acrack_env
    aring oscili kcrack_amp * 0.3, kcrack_freq * 2.1, 1
    aring = aring * acrack_env
    aout = acrack + aring
    aout butterhp aout, ifreq * 0.5
    kpan random 0.2, 0.8
    aleft = aout * (1 - kpan)
    aright = aout * kpan
    outs aleft, aright
endin

; Instrument 3: Low Frequency Rumble
instr 3
    iamp = p4
    ibasefreq = p5
    imoddepth = p6
    kmod oscili imoddepth, 0.5, 1
    arumble oscili iamp, ibasefreq * (1 + kmod), 1
    asub1 oscili iamp * 0.5, ibasefreq * 0.5, 1
    asub2 oscili iamp * 0.25, ibasefreq * 0.25, 1
    aout = arumble + asub1 + asub2
    aout butterlp aout, 200
    adelay_l delay aout, 0.01
    adelay_r delay aout, 0.015
    outs aout + adelay_l * 0.3, aout + adelay_r * 0.3
    ga_reverb_l = ga_reverb_l + aout * 0.4
    ga_reverb_r = ga_reverb_r + aout * 0.4
endin

; Instrument 4: Impact/Dislocation Events
instr 4
    istrength = p4
    ifreq = p5
    imetal = p6
    aenv expseg 1, 0.003, 0.5, 0.01, 0.1, 0.05, 0.001
    anoise rand istrength
    aosc oscili istrength * imetal, ifreq, 1
    ares1 reson anoise, ifreq, ifreq * 0.01, 1
    ares2 reson anoise, ifreq * 2.3, ifreq * 0.02, 1
    ares3 reson anoise, ifreq * 4.7, ifreq * 0.03, 1
    aimpact = (anoise * (1 - imetal) + aosc * imetal) * aenv
    aresonance = (ares1 + ares2 * 0.5 + ares3 * 0.25) * aenv * 0.3
    aout = aimpact + aresonance
    aout butterhp aout, ifreq * 0.2
    aout butterlp aout, ifreq * 10
    outs aout * 0.7, aout * 0.7
endin

; Instrument 98: Delay Effect
instr 98
    al = ga_delay_l
    ar = ga_delay_r
    adel_l delay al, 0.15
    adel_r delay ar, 0.17
    ga_delay_l = adel_l * 0.4
    ga_delay_r = adel_r * 0.4
    outs adel_l * 0.5, adel_r * 0.5
    ga_delay_l = 0
    ga_delay_r = 0
endin

; Instrument 99: Global Reverb
instr 99
    al = ga_reverb_l
    ar = ga_reverb_r
    aearly1_l delay al, 0.023
    aearly2_l delay al, 0.041
    aearly3_l delay al, 0.067
    aearly4_l delay al, 0.093
    aearly1_r delay ar, 0.027
    aearly2_r delay ar, 0.043
    aearly3_r delay ar, 0.071
    aearly4_r delay ar, 0.097
    aearly_l = (aearly1_l + aearly2_l * 0.8 + aearly3_l * 0.6 + aearly4_l * 0.4) * 0.3
    aearly_r = (aearly1_r + aearly2_r * 0.8 + aearly3_r * 0.6 + aearly4_r * 0.4) * 0.3
    acomb1_l comb al + aearly_l, 2, 0.05
    acomb2_l comb al + aearly_l, 2, 0.067
    acomb3_l comb al + aearly_l, 2, 0.083
    acomb4_l comb al + aearly_l, 2, 0.099
    acomb1_r comb ar + aearly_r, 2, 0.053
    acomb2_r comb ar + aearly_r, 2, 0.069
    acomb3_r comb ar + aearly_r, 2, 0.087
    acomb4_r comb ar + aearly_r, 2, 0.103
    arev_l = (acomb1_l + acomb2_l + acomb3_l + acomb4_l) * 0.25
    arev_r = (acomb1_r + acomb2_r + acomb3_r + acomb4_r) * 0.25
    arev_l butterlp arev_l, 8000
    arev_r butterlp arev_r, 8000
    outs arev_l * 0.7, arev_r * 0.7
    ga_reverb_l = 0
    ga_reverb_r = 0
endin
</CsInstruments>"""

def load_and_interpolate_csv(csv_file, target_points=1000):
    """Load CSV and interpolate to create smooth transitions"""
    strains = []
    stresses = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row['strain']))
            stresses.append(float(row['stress_MPa']))
    
    # Create interpolation function
    f_interp = interpolate.interp1d(range(len(stresses)), stresses, 
                                   kind='cubic', fill_value='extrapolate')
    strain_interp = interpolate.interp1d(range(len(strains)), strains, 
                                        kind='cubic', fill_value='extrapolate')
    
    # Generate more points
    x_new = np.linspace(0, len(stresses)-1, target_points)
    stresses_smooth = f_interp(x_new)
    strains_smooth = strain_interp(x_new)
    
    return strains_smooth, stresses_smooth

def detect_regions(stresses):
    """Detect material regions based on stress levels"""
    regions = []
    for stress in stresses:
        if stress < 200:
            regions.append(0)  # Elastic
        elif stress < 250:
            regions.append(1)  # Yield
        elif stress < 380:
            regions.append(2)  # Plastic
        elif stress < 400:
            regions.append(3)  # Necking
        else:
            regions.append(4)  # Fracture
    return regions

def detect_serrations(stresses, threshold=5):
    """Detect stress drops (serrations) in the data"""
    serrations = []
    for i in range(1, len(stresses)):
        stress_drop = stresses[i-1] - stresses[i]
        if stress_drop > threshold:
            serrations.append((i, stress_drop))
    return serrations

def generate_csound_score(strains, stresses, duration=30.0):
    """Generate Csound score with realistic timing"""
    score_lines = []
    num_points = len(stresses)
    # Only use first 20 seconds for actual data
    data_duration = 20.0  
    time_per_point = data_duration / num_points
    
    regions = detect_regions(stresses)
    serrations = detect_serrations(stresses)
    serration_times = [s[0] * time_per_point for s in serrations]
    
    # Track region changes
    prev_region = -1
    region_start_time = 0
    region_start_idx = 0
    
    # Low frequency rumble throughout data duration
    score_lines.append(f"i3 0 {data_duration} 0.05 30 0.1  ; Base rumble")
    
    current_time = 0
    
    for i in range(num_points):
        current_region = regions[i]
        
        # Check for region change
        if current_region != prev_region:
            if prev_region != -1:
                # End previous region with interpolated values
                segment_duration = current_time - region_start_time
                if segment_duration > 0:
                    score_lines.append(
                        f"i1 {region_start_time:.3f} {segment_duration:.3f} "
                        f"{stresses[region_start_idx]:.1f} {stresses[i-1]:.1f} "
                        f"{strains[region_start_idx]:.6f} {strains[i-1]:.6f} {prev_region}"
                    )
                
                # Add transition effects
                if current_region == 1:  # Entering yield
                    score_lines.append(f"i4 {current_time:.3f} 0.1 0.5 {stresses[i]:.0f} 0.8  ; Yield point impact")
                elif current_region == 2:  # Entering plastic
                    score_lines.append(f"i4 {current_time:.3f} 0.05 0.3 {stresses[i]*2:.0f} 0.6  ; Plastic transition")
                elif current_region == 3:  # Entering necking
                    score_lines.append(f"i2 {current_time:.3f} 1.0 0.5 {stresses[i]*10:.0f} 10  ; Necking cracks")
                elif current_region == 4:  # Fracture
                    score_lines.append(f"i4 {current_time:.3f} 0.5 0.4 100 0.9  ; Fracture bang (reduced)")
                    score_lines.append(f"i2 {current_time:.3f} 2.0 0.4 8000 30  ; Fracture cracking (reduced)")
            
            region_start_time = current_time
            region_start_idx = i
            prev_region = current_region
        
        # Add periodic texture based on region
        if current_region == 2 and i % 50 == 0:  # Plastic creaking
            score_lines.append(
                f"i2 {current_time:.3f} 0.5 {0.3 + strains[i]*2:.2f} "
                f"{1000 + stresses[i]*5:.0f} {5 + strains[i]*20:.1f}  ; Plastic creaking"
            )
        elif current_region == 3 and i % 20 == 0:  # Necking instability
            score_lines.append(
                f"i4 {current_time:.3f} 0.02 {0.2 + strains[i]:.2f} "
                f"{200 + stresses[i]:.0f} 0.7  ; Necking pops"
            )
        
        # Check for serrations
        if current_time in serration_times:
            idx = serration_times.index(current_time)
            drop_magnitude = serrations[idx][1]
            score_lines.append(
                f"i4 {current_time:.3f} 0.05 {drop_magnitude/20:.2f} "
                f"{100 + stresses[i]:.0f} 0.5  ; Serration event"
            )
        
        # Add subtle continuous texture
        if i % 100 == 0 and current_region > 0:
            score_lines.append(
                f"i3 {current_time:.3f} 0.5 {0.02 + stresses[i]/10000:.3f} "
                f"{20 + stresses[i]/10:.1f} {0.05 + strains[i]:.3f}  ; Stress rumble"
            )
        
        current_time += time_per_point
    
    # Final region segment (up to data_duration)
    if prev_region != -1:
        segment_duration = data_duration - region_start_time
        if segment_duration > 0:
            score_lines.append(
                f"i1 {region_start_time:.3f} {segment_duration:.3f} "
                f"{stresses[region_start_idx]:.1f} {stresses[-1]:.1f} "
                f"{strains[region_start_idx]:.6f} {strains[-1]:.6f} {prev_region}"
            )
    
    # Add final fracture event if reached
    if regions[-1] >= 2:  # If we reached at least plastic deformation
        score_lines.append(f"i4 {data_duration-0.5:.3f} 1.0 0.5 50 0.9  ; Final fracture (reduced)")
        score_lines.append(f"i2 {data_duration-0.3:.3f} 5.0 0.4 12000 50  ; Fracture aftermath (reduced)")
        # Add resonating decay after fracture (main data ends at 20 seconds)
        score_lines.append(f"i1 {data_duration:.3f} 8.0 {stresses[-1]:.1f} 50.0 {strains[-1]:.6f} 0.100000 4  ; Fracture decay")
        score_lines.append(f"i3 {data_duration:.3f} 10.0 0.15 15 0.3  ; Deep rumble decay (reduced)")
        score_lines.append(f"i2 {data_duration+2:.3f} 5.0 0.3 5000 10  ; Metal ringing (reduced)")
        score_lines.append(f"i4 {data_duration+4:.3f} 0.5 0.2 80 0.7  ; Final resonance (reduced)")
    
    return score_lines

def create_enhanced_csd(csv_file, output_file='stress-strain_curve.csd', duration=30.0):
    """Create enhanced CSD file with interpolated data"""
    
    # Load and interpolate data
    try:
        strains, stresses = load_and_interpolate_csv(csv_file, target_points=1000)
    except:
        print("Using numpy-free interpolation...")
        # Fallback without scipy
        strains = []
        stresses = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                strains.append(float(row['strain']))
                stresses.append(float(row['stress_MPa']))
        
        # Simple linear interpolation
        new_strains = []
        new_stresses = []
        for i in range(len(strains)-1):
            new_strains.append(strains[i])
            new_stresses.append(stresses[i])
            # Add interpolated points
            for j in range(1, 10):
                alpha = j / 10.0
                new_strains.append(strains[i] * (1-alpha) + strains[i+1] * alpha)
                new_stresses.append(stresses[i] * (1-alpha) + stresses[i+1] * alpha)
        new_strains.append(strains[-1])
        new_stresses.append(stresses[-1])
        strains = new_strains
        stresses = new_stresses
    
    # Generate score
    score_lines = generate_csound_score(strains, stresses, duration)
    
    # Read orchestra from existing file if available, otherwise use embedded
    try:
        with open('stress-strain_curve.csd', 'r') as f:
            content = f.read()
            start = content.find('<CsInstruments>')
            end = content.find('</CsInstruments>') + len('</CsInstruments>')
            orchestra = content[start:end]
    except FileNotFoundError:
        # Use embedded orchestra definition
        orchestra = create_orchestra_section()
    
    # Create complete CSD
    csd_content = f"""<CsoundSynthesizer>
<CsOptions>
-o {output_file.replace('.csd', '.wav')} -W
</CsOptions>

{orchestra}

<CsScore>
; Function tables
f1 0 8192 10 1 0.5 0.3 0.25 0.2 0.167 0.14 0.125 0.111 0.1 ; Rich harmonic series
f2 0 8192 10 1 0 0.5 0 0.33 0 0.25 0 0.2 ; Odd harmonics only
f3 0 8192 7 0 2048 1 4096 1 2048 0 ; Triangle envelope
f4 0 8192 5 1 4096 0.01 4096 1 ; Exponential curve

; Generated score from interpolated CSV data
; Duration: {duration} seconds
; Data points: {len(stresses)}

t 0 60  ; Tempo 60 BPM

; Effects run throughout
i98 0 {duration + 5}  ; Delay
i99 0 {duration + 7}  ; Reverb

; Main sonification
"""
    
    # Add score lines
    for line in score_lines:
        csd_content += line + '\n'
    
    csd_content += """
e
</CsScore>
</CsoundSynthesizer>"""
    
    # Write output file
    with open(output_file, 'w') as f:
        f.write(csd_content)
    
    print(f"Created {output_file}")
    print(f"Duration: {duration} seconds")
    print(f"Data points: {len(stresses)}")
    
    return output_file

if __name__ == "__main__":
    csv_file = "stress_strain_serrations.csv"
    output_file = create_enhanced_csd(csv_file, duration=30.0)
    print(f"Successfully created {output_file}")
    print("Run: csound " + output_file)