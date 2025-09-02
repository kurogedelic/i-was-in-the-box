<CsoundSynthesizer>
<CsOptions>
-o stress-strain_curve.wav -W
</CsOptions>

<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

; Global reverb sends
ga_reverb_l init 0
ga_reverb_r init 0
ga_delay_l init 0
ga_delay_r init 0

; ======================
; Instrument 1: Main Metal Stress Sonification
; ======================
instr 1
    ; p4 = start stress (MPa)
    ; p5 = end stress (MPa)
    ; p6 = start strain
    ; p7 = end strain
    ; p8 = region (0=elastic, 1=yield, 2=plastic, 3=necking, 4=fracture)
    
    istart_stress = p4
    iend_stress = p5
    istart_strain = p6
    iend_strain = p7
    iregion = p8
    idur = p3
    
    ; Interpolate stress and strain over duration
    kstress line istart_stress, idur, iend_stress
    kstrain line istart_strain, idur, iend_strain
    
    ; Base frequency increases with stress
    kbase_freq = 60 + kstress * 0.3
    
    ; === AMPLITUDE ENVELOPE BASED ON REGION ===
    if (iregion == 0) then
        ; Elastic - gentle amplitude
        kamp linseg 0, 0.01, 0.15, idur-0.02, 0.15, 0.01, 0
    elseif (iregion == 1) then
        ; Yield - sudden increase
        kamp linseg 0, 0.001, 0.35, idur*0.3, 0.4, idur*0.7-0.01, 0.25, 0.01, 0
    elseif (iregion == 2) then
        ; Plastic - sustained with fluctuations
        kamp linseg 0, 0.01, 0.3, idur-0.02, 0.3, 0.01, 0
        kfluc oscili 0.05, 3 + kstrain * 10, 1
        kamp = kamp + kfluc
    elseif (iregion == 3) then
        ; Necking - decreasing but unstable
        kamp linseg 0, 0.01, 0.25, idur*0.5, 0.2, idur*0.5-0.01, 0.15, 0.01, 0
        kchaos randh 0.08, 20
        kamp = kamp + kchaos
    else
        ; Fracture - explosive
        kamp expon 0.8, idur, 0.001
    endif
    
    ; === SYNTHESIS LAYERS ===
    
    ; Layer 1: Fundamental metallic tone
    kfund_mod oscili 0.02, 0.7 + kstrain * 2, 1  ; Slight vibrato
    afund oscili kamp * 0.4, kbase_freq * (1 + kfund_mod), 1
    
    ; Layer 2: Harmonic resonances (metallic timbre)
    ares1 oscili kamp * 0.2, kbase_freq * 2.76, 1
    ares2 oscili kamp * 0.1, kbase_freq * 5.40, 1
    ares3 oscili kamp * 0.05, kbase_freq * 8.93, 1
    
    ; Layer 3: Inharmonic partials (bell-like metallic sound)
    ainhar1 oscili kamp * 0.15, kbase_freq * 1.73, 1
    ainhar2 oscili kamp * 0.08, kbase_freq * 3.89, 1
    ainhar3 oscili kamp * 0.04, kbase_freq * 7.23, 1
    
    ; Layer 4: Noise component (texture)
    if (iregion <= 1) then
        ; Minimal noise in elastic/yield
        anoise rand kamp * 0.02
        anoise butterlp anoise, 2000
    elseif (iregion == 2) then
        ; Moderate noise in plastic (creaking)
        anoise rand kamp * 0.08
        anoise butterbp anoise, kbase_freq * 4, kbase_freq
    else
        ; Heavy noise in necking/fracture
        anoise rand kamp * 0.15
        anoise butterhp anoise, 1000
    endif
    
    ; Layer 5: Stress-dependent distortion
    ametal_sum = afund + ares1 + ares2 + ares3 + ainhar1 + ainhar2 + ainhar3 + anoise
    
    if (kstress > 300) then
        ; Add distortion for high stress
        kdist = (kstress - 300) / 100
        ametal_sum tanh ametal_sum * (1 + kdist)
    endif
    
    ; === FILTERING BASED ON MATERIAL STATE ===
    if (iregion == 0) then
        ; Clean sound for elastic
        aout = ametal_sum
        aout butterlp aout, 8000
    elseif (iregion == 1) then
        ; Brighter at yield point
        aout = ametal_sum
        aout butterhp aout, 200
    elseif (iregion == 2) then
        ; Band-passed for plastic
        aout = ametal_sum
        aout butterbp aout, kbase_freq * 2, kbase_freq * 4
    elseif (iregion == 3) then
        ; Resonant filter for necking
        aout = ametal_sum
        aout reson aout, kbase_freq * 3, kbase_freq * 0.1, 1
    else
        ; Wide spectrum for fracture
        aout = ametal_sum
    endif
    
    ; === SPATIALIZATION ===
    ; Stress affects panning (material deformation in space)
    kpan = 0.5 + sin(kstress * 0.01) * 0.3 * (1 + kstrain)
    aleft = aout * sqrt(1 - kpan)
    aright = aout * sqrt(kpan)
    
    outs aleft, aright
    
    ; Send to effects
    ga_reverb_l = ga_reverb_l + aleft * 0.3
    ga_reverb_r = ga_reverb_r + aright * 0.3
    ga_delay_l = ga_delay_l + aleft * 0.1
    ga_delay_r = ga_delay_r + aright * 0.1
endin

; ======================
; Instrument 2: Creaking and Cracking Texture
; ======================
instr 2
    ; p4 = intensity (0-1)
    ; p5 = frequency center
    ; p6 = crack density
    
    iintensity = p4
    ifreq = p5
    idensity = p6
    
    ; Generate crack events
    kcrack_trig metro idensity
    
    if (kcrack_trig == 1) then
        ; Random crack parameters
        kcrack_amp random 0.1, 0.5
        kcrack_freq random ifreq * 0.8, ifreq * 1.2
    endif
    
    ; Crack sound synthesis
    acrack_env expseg 0.001, 0.001, 1, 0.02, 0.1, 0.1, 0.001
    acrack oscili kcrack_amp * iintensity, kcrack_freq, 1
    acrack = acrack * acrack_env
    
    ; Add metallic ring
    aring oscili kcrack_amp * 0.3, kcrack_freq * 2.1, 1
    aring = aring * acrack_env
    
    ; Combine and filter
    aout = acrack + aring
    aout butterhp aout, ifreq * 0.5
    
    ; Random stereo positioning for each crack
    kpan random 0.2, 0.8
    aleft = aout * (1 - kpan)
    aright = aout * kpan
    
    outs aleft, aright
endin

; ======================
; Instrument 3: Low Frequency Rumble (Material Stress)
; ======================
instr 3
    ; p4 = amplitude
    ; p5 = base frequency
    ; p6 = modulation depth
    
    iamp = p4
    ibasefreq = p5
    imoddepth = p6
    
    ; Low frequency oscillator with modulation
    kmod oscili imoddepth, 0.5, 1
    arumble oscili iamp, ibasefreq * (1 + kmod), 1
    
    ; Add sub-harmonics
    asub1 oscili iamp * 0.5, ibasefreq * 0.5, 1
    asub2 oscili iamp * 0.25, ibasefreq * 0.25, 1
    
    ; Mix and filter
    aout = arumble + asub1 + asub2
    aout butterlp aout, 200
    
    ; Slight stereo spread
    adelay_l delay aout, 0.01
    adelay_r delay aout, 0.015
    
    outs aout + adelay_l * 0.3, aout + adelay_r * 0.3
    
    ; Send to reverb
    ga_reverb_l = ga_reverb_l + aout * 0.4
    ga_reverb_r = ga_reverb_r + aout * 0.4
endin

; ======================
; Instrument 4: Impact/Dislocation Events
; ======================
instr 4
    ; p4 = impact strength
    ; p5 = frequency
    ; p6 = metallic content (0-1)
    
    istrength = p4
    ifreq = p5
    imetal = p6
    
    ; Impact envelope
    aenv expseg 1, 0.003, 0.5, 0.01, 0.1, 0.05, 0.001
    
    ; Impact sound (mix of noise and tone)
    anoise rand istrength
    aoscil oscili istrength * imetal, ifreq, 1
    
    ; Metallic resonance
    ares1 reson anoise, ifreq, ifreq * 0.01, 1
    ares2 reson anoise, ifreq * 2.3, ifreq * 0.02, 1
    ares3 reson anoise, ifreq * 4.7, ifreq * 0.03, 1
    
    ; Mix components
    aimpact = (anoise * (1 - imetal) + aoscil * imetal) * aenv
    aresonance = (ares1 + ares2 * 0.5 + ares3 * 0.25) * aenv * 0.3
    
    aout = aimpact + aresonance
    
    ; Filter based on frequency
    aout butterhp aout, ifreq * 0.2
    aout butterlp aout, ifreq * 10
    
    outs aout * 0.7, aout * 0.7
endin

; ======================
; Instrument 98: Delay Effect
; ======================
instr 98
    ; Simple delay with feedback
    al = ga_delay_l
    ar = ga_delay_r
    
    ; Delay lines
    adel_l delay al, 0.15
    adel_r delay ar, 0.17
    
    ; Feedback
    ga_delay_l = adel_l * 0.4
    ga_delay_r = adel_r * 0.4
    
    ; Output
    outs adel_l * 0.5, adel_r * 0.5
    
    ; Clear for next pass
    ga_delay_l = 0
    ga_delay_r = 0
endin

; ======================
; Instrument 99: Global Reverb
; ======================
instr 99
    ; Hall reverb simulation
    al = ga_reverb_l
    ar = ga_reverb_r
    
    ; Early reflections
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
    
    ; Comb filters for reverb tail
    acomb1_l comb al + aearly_l, 2, 0.05
    acomb2_l comb al + aearly_l, 2, 0.067
    acomb3_l comb al + aearly_l, 2, 0.083
    acomb4_l comb al + aearly_l, 2, 0.099
    
    acomb1_r comb ar + aearly_r, 2, 0.053
    acomb2_r comb ar + aearly_r, 2, 0.069
    acomb3_r comb ar + aearly_r, 2, 0.087
    acomb4_r comb ar + aearly_r, 2, 0.103
    
    ; Mix and filter
    arev_l = (acomb1_l + acomb2_l + acomb3_l + acomb4_l) * 0.25
    arev_r = (acomb1_r + acomb2_r + acomb3_r + acomb4_r) * 0.25
    
    arev_l butterlp arev_l, 8000
    arev_r butterlp arev_r, 8000
    
    ; Output with dry/wet control
    outs arev_l * 0.7, arev_r * 0.7
    
    ; Clear global sends
    ga_reverb_l = 0
    ga_reverb_r = 0
endin

</CsInstruments>

<CsScore>
; Function tables
f1 0 8192 10 1 0.5 0.3 0.25 0.2 0.167 0.14 0.125 0.111 0.1 ; Rich harmonic series
f2 0 8192 10 1 0 0.5 0 0.33 0 0.25 0 0.2 ; Odd harmonics only
f3 0 8192 7 0 2048 1 4096 1 2048 0 ; Triangle envelope
f4 0 8192 5 1 4096 0.01 4096 1 ; Exponential curve

; Generated score from interpolated CSV data
; Duration: 30.0 seconds
; Data points: 1000

t 0 60  ; Tempo 60 BPM

; Effects run throughout
i98 0 35.0  ; Delay
i99 0 37.0  ; Reverb

; Main sonification
i3 0 20.0 0.05 30 0.1  ; Base rumble
i1 0.000 5.140 0.0 199.5 0.000000 0.000950 0
i4 5.140 0.1 0.5 200 0.8  ; Yield point impact
i3 6.000 0.5 0.043 43.4 0.051  ; Stress rumble
i1 5.140 1.280 200.3 249.5 0.000954 0.001183 1
i4 6.420 0.05 0.3 500 0.6  ; Plastic transition
i2 7.000 0.5 0.30 2269 5.0  ; Plastic creaking
i1 6.420 0.880 250.1 250.3 0.001193 0.001998 2
i4 7.300 0.1 0.5 250 0.8  ; Yield point impact
i3 8.000 0.5 0.045 44.6 0.053  ; Stress rumble
i1 7.300 1.120 250.0 250.0 0.002017 0.003045 1
i4 8.420 0.05 0.3 500 0.6  ; Plastic transition
i1 8.420 0.500 250.2 250.0 0.003064 0.003513 2
i4 8.920 0.1 0.5 250 0.8  ; Yield point impact
i3 10.000 0.5 0.045 44.5 0.055  ; Stress rumble
i3 12.000 0.5 0.044 44.2 0.056  ; Stress rumble
i1 8.920 4.020 249.8 249.7 0.003532 0.008003 1
i4 12.940 0.05 0.3 504 0.6  ; Plastic transition
i2 13.000 0.5 0.32 2291 5.2  ; Plastic creaking
i2 14.000 0.5 0.34 2711 5.4  ; Plastic creaking
i3 14.000 0.5 0.054 54.2 0.068  ; Stress rumble
i1 12.940 1.800 252.0 379.9 0.008197 0.024626 2
i2 14.740 1.0 0.5 3807 10  ; Necking cracks
i4 14.800 0.02 0.23 583 0.7  ; Necking pops
i4 15.200 0.02 0.23 597 0.7  ; Necking pops
i1 14.740 0.600 380.7 400.0 0.024811 0.030169 3
i4 15.340 0.5 0.4 100 0.9  ; Fracture bang (reduced)
i2 15.340 2.0 0.4 8000 30  ; Fracture cracking (reduced)
i3 16.000 0.5 0.061 61.3 0.086  ; Stress rumble
i3 18.000 0.5 0.062 62.0 0.109  ; Stress rumble
i1 15.340 3.680 400.5 400.1 0.030354 0.078710 4
i2 19.020 1.0 0.5 3992 10  ; Necking cracks
i4 19.200 0.02 0.28 590 0.7  ; Necking pops
i1 19.020 0.360 399.2 380.0 0.079107 0.085843 3
i4 19.380 0.05 0.3 757 0.6  ; Plastic transition
i1 19.380 0.620 378.6 320.0 0.086239 0.098127 2
i4 19.500 1.0 0.5 50 0.9  ; Final fracture (reduced)
i2 19.700 5.0 0.4 12000 50  ; Fracture aftermath (reduced)
i1 20.000 8.0 320.0 50.0 0.098127 0.100000 4  ; Fracture decay
i3 20.000 10.0 0.15 15 0.3  ; Deep rumble decay (reduced)
i2 22.000 5.0 0.3 5000 10  ; Metal ringing (reduced)
i4 24.000 0.5 0.2 80 0.7  ; Final resonance (reduced)

e
</CsScore>
</CsoundSynthesizer>