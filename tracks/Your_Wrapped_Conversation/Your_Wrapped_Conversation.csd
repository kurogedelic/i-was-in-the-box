<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>
; "Your Wrapped Conversation by Leo Kuroshita 2025"

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1


gksection init 0
gkintensity init 0.5
gktempo init 280


opcode euclidean, k, kk
    ksteps, kpulses xin
    kcount init 0
    kout init 0
    
    if ksteps > 0 && kpulses > 0 then
        kthresh = kpulses / ksteps
        krandom random 0, 1
        if krandom < kthresh then
            kout = 1
        else
            kout = 0
        endif
    endif
    
    xout kout
endop


gaRevL init 0
gaRevR init 0
gaDelL init 0
gaDelR init 0

gaCompL init 0
gaCompR init 0


instr 1
    idur = p3
    iamp = p4
    ipan = p5
    ivariation = p6  
    
    
    if ivariation == 1 then
        kfreq expon 12000, idur*0.05, 20  
    elseif ivariation == 2 then
        kfreq linseg 2000, idur*0.3, 500, idur*0.7, 100  
    else
        kfreq expon 8000, idur*0.1, 50
    endif
    
    anoise noise iamp*0.8, 0
    azap butterbp anoise, kfreq, kfreq*0.3
    
    
    kmod oscil 500*gkintensity, 17, 1
    azap2 butterbp anoise, kfreq+kmod, 200
    
    
    amix = azap*0.6 + azap2*0.4
    amix tanh amix * 1.2
    
    
    kenv expon 1, idur, 0.001
    aout = amix * kenv * iamp * (0.5 + gkintensity*0.5)
    
    
    aL, aR pan2 aout, ipan
    
    
    gaCompL += aL
    gaCompR += aR
    
    
    gaRevL += aL * 0.6
    gaRevR += aR * 0.6
    gaDelL += aL * 0.08
    gaDelR += aR * 0.08
endin


instr 2
    idur = p3
    iamp = p4
    ipitch = p5
    idist = p6  
    
    
    kpenv linseg ipitch*4, 0.02, ipitch, idur-0.02, ipitch*0.5
    
    
    asub oscil iamp*0.5, kpenv*0.5, 2
    
    
    asig oscil iamp, kpenv, 2
    
    
    aclick noise iamp*0.3, 0
    aclick butterlp aclick, 2000
    aclick = aclick * expon:a(1, 0.005, 0.001)
    
    
    amix = asig + aclick + asub*gkintensity
    
    
    kenv expon 1, idur*0.8, 0.001
    aout = amix * kenv
    
    
    aout tanh aout * 1.3
    aout = aout * (0.7 / (1 + idist*0.5))
    
    
    kfiltfreq = 1000 + gkintensity*3000
    aout butterlp aout, kfiltfreq
    
    
    gaCompL += aout
    gaCompR += aout
    
    
    gaRevL += aout * 0.4
    gaRevR += aout * 0.4
endin


instr 3
    idur = p3
    iamp = p4
    itype = p5  
    
    
    if itype == 1 then
        
        kfreq randomh 2000, 12000, 100
        aring oscil 1, kfreq, 1
        anoise noise iamp*0.5, 0
        aglitch = anoise * aring
        aglitch comb aglitch, 0.5, 0.002  
    elseif itype == 2 then
        
        kfreq randomh 100, 4000, 200
        aring oscil 1, kfreq, 1
        anoise noise iamp, 0
        aglitch = anoise * aring
    else
        
        kfreq randomh 200, 8000, 50
        aring oscil 1, kfreq, 1
        anoise noise iamp, 0
        aglitch = anoise * aring
    endif
    
    
    kstart = 8000*(0.5+gkintensity*0.5)
    if kstart < 1 then
        kstart = 1000
    endif
    kfilt expon i(kstart), idur, 100
    aglitch butterbp aglitch, kfilt, kfilt*0.5
    
    
    amix = aglitch*0.8
    amix tanh amix * 1.1
    
    
    kenv linseg 0, 0.001, 1, idur*0.1, 0.3, idur*0.8, 0
    aout = aglitch * kenv * iamp * (0.3 + gkintensity*0.7)
    
    
    kpan randomh -1, 1, 3
    aL, aR pan2 aout, kpan*0.5 + 0.5
    
    
    gaCompL += aL
    gaCompR += aR
    
    gaRevL += aL * 0.5
    gaRevR += aR * 0.5
    gaDelL += aL * 0.04
    gaDelR += aR * 0.04
endin


instr 4
    idur = p3
    iamp = p4
    ifreq = p5
    itype = p6  
    
    if itype == 0 then
        
        a1 oscil iamp*0.3, ifreq, 1
        a2 oscil iamp*0.3, ifreq*1.01, 1
        a3 oscil iamp*0.3, ifreq*0.5, 2
        adrone = a1 + a2 + a3
        adrone butterlp adrone, 800
    elseif itype == 1 then
        
        a1 oscil iamp*0.2, ifreq*2, 1
        a2 oscil iamp*0.2, ifreq*2.01, 1
        a3 oscil iamp*0.2, ifreq*3, 1
        a4 oscil iamp*0.2, ifreq*4, 1
        adrone = a1 + a2 + a3 + a4
        kfiltmod oscil 2000, 0.3, 1
        adrone butterbp adrone, 2000+kfiltmod, 1000
    else
        
        anoise1 noise iamp*0.5, 0
        anoise2 noise iamp*0.5, 0.5
        adrone = anoise1 + anoise2
        kfiltfreq linseg 100, idur*0.5, 4000, idur*0.5, 100
        adrone butterbp adrone, kfiltfreq, kfiltfreq*0.3
    endif
    
    
    kmodamp oscil 0.3, 0.1*gkintensity, 1
    adrone = adrone * (0.7 + kmodamp)
    
    
    kenv linseg 0, idur*0.1, 1, idur*0.7, 0.8, idur*0.2, 0
    aout = adrone * kenv * gkintensity
    
    
    aL = aout * 0.6
    aR = aout * 0.6
    aL delay aL, 0.013
    aR delay aR, 0.017
    
    
    gaCompL += aL
    gaCompR += aR
    
    gaRevL += aL * 0.6
    gaRevR += aR * 0.6
endin


instr 5
    idur = p3
    iamp = p4
    idir = p5  
    
    
    if idir == 1 then
        kfreq expon 50, idur, 8000
    else
        kfreq expon 8000, idur, 50
    endif
    
    
    a1 oscil iamp*0.3, kfreq, 1
    a2 oscil iamp*0.3, kfreq*1.5, 1
    a3 oscil iamp*0.2, kfreq*2, 1
    anoise noise iamp*0.2, 0
    anoise butterbp anoise, kfreq, kfreq*0.5
    
    amix = a1 + a2 + a3 + anoise
    
    
    kfilt = kfreq * 2
    amix butterlp amix, kfilt
    
    
    kenv linseg 0, 0.01, 0.5, idur*0.9, 0.5, idur*0.09, 0
    aout = amix * kenv * gkintensity * 0.5
    
    
    kpan linseg -0.8*idir, idur, 0.8*idir
    aL, aR pan2 aout, kpan*0.5 + 0.5
    
    
    gaCompL += aL
    gaCompR += aR
    
    gaRevL += aL * 0.5
    gaRevR += aR * 0.5
endin


instr 97
    
    aL = gaCompL
    aR = gaCompR
    
    
    krmsL rms aL
    krmsR rms aR
    krms = (krmsL + krmsR) * 0.5
    
    
    kthresh = 0.5    
    kratio = 4       
    kattack = 0.002  
    krelease = 0.05  
    
    
    kdb = 20 * log10(krms + 0.00001)
    kthreshdb = 20 * log10(kthresh)
    
    if kdb > kthreshdb then
        kreduction = (kdb - kthreshdb) * (1 - 1/kratio)
        kgain = 10^(-kreduction/20)
    else
        kgain = 1
    endif
    
    
    ksmooth portk kgain, kattack
    
    
    aL = aL * ksmooth * 1.5  
    aR = aR * ksmooth * 1.5
    
    
    aL tanh aL * 0.9
    aR tanh aR * 0.9
    
    
    outs aL, aR
    
    
    gaCompL = 0
    gaCompR = 0
endin


instr 6
    idur = p3
    iamp = p4
    itype = p5  
    
    
    anoise noise iamp, 0
    
    if itype == 0 then
        
        kenv expon 1, idur*0.2, 0.001
        kfilt = 8000
    else
        
        kenv expon 1, idur*0.8, 0.001  
        kfilt = 12000
    endif
    
    
    ahihat butterhp anoise, 4000
    ahihat butterlp ahihat, kfilt
    
    
    aring oscil 1, 5500, 1
    ahihat = ahihat * aring * 0.7 + ahihat * 0.3
    
    aout = ahihat * kenv * 0.4  
    
    
    aL = aout * 0.7
    aR = aout * 0.7
    aL delay aL, 0.001
    aR delay aR, 0.002
    
    
    gaCompL += aL
    gaCompR += aR
    
    gaRevL += aL * 0.1
    gaRevR += aR * 0.1
endin


instr 7
    idur = p3
    iamp = p4
    ifreq = p5
    
    
    a1 oscil iamp*0.3, ifreq, 1
    a2 oscil iamp*0.3, ifreq*1.01, 1
    a3 oscil iamp*0.3, ifreq*0.99, 1
    a4 oscil iamp*0.2, ifreq*2, 1
    
    amix = a1 + a2 + a3 + a4
    
    
    kenv linseg 0, 0.002, 1, idur*0.1, 0.7, idur*0.8, 0
    
    
    kfilt expon 5000, idur, 500
    amix butterlp amix, kfilt
    
    aout = amix * kenv * 0.4  
    
    
    aL = aout * 0.6
    aR = aout * 0.6
    aL delay aL, 0.005
    aR delay aR, 0.007
    
    
    gaCompL += aL
    gaCompR += aR
    
    gaRevL += aL * 0.4
    gaRevR += aR * 0.4
endin


instr 98
    aL = gaDelL
    aR = gaDelR
    
    
    kfeedback = 0.4 + gkintensity*0.2
    adelL delay aL, 0.375
    adelR delay aR, 0.25
    
    adelL = adelL * kfeedback
    adelR = adelR * kfeedback
    
    gaDelL = adelL
    gaDelR = adelR
    
    
    gaCompL += adelL*0.3
    gaCompR += adelR*0.3
    
    gaDelL = 0
    gaDelR = 0
endin


instr 99
    
    ksize = 0.4 + gkintensity*0.1  
    khfDamp = 8000 + gkintensity*2000  
    
    
    adelL delay gaRevL, 0.015
    adelR delay gaRevR, 0.012
    
    
    aL, aR reverbsc adelL, adelR, ksize, khfDamp
    
    
    aL butterhp aL, 150  
    aR butterhp aR, 150
    aL butterlp aL, 10000  
    aR butterlp aR, 10000
    
    
    kmod oscil 0.002, 0.3, 1
    adelmod1 = 0.001 + kmod
    adelmod2 = 0.001 - kmod
    aLmod vdelay aL, adelmod1*1000, 10
    aRmod vdelay aR, adelmod2*1000, 10
    aL = aLmod
    aR = aRmod
    
    
    gaCompL += aL*0.8
    gaCompR += aR*0.8
    
    gaRevL = 0
    gaRevR = 0
endin


instr 100
    
    ktime times
    
    
    if ktime < 16 then
        
        gksection = 0
        gkintensity = 0.3 + ktime/50
        gktempo = 280
    elseif ktime < 48 then
        
        gksection = 1
        gkintensity = 0.5 + (ktime-16)/60
        gktempo = 280 + (ktime-16)*1.25  
    elseif ktime < 64 then
        
        gksection = 2
        gkintensity = 0.4
        gktempo = 260
    elseif ktime < 128 then
        
        gksection = 3
        gkintensity = 0.95
        gktempo = 320 + (ktime-64)*1.25  
    elseif ktime < 163 then
        
        gksection = 3
        gkintensity = 1.0
        gktempo = 400  
    else
        
        gksection = 5
        gkintensity = 0
        gktempo = 0
    endif
    
    
    kbeat metro gktempo/60
    
    
    kcount init 0
    kbar init 0
    
    if kbeat == 1 then
        kcount += 1
        
        
        if kcount % 16 == 0 then
            kbar += 1
        endif
        
        
        if gksection == 0 then
            
            kpat1 euclidean 16, 6  
            kpat2 euclidean 16, 1
            kpat3 euclidean 16, 1
        elseif gksection == 1 then
            
            kpat1 euclidean 16, int(8 + gkintensity*6)  
            kpat2 euclidean 16, int(2 + gkintensity*3)
            kpat3 euclidean 8, int(2 + gkintensity*3)
        elseif gksection == 2 then
            
            kpat1 euclidean 16, 8  
            kpat2 euclidean 32, 2
            kpat3 euclidean 16, 2
        elseif gksection == 3 then
            
            kpat1 euclidean 16, int(10 + (kbar % 4)*2)  
            kpat2 euclidean 16, int(3 + (kbar % 2)*2)   
            kpat3 euclidean 8, int(4 + (kbar % 4))      
        else
            
            kpat1 = 0
            kpat2 = 0
            kpat3 = 0
        endif
        
        
        if kpat1 == 1 then
            kdur random 0.02, 0.15  
            kamp random 0.3, 0.7
            kpan random -0.8, 0.8
            kvar = (gksection == 3) ? 1 : ((gksection == 2) ? 2 : 0)
            event "i", 1, 0, kdur, kamp, kpan, kvar
        endif
        
        
        
        if kcount % 3 == 1 then
            event "i", 1, 0, 0.03, 0.4, random:i(-1, 1), 0
        endif
        
        
        if kcount % 8 == 3 then
            ki = 0
            while ki < 5 do
                event "i", 1, ki*0.02, 0.02, 0.5, random:i(-0.5, 0.5), 1
                ki += 1
            od
        endif
        
        
        if kpat2 == 1 then
            
            if gksection == 3 then
                kpitch random 30, 50
                kdist = 1.0
                event "i", 2, 0, 0.15, 0.9, kpitch, kdist
            elseif (kcount % 4) == 0 then
                kpitch random 40, 60
                kdist = gkintensity
                event "i", 2, 0, 0.2, 0.8, kpitch, kdist
            endif
        endif
        
        
        if kpat3 == 1 then
            
            if gksection == 3 then
                kdur random 0.01, 0.05
                kamp random 0.4, 0.8
                ktype = 1  
                event "i", 3, 0, kdur, kamp, ktype
            elseif (kcount % 3) != 0 then
                kdur random 0.02, 0.1
                kamp random 0.2, 0.5
                ktype = int(random:i(0, 3))
                event "i", 3, 0, kdur, kamp, ktype
            endif
        endif
        
        
        if gksection == 0 then
            
            if kcount % 64 == 1 then
                event "i", 4, 0, 8, 0.3, 50, 0
            endif
        elseif gksection == 1 then
            
            if kcount % 128 == 1 then
                event "i", 5, 0, 4, 0.2, 1
            endif
            
            if kcount % 32 == 1 then
                event "i", 4, 0, 4, 0.2, 100, 1
            endif
        elseif gksection == 2 then
            
            if kcount % 16 == 1 then
                event "i", 4, 0, 2, 0.4, 30, 2
            endif
        elseif gksection == 3 then
            
            if kcount % 32 == 1 then
                event "i", 4, 0, 2, 0.5, random:i(50, 200), int(random:i(0, 3))
            endif
            if kcount % 64 == 1 then
                event "i", 5, 0, 2, 0.3, (kbar % 2 == 0) ? 1 : -1
            endif
            
            krand random 0, 1
            if krand > 0.7 then  
                knum random 8, 12  
                ki = 0
                while ki < int(knum) do
                    event "i", 1, ki*0.01, 0.02, 0.6, random:i(-1, 1), 1
                    ki += 1
                od
            endif
            
            
            if kbar % 4 == 0 && kcount % 64 == 1 then
                
                ki = 0
                while ki < 8 do
                    event "i", 3, ki*0.03, 0.02, 0.3, 1  
                    ki += 1
                od
            endif
            
            if kbar % 4 == 2 && kcount % 32 == 1 then
                
                event "i", 2, 0, 0.5, 0.6, 25, 0.5
            endif
        elseif gksection == 4 then
            
        endif
        
        
        if kbar % 8 == 0 && kcount % 128 == 0 then
            
            if gksection == 1 || gksection == 3 then
                
                ki = 0
                while ki < 16 do
                    event "i", 3, ki*0.05, 0.03, 0.7, int(random:i(0, 3))
                    ki += 1
                od
            endif
        endif
        
        
        if kbar % 4 == 1 && gksection >= 1 then
            
            if kcount % 2 == 1 then
                event "i", 6, 0, 0.05, 0.3, 0  
            endif
            if kcount % 8 == 7 then
                event "i", 6, 0, 0.2, 0.4, 1   
            endif
        endif
        
        
        if kbar % 4 == 3 && gksection >= 1 then
            
            if kcount % 16 == 5 || kcount % 16 == 13 then
                kfreq = (gksection == 3) ? 100 : 150
                event "i", 7, 0, 0.1, 0.4, kfreq
            endif
        endif
        
        
        if gksection == 3 then
            
            kdens = (kbar % 8) / 8.0
            
            
            if kdens > 0.5 && kcount % 3 == 0 then
                event "i", 3, 0, 0.02, 0.3*kdens, 1
            endif
            
            
            if kbar % 2 == 0 then
                
                if kcount % 2 == 0 then
                    event "i", 1, 0, 0.01, 0.2, random:i(-1, 1), 0
                endif
            else
                
                if kcount % 8 == 3 then
                    event "i", 1, 0, 0.05, 0.4, random:i(-1, 1), 2
                endif
            endif
        endif
    endif
endin

</CsInstruments>
<CsScore>


; Function tables
f1 0 4096 10 1                    ; Sine wave
f2 0 4096 10 1 0.5 0.3 0.25 0.2  ; Complex waveform

; Start effects
i97 0 164  ; Compressor
i98 0 164  ; Delay
i99 0 164  ; Reverb

; Start master sequencer
i100 0 164  ; 2:44 composition

e
</CsScore>
</CsoundSynthesizer>