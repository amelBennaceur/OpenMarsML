# Defines function for calculating tide amplitudes and phases.
# Heavily modified from IDL script mpdailyft.pro by SRL.
# KR, July 2021
import numpy as np

# Calculate the phase as hour of first maximum at Airy-0 (midnight is zero):
def phase_hr(w,phi):
    # based on Fourier component like: cos(2*pi*w*t + phi), t = 0,...,1
    # range of phi: (-pi,pi]
    if w==0:
        #print('Stationary wave')
        return np.nan
    else:    
        # The phase is P = w*t + phi/(2*pi), and t is restricted to be between 0 and 1
        # The phase hour is the smallest t value that makes P an integer, while remaining between 0 and 1
        # The three closest integers from midnight (t=0) are P = -1, 0, 1. 
        # Calculate the t values for each of these possibilities:        
        tm = (-1-(phi/(2*np.pi)))/w
        t0 = -phi/(2*np.pi*w)
        tp = (1-(phi/(2*np.pi)))/w        
        # restrict to the values that are within range
        inrange = []
        for t in [tm,t0,tp]:
            if t >=0 and t <=1:
                inrange.append(t)
        # Final t is the smallest value in inrange
        if len(inrange): # inrange has at least one element
            t = min(inrange)
        else: # no values - return nan
            return np.nan
        # t currently in range [0,1], with midnight=0. Change to range (-12,12], with midnight=0
        if t <= 0.5: 
            t = t*24
        else:
            t = t*24-24
        return t         

# Calculate the phase between (-pi,pi], given the hour of first maximum at Airy-0 (this ranges from -12 to 12)
def hr_phase(w,t):
    if w == 0:
        #print('Stationary wave')
        return np.nan
    if t == -12: # convert to +12 for consistency
        t+=24
    # convert time range from (-12,12] to [0,1]
    if t <= 12:
        t = t/24
    else:
        t = (t+24)/24
    # The phase is P = w*t + phi/(2*pi), and t is restricted to be between 0 and 1
    # t is the smallest value between 0 and 1 that makes P an integer, for the given phase phi/(2*pi)
    # The three closest integers from midnight (t=0) are P = -1, 0, 1. 
    # Calculate the phi values for each of these possibilities:            
    phim = 2*np.pi*(-1-w*t)
    phi0 = 2*np.pi*(0-w*t)
    phip = 2*np.pi*(1-w*t)
    # restrict to the phi values within range
    inrange = []
    for phi in [phim,phi0,phip]:
        if phi > -np.pi and phi <= np.pi:
            inrange.append(phi)
    # There should be only one entry
    if len(inrange) != 1:
        print('inrange:',inrange)
        raise RuntimeError('Expected only 1 possible phi value, but calculated '+str(len(inrange))+'.')
    phi = inrange[0]
    return phi
    

def tides_2d(ps, lat, lon, tim, tday, lt=False,verbose=False):
    # ps - time x lat x lon field array
    # lat, lon, tim - vectors for the resp space and time values
    # tday - number of timesteps per sol
    # lt - if True, returns phase as local time of first maximum at Airy-0
    # returned phases are in range (-pi,pi]

    # Definitions of model parameters
    lmax = lon.size
    mmax = lat.size
    tday = tday
    nlat = lat.size
    tmax = tim.size
    numdays = int(tim.size / tday)

    # ptl
    ptl = np.zeros((lmax, tmax, nlat))
    for i in range(nlat):
        ptl[:,:,i] = ps[:,i,:].transpose()

    # Definition of pressure field
    tp = np.zeros((lmax, tday))

    # Define a new time field
    time_day = np.zeros((numdays))

    ## Labelling of necessary parameters
    #lon = 360. * np.arange(lmax)/float(lmax)
    #lat = 180. * ((np.arange(mmax) + 0.5) / float(mmax) - 0.5)
    #tim = np.arange(tday)/float(tday)
    nfreq = tday# + 1

    # First define some common longitude and time indices
    nl2 = int(lmax/2+1) # Number of longitudinal frequencies. Includes a Nyquist frequency if even, doesn't if odd    
    ind0 = int((nfreq-1)/2) # index of the temporal zero frequency
    nfpm = ind0 # the number of positive and negative frequency pairs
    nf2 = int(nfreq/2) # temporal Nyquist frequency index (if even)
    # We will check for the Nyquist frequency later
    
    # Define the set of longitudinal wavenumbers:
    k = np.arange(nl2)    
    # Define the set of positive and negative frequencies:             
    w = np.arange(nfreq)-nf2+1 
    # if nfreq = 9: w = [-4,-3,-2,-1,0,1,2,3,4]
    # if nfreq=10: w = [-4,-3,-2,-1,0,1,2,3,4,5]    

    amp = np.zeros((nfreq, nl2))
    pha = np.zeros((nfreq, nl2))
    bmp = np.zeros((numdays, nfreq, nl2, nlat))
    qha = np.zeros((numdays, nfreq, nl2, nlat))

    for lati in range(nlat):
        # Loop for different days
        for day in range(numdays):
            if verbose == True:
                print('Calculating spectrum for day ',day+1,' of ',numdays,', latitude ',lati+1,' of ',nlat)
            time_day[day] = day
            tp[:,:] = ptl[:,tday*day:tday*day+tday,lati] # tp shape is lmax x tday       
            #vv = np.zeros((lmax))
            
            # Initialise array for holding first fourier transform amplitude
            ss = np.zeros((lmax, tday), dtype=np.complex_)

            # Initialise arrays for holding real and imaginary parts of fourier amplitudes            
            aa = np.zeros((nl2,tday)) # real component of ssr
            bb = np.zeros((nl2,tday)) # imaginary component of ssr
            cc = np.zeros((nl2,tday)) # real component of ssi
            dd = np.zeros((nl2,tday)) # imaginary component of ssi
            
            # Step 1: Fourier transform in longitude at each time level, and store results in ss
            for n in range(tday):
                vv = tp[:,n]
                ss[:,n] = np.fft.fft(vv) / lmax # Normalise by 1/nlon
                # ss[k,:] is time series (over one day) of the kth longitudinal spectral component
            # Step 2: For each longitudinal wavenumber, split ss into real and imaginary components to get real time series, and then Fourier transform each of these in time
            for l in range(nl2): # for all longitude components
                ddr = ss[l,:].real # the (real) time series of the real component of the l'th mode
                ddi = ss[l,:].imag # the (real) time series of the imaginary component of the l'th mode                                
                # Calculate discrete fourier transform for each. Normalise by 1/nt since the numpy default definition is different
                ssi = np.fft.fft(ddi) / tday 
                ssr = np.fft.fft(ddr) / tday 
                # ssr,ssi are the (complex) time spectra of the l'th longitudinal component's real and imaginary parts
                # Store the real and imaginary part of these time series in aa,bb,cc,dd:                
                aa[l,:] = ssr.real; bb[l,:] = ssr.imag
                cc[l,:] = ssi.real; dd[l,:] = ssi.imag
            
            # Step 3: calculate the amplitudes and phases for the different wavenumber-frequency pairs, exploiting symmetries where present                                       
            # In the longitudinal frequency space, all frequencies except the zero and Nyquist frequencies occur as conjugate pairs since the original time series is real. 
            # The information from the pair can be thus fully represented given only one part of the pair, since the other is simply the complex conjugate
            # The net real contribution of these pairs yields an amplitude that is double that of the single part, hence we double the amplitudes
            # This does not apply to the zero and Nyquist frequencies as they are already real and are not paired; hence these two are treated separately without doubling.
            for l in range(nl2): # each longitudinal frequency / frequency pair                
                if l == 0 or (lmax%2==0 and l == nl2-1): # zero and Nyquist frequencies, don't double the amplitudes
                    pass
                else: # double all the other components
                    aa[l,:] = 2*aa[l,:]; bb[l,:] = 2*bb[l,:]; 
                    cc[l,:] = 2*cc[l,:]; dd[l,:] = 2*dd[l,:];
                
                # The set of coefficients can now be used to derive the amplitudes and phases of both the positive and negative temporal frequencies
                # Start with the m=0 case      
                amp[ind0,l] = np.sqrt((aa[l,0]-dd[l,0])**2 + (bb[l,0]+cc[l,0])**2) # we are keeping the formal formula, but in this case dd=0 and bb=0
                pha[ind0,l] = np.angle((aa[l,0]-dd[l,0])+(bb[l,0]+cc[l,0])*1j)    
                # Now do all the positive and negative temporal frequency pairs
                for m in range(1,nfpm+1):
                    # positive frequencies
                    amp[ind0+m,l] = np.sqrt((aa[l,m]-dd[l,m])**2 + (bb[l,m]+cc[l,m])**2)
                    pha[ind0+m,l] = np.angle((aa[l,m]-dd[l,m])+(bb[l,m]+cc[l,m])*1j)    
                    if lt is True:
                        # convert the phase into hour of first peak
                        pha[ind0+m,l] = phase_hr(w[ind0+m],pha[ind0+m,l])                    
                    # negative frequencies
                    amp[ind0-m,l] = np.sqrt((aa[l,m]+dd[l,m])**2 + (-bb[l,m]+cc[l,m])**2)
                    pha[ind0-m,l] = np.angle((aa[l,m]+dd[l,m])+(-bb[l,m]+cc[l,m])*1j)    
                    if lt is True:
                        # convert the phase into hour of first peak
                        pha[ind0-m,l] = phase_hr(w[ind0-m],pha[ind0-m,l])                                        
                # Finally, do the temporal Nyquist frequency (if it exists)
                if nfreq/2 % 2 == 0: # temporal Nyquist frequency exists
                    amp[ind0+nf2,l] = np.sqrt((aa[l,nf2]-dd[l,nf2])**2 + (bb[l,nf2]+cc[l,nf2])**2) # we are keeping the formal formula, but in this case dd=0 and bb=0
                    pha[ind0+nf2,l] = np.angle((aa[l,nf2]-dd[l,nf2])+(bb[l,nf2]+cc[l,nf2])*1j)                     
                    if lt is True:
                        # convert the phase into hour of first peak
                        pha[ind0+nf2,l] = phase_hr(w[ind0+nf2],pha[ind0+nf2,l])   
            # Store the outputs for this sol in bmp and qha
            bmp[day,:,:,lati] = amp[:,:]
            qha[day,:,:,lati] = pha[:,:]
    return bmp,qha,w,k

def reconstruct(bmp,qha,k,w,nlon,lt):
    # function that reconstructs a signal based on its spectral data.
    # the array returned, ps_r, has dimensions nl2 x nfreq x nlon x (ndays*nfreq) x nlat
    # So ps_r[i,j,:,:,lati] is the longitude x time array of the mode with wavenumber k[i] and frequency w[j], at latitude index lati.    
    ndays,nfreq,nl2,nlat = bmp.shape
    tday=nfreq
    ps_r = np.zeros((nl2,nfreq,nlon,ndays*nfreq,nlat))
    for i_lat in range(nlat):
        for i_day in range(ndays): # for each day
            print('Reconstructing day ',i_day+1,' of ',ndays,' latitude ',i_lat+1,' of ',nlat)
            for i_lon in range(nlon):
                for i_t in range(tday):
                    amp = bmp[i_day,:,:,i_lat] 
                    pha = qha[i_day,:,:,i_lat]                
                    for i_l in range(nl2):
                        for i_m in range(nfreq):
                            if lt is True:
                                # convert phase back to normal range [-pi,pi]
                                pha[i_m,i_l] = hr_phase(w[i_m],pha[i_m,i_l])                            
                            # calculate the value of this mode at this longitude and time
                            ps_r[i_l,i_m,i_lon,i_day*nfreq+i_t,i_lat] = \
                                amp[i_m,i_l]*np.cos(2*np.pi*(w[i_m]*i_t/tday+k[i_l]*i_lon/nlon)+pha[i_m,i_l])
    return ps_r

def getmodes(ps, lat, lon, tim, tday, bmp, qha, w, N, modes, output):

    # ps - time x lat x lon field array on which fourier transform was applied
    # lat, lon, tim - vectors for the resp space and time values
    # tday - number of timesteps per sol
    # bmp, qha - ndays x nfreq x nwave x nlat arrays of fourier mode amplitudes and phases
    # w - vector of spectral frequencies (cycles per day)
    # N - integer, number of days over which to do a block time-average (N >= 1)
    # modes - 'diurnal', 'semidiurnal', 'kelvin1', 'kelvin2', 'kelvin3'
    # output - 'raw' (original field units) or 'normalised' (normalised percentages)

    # Definitions of model parameters
    lmax = lon.size
    mmax = lat.size
    tday = tday
    nlat = lat.size
    tmax = tim.size
    numdays = int(tim.size / tday)
    nfreq = tday + 1
    
    # ptl
    ptl = np.zeros((lmax, tmax, nlat))
    for i in range(nlat):
        ptl[:,:,i] = ps[:,i,:].transpose()

    # Define a new time field
    time_day = np.arange(numdays)
    
    # Get specific tidal modes from bmp (amplitude) and qha (phase)
    
    # Define arrays to hold the daily spectral data for each of the requested modes
    ma = [np.zeros((numdays,nlat)) for i in range(len(modes))]
    mp = [np.zeros((numdays,nlat)) for i in range(len(modes))]

    # Define arrays to hold the multi-day average
    ma_rm = [np.zeros((int(numdays/N),nlat)) for i in range(len(modes))]
    mp_rm = [np.zeros((int(numdays/N),nlat))  for i in range(len(modes))]
        
    # Loop over latitudes
    for lati in range(nlat):
        # Calculate the daily mean of the original field
        pres = ptl[:,:,lati] # lon x time 
        ave = pres.mean(axis=0) # zonal mean at each time point
        # Now calculate the time mean for each day
        mean = np.zeros((numdays,))
        for k in range(numdays):
            if (output=='normalised'):
                mean[k] = np.sum(ave[k*tday:(k+1)*tday]) / tday
            elif (output=='raw'):
                mean[k] = 1.
        
        for ind in range(len(modes)):
            mode = modes[ind]
            if (mode=='diurnal'): # Diurnal
                freq = 1; waven = 1
            elif (mode=='semidiurnal'): # Semidiurnal
                freq = 2; waven = 2
            elif (mode=='terdiurnal'): # Terdiurnal
                freq = 3; waven = 3
            elif (mode=='kelvin1'): # Kelvin mode, wavenumber 1
                freq = -1; waven = 1
            elif (mode=='kelvin2'): # Kelvin mode, wavenumber 2
                freq = -1; waven = 2
            elif (mode=='kelvin3'): # Kelvin mode, wavenumber 3
                freq = -1; waven = 3    
            # get index of location of frequency
            freq_index = np.where(w == freq)[0][0]
            
            for ti in range(numdays):
                # Assign relevant subset of the bmp amplitude array to ma, performing the relevant normalisation
                ma[ind][ti,lati] = bmp[ti,freq_index,waven,lati] / mean[ti]
                # Translate the phase to the day hour of first maximum at Airy-0, using phase_hr
                mp[ind][ti,lati] = phase_hr(w[freq_index],qha[ti,freq_index,waven,lati])
    
            # Apply a time mean over N days
            for nn in range(int(numdays/N)):
                ma_rm[ind][nn,lati] = ma[ind][nn*N:nn*N +N,lati].mean(axis=0)
                mp_rm[ind][nn,lati] = mp[ind][nn*N:nn*N +N,lati].mean(axis=0)
                
    if (output=='normalised'):
        for ind in range(len(modes)):
            ma_rm[ind][:] = 100. * ma_rm[ind][:]
            
    Nh1 = int(N/2); Nh2 = N - Nh1
    time_day_rm = time_day[Nh1:-Nh2:N]
    
    return ma_rm, mp_rm, time_day_rm
