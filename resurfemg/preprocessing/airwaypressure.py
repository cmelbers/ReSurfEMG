# # Standard code libraries
import numpy as np
from scipy.signal import find_peaks

from resurfemg.preprocessing import filtering as filt

def find_peaks_vent(
        signal,
        fs,
        min_dif,
        peak,
):
    """
    Find peaks of ventilator signals
    """
    if peak == 'minima':
        indx, __ = find_peaks(-signal)
    elif peak == 'maxima':
        indx, __ = find_peaks(signal)
    else:
        print('Define if peaks are minima or maxima')

    indx = indx.tolist()
    val = signal[indx]
    val = val.tolist()

    # Remove false peaks by controling interval between peaks
    i = 0
    while i < len(indx)-1:
        dif_samples = indx[i+1]-indx[i]
        if dif_samples/fs < min_dif: # To do: betere argumentatie hiervoor vinden
            if val[i] >= val[i+1]:
                if peak == 'minima':
                    del indx[i]
                    del val[i]
                else:
                    del indx[i+1]
                    del val[i+1]
            else:
                if peak == 'minima':
                    del indx[i+1]
                    del val[i+1]
                else:
                    del indx[i]
                    del val[i]
        else:
            i = i+1

    # Remove false peaks by controling value

    indx = np.array(indx)
    val = np.array(val)
    time = indx/fs

    return time, val, indx

def remove_small_peaks(
        val, 
        indx, 
        fs
    ):
    """
    Remove smaller peaks based on a threshold determined by the mean of all peaks.
    """
    mean_peak_end = val.mean()
    indx = indx.tolist()
    val = val.tolist()

    i=0
    while i < len(indx)-1:
        if val[i]<mean_peak_end*0.7:
            del indx[i]
            del val[i]
        else:
            i = i+1

    indx = np.array(indx)
    val = np.array(val)
    time = indx/fs

    return time, val, indx

def extract_breathing_effort(
        indx_start_in_mv, 
        raw_signal, 
        fs
    ):
    """
    Extract breathing effort using the derivative of the signal and the start of the by the mechanical
    ventilation supported inhalation.
    """
    # Filter signal
    clean_signal = filt.emg_lowpass_butter_sample(
            raw_signal, 3, fs,)
    # Find part of signal with derivative of less than -5 cm H2O per second (= -0.25 cm H2O per 5 samples)
    indx_be_all = []
    nsamps = range(5,30)
    for i in indx_start_in_mv:
        for nsamp in nsamps:
            if (clean_signal[i-nsamp+5]-clean_signal[i-nsamp+1]) <= -0.25:
                for s in range(i-nsamp+1, i+1):
                    if s not in indx_be_all:
                        indx_be_all.extend(range(i-nsamp+1, i+1)) 
            old_nsamp = nsamp
    indx_be_all.sort()

    # Find beginning per part
    indx_be = [indx_be_all[0]]
    for i in range(1,len(indx_be_all)):
        if indx_be_all[i]-indx_be_all[i-1] > 1:
            indx_be.append(indx_be_all[i])

    # Use indexes on raw signal
    indx_be = np.array(indx_be)
    time_be = indx_be/fs
    be = raw_signal[indx_be]

    return time_be, be, indx_be

def remove_false_mininima(
        raw_signal,
        indx_start_in_mv, 
        indx_end_in_mv, 
        indx_be,
        fs
    ):
    """
    Removes minimum if they are not right before a maximum or right after a slope created by the breathing effort.
    """
    indx_start_in_mv = indx_start_in_mv.tolist()
    i = 0
    j = 0
    for indx in range(0,len(indx_end_in_mv)):
        list_of_minima_indx = []
        list_of_be_indx = []

    # Determine number of minima and starts of breathing effort before one tidal breath
        while i < len(indx_start_in_mv) and indx_start_in_mv[i] < indx_end_in_mv[indx]:
            list_of_minima_indx.append(indx_start_in_mv[i])
            i = i+1
        while j < len(indx_be) and indx_be[j] < indx_end_in_mv[indx]:
            list_of_be_indx.append(indx_be[j])
            j = j+1

        k=0
        if len(list_of_minima_indx) > 1:
            for indx_list in range(0,len(list_of_minima_indx)-1):
                if indx_list < len(list_of_minima_indx)-1:
                    if len(list_of_be_indx) > 0:
                        if list_of_be_indx[indx_list-k] > list_of_minima_indx[indx_list]:
                            indx_start_in_mv.remove(list_of_minima_indx[indx_list])
                            k = k+1
                    else:
                        indx_start_in_mv.remove(list_of_minima_indx[indx_list])

    indx_start_in_mv = np.array(indx_start_in_mv)
    time_start_in_mv = indx_start_in_mv/fs
    peak_start_in_mv = raw_signal[indx_start_in_mv]

    return time_start_in_mv, peak_start_in_mv, indx_start_in_mv

def pipeline_process_vent(
        raw_signal,
        fs
    ):
    """
    Process airway pressure data.
    """
    time_start_in_mv, peak_start_in_mv, indx_start_in_mv = find_peaks_vent(raw_signal, fs, 1.5, peak='minima') # so max respiratory rate of 40
    time_end_in_mv, peak_end_in_mv,  indx_end_in_mv = find_peaks_vent(raw_signal, fs, 1.5, peak='maxima',) # so max respiratory rate of 40

    time_end_in_mv, peak_end_in_mv,  indx_end_in_mv = remove_small_peaks(peak_end_in_mv,indx_end_in_mv, fs)

    time_be, be, indx_be = extract_breathing_effort(indx_start_in_mv, raw_signal, fs)

    time_start_in_mv, peak_start_in_mv, indx_start_in_mv = remove_false_mininima(raw_signal, indx_start_in_mv, indx_end_in_mv, indx_be,fs)

    return time_start_in_mv, peak_start_in_mv, time_end_in_mv, peak_end_in_mv, time_be, be
