# # Standard code libraries
import numpy as np
from scipy.signal import find_peaks

from resurfemg.preprocessing import filtering as filt

#To do: integereren met extract_val algemeen
def extract_val(
        time, 
        idxs,
        signal, 
        fs
    ):
    """ 
     Find values which are correlated to the timing or indexes of events. Choose to extract
     the values with the time with input idxs=None. Extract the values with the
     indexes with input time=None and fs=None.

     Input:
    :param min_dif: minimum distance between 2 breaths
    :type min_dif: ~int
    :param peak: kind of peak to detect (maxima or minima)
    :type peak: ~str    

    Output:
    :param time: times from which the values need to be extracted
    :type time: ~numpy.ndarray
    :param idxs: idxs from which the values need to be extracted
    :type idxs: ~numpy.ndarray
    :param signal: signal from which the values need to be extracted
    :type signal: ~numpy.ndarray
    param fs: sample frequency
    :type fs: int

    Output:
    :param val: extracted values
    :type val: ~numpy.ndarray
     """
    if idxs is None:
        idxs = [int(t * fs) for t in time] 
    val = signal[idxs]

    return val

def find_peaks_vent(
        self,
        min_dif,
        peak,
):
    """
    Find peaks of ventilator signals and remove the samples too
    close to each other.
    
    Input:
    :param min_dif: minimum distance between 2 breaths
    :type min_dif: ~int
    :param peak: kind of peak to detect (maxima or minima)
    :type peak: ~str    

    Output:
    :param time: times at which peaks occur
    :type time: ~numpy.ndarray
    :param vals: values corresponding to the times at which peaks occur
    :type vals: ~numpy.ndarray
    :param idxs: indexes at which peaks occur
    :type idxs: ~numpy.ndarray
    """
    signal = self.channels[0].y_raw
    fs = self.channels[0].fs
    # Invert signal if minima need to be found
    if peak == 'minima':
        idxs, __ = find_peaks(-signal)
    elif peak == 'maxima':
        idxs, __ = find_peaks(signal)
    else:
        print('Define if peaks are minima or maxima')

    # Remove false peaks by controling interval between peaks
    vals = extract_val(None, idxs, signal, fs)
    idxs = idxs.tolist()
    vals = vals.tolist()
    
    i = 0
    while i < len(idxs)-1:
        dif_samples = idxs[i+1]-idxs[i]
        if dif_samples/fs < min_dif: 
            if vals[i] >= vals[i+1]:
                if peak == 'minima':
                    del idxs[i]
                    del vals[i]
                else:
                    del idxs[i+1]
                    del vals[i+1]
            else:
                if peak == 'minima':
                    del idxs[i+1]
                    del vals[i+1]
                else:
                    del idxs[i]
                    del vals[i]
        else:
            i = i+1

    idxs = np.array(idxs)
    vals = extract_val(None, idxs, signal, fs)
    time = idxs/fs

    return time, vals, idxs

def remove_small_peaks(
        self,
        threshold_percentile=0.7
    ):
    """
    Remove smaller peaks based on a threshold (default = 0.7) determined 
    by the mean of all peaks.

    Input:
    :param threshold: percentile under which peaks are removed
    :type threshold: ~float

    Output:
    :param time: times at which valid peaks occur
    :type time: ~numpy.ndarray
    :param vals: values corresponding to the times at which valid peaks occur
    :type vals: ~numpy.ndarray
    :param idxs: indexes at which valid peaks occur
    :type idxs: ~numpy.ndarray
    """

    val = self.peak_end_in_mv
    idx = self.idx_end_in_mv
    fs = self.fs
    
    # Define threshold
    mean_peak_end = val.mean()
    threshold = mean_peak_end*threshold_percentile

    # Remove peaks that are below the threshold
    idx = idx.tolist()
    val = val.tolist()
    i=0
    while i < len(idx)-1:
        if val[i]<threshold:
            del idx[i]
            del val[i]
        else:
            i = i+1

    idx = np.array(idx)
    val = np.array(val)
    time = idx/fs

    return time, val, idx

def extract_breathing_trigger(
        self
    ):
    """
    Extract breathing trigger (bt) using the derivative of the signal and the start of the by the mechanical
    ventilation supported inhalation.

    Output:
    :param time_bt: times at which a breathing trigger occur
    :type time_bt: ~numpy.ndarray
    :param val_bt: values corresponding to the times at which a breathing trigger occur
    :type val_bt: ~numpy.ndarray
    :param idx_bt: indexes at which a breathing trigger occur
    :type idx_bt: ~numpy.ndarray
    """
    idx_start_mv = self.idx_start_in_mv
    raw_signal = self.channels[0].y_raw 
    fs = self.fs
    
    # Filter signal
    clean_signal = filt.emg_lowpass_butter_sample(
            raw_signal, 3, fs,)
    # Find part of signal with derivative of less than -5 cm H2O per second (= -0.25 cm H2O per 5 samples)
    # infront of the start of the support from the mechanical ventilation (mv)
    idx_bt_all = []
    nsamps = range(5,30)
    for i in idx_start_mv:
        for nsamp in nsamps:
            if (clean_signal[i-nsamp+5]-clean_signal[i-nsamp+1]) <= -0.2:
                for s in range(i-nsamp+1, i+1):
                    if s not in idx_bt_all:
                        idx_bt_all.extend(range(i-nsamp+1, i+1))
    idx_bt_all.sort()

    # Find the first sample per part with a sufficient derivative
    idx_bt = [idx_bt_all[0]]
    for i in range(1,len(idx_bt_all)):
        if idx_bt_all[i]-idx_bt_all[i-1] > 1:
            idx_bt.append(idx_bt_all[i])

    # Find values by using indexes and raw signal
    idx_bt = np.array(idx_bt)
    time_bt = idx_bt/fs
    val_bt = raw_signal[idx_bt]

    return time_bt, val_bt, idx_bt

def remove_false_mv_starts(self):
    """
    Removes the start of the mechanical ventilator (mv) supporting the inhalation if they are not before 
    a an end of the mv supporting an inhalation or after the start of a breathing trigger.

    Output:
    :param time_start_in_mv: times at which the MV starts to support
    :type time_start_in_mv: ~numpy.ndarray
    :param val_start_in_mv: values corresponding to the times at which the MV starts to support
    :type val_start_in_mv: ~numpy.ndarray
    :param idx_start_in_mv: indexes at which the MV starts to support
    :type idx_start_in_mv: ~numpy.ndarray
    """

    raw_signal = self.channels[0].y_raw 
    idx_start_in_mv = self.idx_start_in_mv 
    idx_end_in_mv = self.idx_end_in_mv
    idx_bt = self.idx_end_in_mv
    fs = self.fs

    idx_end_in_mv = idx_end_in_mv.tolist()
    idx_start_in_mv = idx_start_in_mv.tolist()
    i = 0
    j = 0
    for idx in range(0,len(idx_end_in_mv)):
        list_of_start_in_idx = []
        list_of_bt_idx = []

        # Determine which starts of an mv supporting an inhalation and which starts of a breathing trigger 
        # occur before the end of the mv supporting an inhalation
        while i < len(idx_start_in_mv) and idx_start_in_mv[i] < idx_end_in_mv[idx]:
            list_of_start_in_idx.append(idx_start_in_mv[i])
            i = i+1
        while j < len(idx_bt) and idx_bt[j] < idx_end_in_mv[idx]:
            list_of_bt_idx.append(idx_bt[j])
            j = j+1

        # When multiple starts of the mv supporting an inhalation occur and breathing triggers are 
        # detected, remove the starts wich do not follow a breathing trigger. If there are no breathing
        # triggers, keep only the lowest minimum.
        k=0
        if len(list_of_start_in_idx) > 1:
            if len(list_of_bt_idx) > 0:
                for l in range(len(list_of_start_in_idx)-1):
                    if k < (len(list_of_bt_idx)):
                        if list_of_bt_idx[k] > list_of_start_in_idx[l]:
                            idx_start_in_mv.remove(list_of_start_in_idx[l])
                        else:
                            k += 1
            else:
                list_of_start_in_idx.sort()
                higher_minima = list_of_start_in_idx[1:]
                for minimum in higher_minima:
                    idx_start_in_mv.remove(minimum)

    idx_start_in_mv = np.array(idx_start_in_mv)
    time_start_in_mv = idx_start_in_mv/fs
    val_start_in_mv = raw_signal[idx_start_in_mv]

    return time_start_in_mv, val_start_in_mv, idx_start_in_mv
