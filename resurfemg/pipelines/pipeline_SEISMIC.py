# # Standard code libraries
import os
import platform
import glob
import numpy as np
import pandas as pd
import statistics
import math
import bisect
from scipy import signal
from scipy.signal import butter, lfilter
from resurfemg.preprocessing import envelope as evl
from resurfemg.preprocessing import airwaypressure as paw
from resurfemg.postprocessing import event_detection as ed
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

# Custom code libraries from ReSurfEMG
#repo_path = 'C:\SEISMIC_study\ReSurfEMG'
#sys.path.append(repo_path)

from resurfemg.config.config import Config
from resurfemg.data_classes.data_classes_SEISMIC import (
VentilatorDataGroup, EmgDataGroup, TimeSeries)
from resurfemg.preprocessing.ecg_removal import *

# General functions

def time_to_value(
        times,
        signal,
        fs
    ):
    """
    Extract the value

    :param times: times at which the values need to be extracted
    :type time: ~numpy.ndarray or list
    :param signal: signal from which the values need to be extracted
    :type signal: ~numpy.ndarray
    :param fs: sample frequency
    :type fs: ~int  

    Return:
    :param val: values of times at which the values need to be extracted
    :type val: ~numpy.ndarray or list
    """
    idxs = []
    for time in times:
        idx = time*fs
        idx_round = round(idx)
        idxs.append(idx_round)
    val = signal[idxs]

    return val

# Functions for event detection
def idx_to_value(
        idxs,
        signal
    ):
    """ 
    Find values which are correlated to the timing or indexes of events. Choose to extract
    the values with the time with input idxs=None. Extract the values with the
    indexes with input time=None and fs=None.

    :param time: times at which the values need to be extracted
    :type time: ~numpy.ndarray or list
    :param idxs: indexes at which the values need to be extracted
    :type idxs: ~numpy.ndarray or list
    :param signal: signal from which the values need to be extracted
    :type signal: ~numpy.ndarray
    :param fs: sample frequency
    :type fs: ~int  

    Return:
    :param val: values of indexes at which the values need to be extracted
    :type val: ~numpy.ndarray or list
    """
    val = []

    # To do: test met dit weghalen?
    max_index = len(signal) - 1
    valid_idxs = [idx for idx in idxs if 0 <= idx <= max_index]

    # Extract the values
    if len(valid_idxs) == 0:
        val = []
    else:
        val = signal[valid_idxs]
    val = np.array(val)

    return val

def take_valid_peaks(
        emg_signal, 
        peak_parameters,
        outcomes,
        fs
    ):
    """
    Split the peaks in valid peaks and invalid peaks according to the quality assesment within the peak 
    detection (see event_detection.onoffpeak_baseline_crossing()) and the quality assesment from 
    emg_timeseries.quality_check(). These are combined in the function quality_assement() in data_classes.py.

    :param emg_signal: emg signal on which the results of the quality assesment is integrated
    :type ems_signal: ~numpy.ndarray
    :param peak_parameters: indexes of the top of the peak, the start of the peak and the end of the peak
    :type peak_parameters: nested list
    :param outcomes: the result of the quality assesment per category
    :type outcomes: dataframe

    Return:
    :param dict_peak_parameters: per parameter (top, end and begin) a dictionary with the valid and invalid peaks,
    where the nested dictionary contains the indexes, values and times of the peak
    :type dict_peak_parameters: ~numpy.ndarray or list
    """
    parameter_names = ['peak_top', 'peak_start', 'peak_end']
    # Select peaks that pass all tests as valid (True)
    idxs_peaks_only_true = []
    for i in range(len(outcomes)):
            if outcomes.iloc[i,1] == True and outcomes.iloc[i,2] == True and  outcomes.iloc[i,3] == True and  outcomes.iloc[i,4] == True and  outcomes.iloc[i,5] == True: 
                idxs_peaks_only_true.append(i)

    # Split the lists of the beginning of the peaks, the ending of the peaks and the top of 
    # the peaks in valid and invalid peaks
    dict_peak_parameters ={}
    for i in range(len(peak_parameters)):
        # Create a lists with the valid and invalid indexes
        valid_idxs_parameter = []
        invalid_idxs_parameter = []
        for j in range(len(peak_parameters[i])):
            if j in idxs_peaks_only_true:
                valid_idxs_parameter.append(peak_parameters[i][j])
            else:
                invalid_idxs_parameter.append(peak_parameters[i][j])
        valid_idxs_parameter = np.array(valid_idxs_parameter)
        invalid_idxs_parameter = np.array(invalid_idxs_parameter
                                        )
        
        # Extract values of the parameters
        valid_values_parameter = idx_to_value(valid_idxs_parameter, emg_signal)
        invalid_values_parameter = idx_to_value(invalid_idxs_parameter, emg_signal)

        # Add to the dictionary all indexes, values and times of the parameter with a nested dictionary containing lists 
        # of the valid and invalid indexes, values and times
        dict_peak_parameters[parameter_names[i]] = {'valid':{'idx':valid_idxs_parameter, 'value': valid_values_parameter, 'time':valid_idxs_parameter/fs},
                                        'invalid':{'idx':invalid_idxs_parameter, 'value':invalid_values_parameter, 'time':invalid_idxs_parameter/fs}}

    return dict_peak_parameters

# Functions to detect auto-trigger

def detect_auto_trigger(
        signal, 
        time_start_in_mv, 
        valid_time_activity, 
        invalid_idx_activity,
        fs,
        emg=False
    ):
    """
    Detects autotrigger based on activity before the start of the mechanical ventilation during inhalation.

    :param signal: signal which is used to detect the effort of the patient
    :type signal: ~numpy.ndarray
    :param time_start_in_mv: times at which the start of the MV supporting the inhalation is detected
    :type time_start_in_mv: list
    :param valid_idx_activity: the valid times at which effort was detected
    :type valid_idx_activity: list
    :param invalid_idx_activity: the invalid times at which effort was detected
    :type invalid_idx_activity: list
    :param fs: sampling frequency of the signal
    :type fs: int
    :param emg: if the signal which is used for the detection is an sEMG signal (default = False)
    :type emg: bool

    Return:
    :param time_auto_trigger: times at which the auto-triggers occur
    :type time_auto_trigger: ~numpy.ndarray or list
    :param val_auto_trigger: values of the times at which the auto-triggers occur
    :type val_auto_trigger: ~numpy.ndarray or list
    """
    no_time_auto_trigger = []
    time_auto_trigger = []
    i = 0
    
    # Create a dataframe with labels if the detected efforts of the patient are valid or invalid
    if invalid_idx_activity is not None:
        valid_time_activity = valid_time_activity.tolist()
        invalid_activity = invalid_idx_activity/fs
        invalid_activity = invalid_activity.tolist()
        all_time_activity = valid_time_activity + invalid_activity
    else:
        all_time_activity = valid_time_activity.tolist()

    all_time_activity.sort()
    valid_or_invalid = []

    # Create a list stating if the peak is valid or invalid
    for i in all_time_activity:
        if i in valid_time_activity:
            valid_or_invalid.append(True)
        else:
            valid_or_invalid.append(False)
    df_time_activity = pd.DataFrame(list(zip(all_time_activity, valid_or_invalid)), columns=['Value', 'Valid'])

    # If there is no activity 2 seconds before the start of MV it is detected as auto-trigger. The false activity
    # starts are excluded and therefore noted as no auto-trigger
    i = 0
    no_activity_before = True
    no_activity_after = True
    for time_mv in time_start_in_mv:
        while all_time_activity[i] < time_mv and i <= len(all_time_activity)-2 and time_mv <= len(signal)/fs:
            if time_mv-all_time_activity[i] < 2 or df_time_activity.iloc[i,1] == False:
                no_time_auto_trigger.append(time_mv)
            else:
                # Determine if there was any activity in the 10 seconds before the analysed breath
                if emg == True and time_mv-all_time_activity[i] < 10:
                        no_activity_before = False           
            i = i+1
        # Determine if there was any activity in the 10 seconds after the analysed breath
        if all_time_activity[i]-time_mv < 10 and emg == True:
            no_activity_after = False
        # If no activity was detected around the analysed breath, it was marked as no auto-trigger
        if time_mv not in no_time_auto_trigger and no_activity_before == True and no_activity_after == True:
            no_time_auto_trigger.append(time_mv)
        no_activity_before = True
        no_activity_after = True

    # Couple time to values
    time_auto_trigger = [i for i in time_start_in_mv if i not in no_time_auto_trigger]

    val_auto_trigger = time_to_value(time_auto_trigger, signal, fs)

    return time_auto_trigger, val_auto_trigger

# Functions to detect wasted effort

def minimal_rr_check(
        time_wasted_effort, 
        max_rr=40
    ):
    """
    Removes detected wasted efforts when there are too much wasted efforts in an interval for a 
    maximum respiratory rate.

    :param time_wasted_effort: times at which a wasted effort occurs
    :type time_wasted_effort: ~numpy.ndarray or list
    :param max_rr: maximum respiratory rate with a default of 40 breaths per minute
    :type max_rr: int

    Return:
    :param time_wasted_effort: times at which the valid wasted efforts occur
    :type time_wasted_effort: ~numpy.ndarray or list
    """
    valid_time_wasted_effort = []
    minimal_distance_breaths = 60/max_rr

    # Find the valid wasted efforts
    for i in range(1, len(time_wasted_effort)):
        if time_wasted_effort[i] - time_wasted_effort[i-1] > minimal_distance_breaths:
            valid_time_wasted_effort.append(time_wasted_effort[1])
            
    return time_wasted_effort

def detect_wasted_effort(
        signal, 
        time_valid_start_peak_emg, 
        time_end_in_mv, 
        fs
    ):
    """
    Detects wasted efforts based on activity before the start of the mechanical ventilation during inhalation.

    :param signal: signal which is used to detect the effort of the patient
    :type signal: ~numpy.ndarray
    :param time_true_start_peak_emg: times at which the starts of valid peaks occur
    :type time_true_start_peak_emg: ~numpy.ndarray or list
    :param time_end_in_mv: times at which the ventilator stops to support the inhalation
    :type time_end_in_mv: ~numpy.ndarray or list
    :param fs: sample frequency
    :type fs: ~int

    Return:
    :param time_wasted_effort: times at which the wasted efforts occur
    :type time_wasted_effort: ~numpy.ndarray or list
    :param val_wasted_effort: values of the times at which the wasted efforts occur
    :type val_wasted_effort: ~numpy.ndarray or list
    """
    no_wasted_effort = []
    i = 1

    # If there is no time_end_in_mv 1 seconds after the time_valid_start_peak_emg it is detected as wasted effort
    for idx_peak in range(0, len(time_valid_start_peak_emg)-1):
        while time_end_in_mv[i] < time_valid_start_peak_emg[idx_peak+1] and i <= len(time_end_in_mv)-2:
            if time_end_in_mv[i]-time_valid_start_peak_emg[idx_peak] < 3:                        
                no_wasted_effort.append(time_valid_start_peak_emg[idx_peak])
            i = i+1

    # Couple time to values
    time_wasted_effort = [i for i in time_valid_start_peak_emg if i not in no_wasted_effort]
    time_wasted_effort = minimal_rr_check(time_wasted_effort)
    val_wasted_effort = time_to_value(time_wasted_effort, signal, fs)

    return time_wasted_effort, val_wasted_effort

# Functions to compare the detected asynchronies

def add_to_df(
        df, 
        time_events_signal1, 
        idx_time_signal1, 
        time_events_signal2, 
        closest, 
        none_counter,
        used_time_signal2
    ):
    """
    Add closest event of signal 2 to the event of signal 1 to a dataframe.

    :param df: dataframe to which the information must be added
    :type df: dataframe
    :param time_events_signal1: times at which the event occurs on one of the signals
    :type time_events_signal1: ~numpy.ndarray or list
    :param idx_time_signal1: index of the time in time_events_signal1 that is closest
    :type idx_time_signal1: int
    :param time_events_signal2: times at which the event occurs on one of the signals
    :type time_events_signal2: ~numpy.ndarray or list
    :param closest: the time in signal 2 with the smallest distance to signal1
    :type closest: ~int
    :param none_counter: number of None's that is detected
    :type none_counter: ~int
    :param used_signal2: times in signal 2 that are already coupled to a time in signal 1
    :type used_signal2: ~numpy.ndarray or list

    Return:
    :param df: the matched events (every row is an event)
    :type df: dataframe
    :param last_is_none: information about if the last event was not matched
    :type last_is_none: bool
    :param none_counter: number of not matched events
    :type none_counter: int
    :param closest: index at which the closest event was
    :type closest: int
    :param used_time_signal2: events in second signal that are already matched
    :type used_time_signal2: list
    """
    # When the event in signal 1 is matched with an event in signal 2
    if closest is not None:
        corresponding_event = [time_events_signal1[idx_time_signal1], time_events_signal2[closest]]
        last_is_none = False
        used_time_signal2.append(time_events_signal2[closest])
    # When no event in signal 2 is matched to the event in signal 1
    else:
        corresponding_event = [time_events_signal1[idx_time_signal1], None]
        none_counter += 1
        last_is_none = True
    df.loc[len(df)] = corresponding_event

    return df, last_is_none, none_counter, closest, used_time_signal2

def match_events(
        event,
        signal1_name,
        time_events_signal1,
        signal2_name,
        time_events_signal2,
):
    """
    Compare the time of events detected with events detected on different signals to match the events

    :param event: name of detected event
    :type event: str
    :param signal1_name: name of one of the signals that is compared
    :type signal1_name: str
    :param time_events_signal1: times at which the event occurs on one of the signals
    :type time_events_signal1: ~numpy.ndarray or list
    :param signal2_name: name of other of the signals that is compared
    :type signal2_name: str
    :param time_events_signal2: times at which the event occurs on the other the signal
    :type time_events_signal2: ~numpy.ndarray or list

    Return:
    :param df: all the events matched to events in the other signal at the same time
    :type df: dataframe
    """
    df = pd.DataFrame(columns=[signal1_name,signal2_name])

    # Count the None's which represent the number of events that is not 
    # matched to another event on the other signal
    none_counter_s1 = 0
    none_counter_s2 = 0
    last_is_none = True
    
    used_time_signal2 = [] # list to note the times that are matched to an event on signal 1

    # Set in chronological order
    time_events_signal1.sort() 
    time_events_signal2.sort()
    
    # Find the closest event in time and analyse if it is close enough
    last_idx = 0
    idx_time_signal2 = 0
    if len(time_events_signal1) != 0:
        for idx_time_signal1 in range(0,len(time_events_signal1)-1):
            closest = None
            closest_time = 5 # set as a large difference

            # Re-evaluate the last evaluated time if it was not matched
            if last_is_none is True:
                idx_time_signal2 = last_idx
            last_idx = idx_time_signal2

            # Calculate difference to the times in signal 2 and find the closest
            while idx_time_signal2 < len(time_events_signal2):

                # Evaluate the times till they pass the next time of signal 1
                if time_events_signal2[idx_time_signal2] < time_events_signal1[idx_time_signal1+1]:
                    time_dif = time_events_signal1[idx_time_signal1] - time_events_signal2[idx_time_signal2] 

                    # Compare the calculated difference whith the difference that was the lowest until this moment
                    if abs(time_dif) < 2 and time_dif < closest_time:
                            closest = idx_time_signal2
                            closest_time = time_dif
                    idx_time_signal2 +=1
                else:
                    break

            # Add information about the matched events in a dataframe
            df, last_is_none, none_counter_s1, closest, used_time_signal2 = add_to_df(
                df, time_events_signal1, idx_time_signal1, time_events_signal2, closest, none_counter_s1, used_time_signal2)

        # Do the previous for-loop again, but for the last event detected on signal 1   
        closest = None
        closest_time = 5 # set as a large difference

        # Re-evaluate the last evaluated time if it was not matched
        if last_is_none is True:
            idx_time_signal2 = last_idx
        last_idx = idx_time_signal2

        # Calculate difference to the times in signal 2 and find the closest
        while idx_time_signal2 < len(time_events_signal2):
            time_dif = time_events_signal1[-1] - time_events_signal2[idx_time_signal2]

            # Compare the calculated difference whith the difference that was the lowest until this moment
            if abs(time_dif) < 2 and time_dif < closest_time:
                    closest = idx_time_signal2
                    closest_time = time_dif
            idx_time_signal2 += 1

        # Add information about the matched events in a dataframe
        df, last_is_none, none_counter_s1, closest, used_time_signal2 = add_to_df(
            df, time_events_signal1, -1, time_events_signal2, closest, none_counter_s1, used_time_signal2)

    # Add all the events of signal 2 that were not matched to the dataframe
    for time_event in time_events_signal2:
        if time_event not in used_time_signal2:
            no_match = [None, time_event]
            none_counter_s2 += 1
            df.loc[len(df)] = no_match

    none_counter = none_counter_s1 + none_counter_s2 

    print(event, 'between', signal1_name, 'and', signal2_name, 'compared:')
    print('Number of', event, 'on only one signal: ', none_counter, 'out of', len(df))
    print('Of which', none_counter_s1, 'only on', signal1_name)
    print('and', none_counter_s2, 'only on', signal2_name, '\n')
        
    return df

def match_3_signals_auto_trigger(
        df, 
        time_events_signal3, 
        fs_emg, 
        fs_vent,
        signal1,
        signal2,
        signal3
    ):
    """
    Match the events of 3 signals

    :param df: dataframe cotaining the matched events of the first to signals
    :type df: dataframe
    :param events_signal3: times at which the event occurs on the third signal
    :type events_signal3: ~numpy.ndarray
    :param fs_emg: sample frequency emg
    :type fs_emg: int
    :param fs_vent: sample frequency airway pressure
    :type fs_vent: int
    :param signal1: first signal to compare
    :type signal1: ~numpy.ndarray or list
    :param signal2: second signal to compare
    :type signal2: ~numpy.ndarray or list
    :param signal3: third signal to compare
    :type signal3: ~numpy.ndarray or list

    Return:
    :param all_signals_compared: lists of which signals are found on 1, 2 or 3 signals and specified which signals
    :type all_signals_compared: dict
    :param df: all the events matched to events in the other signals at the same time
    :type df: dataframe
    """
    # Create list add matched events of the third signal to
    signal_3_matched = [None] * len(df)

    # Sort events on signal three in chronical order
    events_signal3_sorted = sorted(time_events_signal3)
    
    # Find the closest event in time and analyse if it is close enough
    for i in range(len(df)):
        closest_time = 5 # set as a large difference
        isnan1 = math.isnan(df.iloc[i, 0])
        isnan2 = math.isnan(df.iloc[i, 1])
        if isnan1 != True:
            idx = bisect.bisect(events_signal3_sorted, df.iloc[i, 0])
            # Calculate difference between the times in signal 1 and in signal 3 and find the closest
            if idx > 0 and abs(events_signal3_sorted[idx - 1] - df.iloc[i, 0]) < closest_time:
                closest_time = abs(events_signal3_sorted[idx - 1] - df.iloc[i, 0])
                closest_event = events_signal3_sorted[idx - 1]
            # Calculate difference between the last time in signal 1 and in signal 3 and find the closest
            if idx < len(events_signal3_sorted) and abs(events_signal3_sorted[idx] - df.iloc[i, 0]) < closest_time:
                closest_time = abs(events_signal3_sorted[idx] - df.iloc[i, 0])
                closest_event = events_signal3_sorted[idx]
        else: 
            idx = bisect.bisect(events_signal3_sorted, df.iloc[i, 1])
            # Calculate difference between the times in signal 2 and in signal 3 and find the closest
            if idx > 0 and abs(events_signal3_sorted[idx - 1] - df.iloc[i, 0]) < closest_time:
                closest_time = abs(events_signal3_sorted[idx - 1] - df.iloc[i, 0])
                closest_event = events_signal3_sorted[idx - 1]
            # Calculate difference between the last time in signal 2 and in signal 3 and find the closest
            if idx < len(events_signal3_sorted) and abs(events_signal3_sorted[idx] - df.iloc[i, 0]) < closest_time:
                closest_time = abs(events_signal3_sorted[idx] - df.iloc[i, 0])
                closest_event = events_signal3_sorted[idx]
                
        # Add to list if an event of signal 3 is matched to the events of signal 1 and 2
        if closest_time != 5:
            signal_3_matched[i]=closest_event
        else:
            signal_3_matched[i]=None

    # Add matched events and not matched events to dataframe
    df['airway_pressure'] = signal_3_matched
    for event in time_events_signal3:
        if event not in signal_3_matched:
            no_match = [None, None, event]
            df.loc[len(df)] = no_match

    # Compare all rows of the dataframe to know on how many signals the events were detected

    # Create lists to contain information
    only_emg0 = []
    only_emg1 = []
    only_vent = []
    emg0_and_emg1 = []
    emg0_and_vent = []
    emg1_and_vent = []
    all_three = []

    # Order the events
    for i in range(len(df)):
        isnan1 = pd.isna(df.iloc[i,0])
        isnan2 = pd.isna(df.iloc[i,1])
        isnan3 = pd.isna(df.iloc[i,2])
        if isnan1 is False:
            if isnan2 is False:
                if isnan3 is False:
                    all_three.append(df.iloc[i,0])
                else:
                    emg0_and_emg1.append(df.iloc[i,0])
            else:
                if isnan3 is False:
                    emg0_and_vent.append(df.iloc[i,0])
                else:
                    only_emg0.append(df.iloc[i,0])
        else:
            if isnan2 is False:
                if isnan3 is False:
                    emg1_and_vent.append(df.iloc[i,1])
                else:
                    only_emg1.append(df.iloc[i,1])
            else:
                only_vent.append(df.iloc[i,2])

    not_all_3_signals = len(only_emg0) + len(only_emg1) + len(only_vent) + len(emg0_and_emg1) + len(emg0_and_vent) + len(emg1_and_vent)

    # Create a dictionary with all information
    # The data is saved multiple time in the dictionary for the next for loop.
    emg0 = {'times': [all_three, emg0_and_emg1, emg0_and_vent, only_emg0], 'signal': signal1, 'fs':fs_emg}
    emg1 = {'times': [all_three, emg0_and_emg1, emg1_and_vent, only_emg1], 'signal': signal2, 'fs':fs_emg}
    vent = {'times': [all_three, emg0_and_vent, emg1_and_vent, only_vent], 'signal': signal3, 'fs':fs_vent}
    all_signals_compared = {'diaphragm':emg0, 'intercostal':emg1, 'airway pressure':vent}

    # Extract values of the events
    for __, data in all_signals_compared.items():
        vals = []
        for t in data['times']:
            val = time_to_value(t, data['signal'], data['fs'])
            vals.append(val)
        data['values'] = vals

    # Print outcomes
    print('Auto-trigger between EMGs and the airwaypressure compared:')
    print('Number of auto-triggers on not all signals: ', not_all_3_signals, 'out of', len(df))
    print('Of which', len(only_emg0), 'only on the diaphragm EMG')
    print('Of which', len(only_emg1), 'only on the intercostal EMG')
    print('Of which', len(only_vent), 'only on the ventilator')
    print('Of which', len(emg0_and_emg1), 'only on both the EMGs')

    return all_signals_compared, df

def seismic_analysis(
        emg_file_chosen, 
        vent_file_chosen
        ):
    
    """
    Run the full analysis to detect auto-triggers and wasted efforts and compare these.

    :param emg_file_chosen: directory to emg signal
    :type emg_file_chosen: str
    :param vent_file_chosen: directory to airway pressure signal
    :type vent_file_chosen: str

    Return:
    :param emg_timeseries: all information retrieved in the method emg_timeseries
    :type emg_timeseries: object
    :param df_wasted_effort: all the wasted efforts in both sEMG signals matched to eachother
    :type df_wasted_effort: dataframe
    :param df_auto_trigger_emgs: all the auto-triggers in both sEMG signals matched to eachother
    :type df_auto_trigger_emgs: dataframe
    """

    # Load the EMG and ventilator data recordings from the selected folders.
    y_emg = np.load(emg_file_chosen)
    fs_emg = 2048
    y_vent = np.load(vent_file_chosen)
    fs_vent = 100

    # Store the EMG data in a group of TimeSeries objects
    emg_timeseries = EmgDataGroup(
        y_emg,
        fs=fs_emg,
        labels=['Costmar', 'Intercost'],
        units=2*['uV'],
        remove_length=5
        )
    
    # Store the ventilator data in a group of TimeSeries objects
    vent_timeseries = VentilatorDataGroup(
        y_vent,
        fs=fs_vent,
        labels=['Paw'],
        units=['cmH2O'],
        remove_length = 5)

    # Filter
    emg_timeseries.filter()

    # Remove outliers    
    emg_timeseries.remove_outliers()
    
    # Gate the EMG
    emg_timeseries.gating(peak_fraction=0.4, remove_outliers=True, gate_width_samples = fs_emg//7)
    
    # Calculate the envelope of the signal
    emg_timeseries.envelope()
    
    # Calculate the baseline for the EMG envelopes and p_vent
    emg_timeseries.baseline()

    # Check the quality of the preprocessed data
    emg_timeseries.quality_check()

    # Create a datframe with the total number of false and true peaks per measurement
    emg_timeseries.quantify_outcomes()

    # Process airway pressure
    vent_timeseries.pipeline_process_vent()

    # Use found peaks and peak assesment to split the true and false peaks    
    peak_idxs = emg_timeseries.channels[0].peaks['breaths'].peak_df.iloc[:,0]
    peak_idxs, peak_start_idxs, peak_end_idxs, __, __, __ = ed.onoffpeak_baseline_crossing(
        emg_timeseries.channels[0].y_env,emg_timeseries.channels[0].y_baseline, peak_idxs)
    peak_parameters = [peak_idxs, peak_start_idxs, peak_end_idxs]
    dict_peak_parameters_emg0 = take_valid_peaks(emg_timeseries.channels[0].y_env, peak_parameters, emg_timeseries.channels[0].df_quality_outcomes, fs_emg)

    peak_idxs = emg_timeseries.channels[1].peaks['breaths'].peak_df.iloc[:,0]
    peak_idxs, peak_start_idxs, peak_end_idxs, __, __, __ = ed.onoffpeak_baseline_crossing(
        emg_timeseries.channels[1].y_env,emg_timeseries.channels[1].y_baseline, peak_idxs)
    peak_parameters = [peak_idxs, peak_start_idxs, peak_end_idxs]
    dict_peak_parameters_emg1 = take_valid_peaks(emg_timeseries.channels[1].y_env, peak_parameters, emg_timeseries.channels[1].df_quality_outcomes, fs_emg)

    # Detect auto-trigger from airway pressure
    #time_auto_trigger_vent, val_auto_trigger_vent = detect_auto_trigger(
    #    vent_timeseries.channels[0].y_raw, vent_timeseries.time_start_in_mv, vent_timeseries.time_bt, None, fs_vent) 

    # Detect auto-trigger from sEMG
    time_auto_trigger_emg0, __ = detect_auto_trigger(
        emg_timeseries.channels[0].y_env, vent_timeseries.time_start_in_mv, dict_peak_parameters_emg0['peak_start']['valid']['time'], 
        dict_peak_parameters_emg0['peak_top']['invalid']['idx'], fs_emg, emg=True)
    time_auto_trigger_emg1, __ = detect_auto_trigger(
        emg_timeseries.channels[1].y_env, vent_timeseries.time_start_in_mv, dict_peak_parameters_emg1['peak_start']['valid']['time'], 
        dict_peak_parameters_emg1['peak_top']['invalid']['idx'], fs_emg, emg=True)

    # Detect wasted effort
    time_wasted_effort_emg0, __ = detect_wasted_effort(
        emg_timeseries.channels[0].y_env, dict_peak_parameters_emg0['peak_start']['valid']['time'], vent_timeseries.time_end_in_mv, fs_emg)
    time_wasted_effort_emg1, __ = detect_wasted_effort(
        emg_timeseries.channels[1].y_env, dict_peak_parameters_emg1['peak_start']['valid']['time'], vent_timeseries.time_end_in_mv, fs_emg)

    df_wasted_effort = match_events('wasted effort','diaphragm sEMG', time_wasted_effort_emg0, 'intercostal sEMG', time_wasted_effort_emg1)
    df_auto_trigger_emgs = match_events('auto-trigger', 'diaphragm sEMG', time_auto_trigger_emg0, 'intercostal sEMG', time_auto_trigger_emg1)
    #dict_auto_trigger, df_auto_trigger = match_3_signals_auto_trigger(df_auto_trigger_emgs, time_auto_trigger_vent,fs_emg, fs_vent, emg_timeseries.channels[0].y_env, 
    #                               emg_timeseries.channels[1].y_env, vent_timeseries.channels[0].y_raw)
    

    return emg_timeseries, df_wasted_effort, df_auto_trigger_emgs