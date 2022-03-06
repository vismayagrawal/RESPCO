"""
General utility functions
"""

import numpy as np
import peakutils
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt

############
### Misc ###
############
import math
def get_z_score(corr, n):
	"""
	Get z score according to Chang 2009
	Z = atah(pearsonr)*sqrt(n-3)
	"""
	return (math.atan(corr))*np.sqrt(n-3)

####################
### Input Output ###
####################
def read_txt(filename):
	with open(filename) as f:
		lines = f.read().splitlines()
	return lines

# def read_txt_m2(filename):
#   with open(filename) as f:
#       lines = [float(i.strip()) for i in f]
#     return lines

def read_txt_xy(path):
	"""
	This reads the text file when the data is both in x and y direction
	e.g. 
	12  234 2.3
	42 1   11

	The function works even if the gap varies
	"""
	content = []
	with open(path) as f:
		for line in f:
			tokens = line.strip().split(' ')
			for token in tokens:
				if len(token)!=0:
					content.append(np.float32(token))
	return content

def write_txt(filename, lines):
	with open(filename, 'w') as f:
		for line in lines:
			f.write("%s\n"%line)

#####################################
### Matplotlib Interactive Mode #####
#####################################
# import time
# import matplotlib.pyplot as plt
# from contextlib import contextmanager
# def end_interactive(fig):
#   """ end interaction in a plot created with %matplotlib notebook """
#   plt.gcf().canvas.draw()
#   time.sleep(0.5)
#   plt.close(fig)
# @contextmanager
# def interactive_plot(close=True, fig=None):
#   if fig is None:
#       fig = plt.figure()
#   yield
#   plt.tight_layout()
#   if close:
#       end_interactive(fig)

#################
### Normalize ###
#################
def std_normalise(x):
	"""zscore normalization of input array"""
	return (x - np.mean(x))/ np.std(x)

def min_max_normalise(x):
	"""min max normalization of input array"""
	return (x - np.min(x))/ (np.max(x) - np.min(x))

def median_translate(x):
	"""median translation of input array"""
	return x - np.median(x)

def mean_translate(x):
	"""mean translation of input array"""
	return x - np.mean(x)

##################
### Trend Line ###
##################
def trendline(index,data, order=1):
	"""Gives slope of the fit"""
	coeffs = np.polyfit(index, list(data), order)
	slope = coeffs[-2]
	return float(slope)


###########################
### related to plotting ###
###########################
def x_val_plot(y, fs):
	return np.linspace(0,len(y)/fs, len(y))

def freqSpectrum(signal, Fs):
	"""returns the amp (y axis) and freq (x axis) to plot a frequency spectrum of input signal"""
	amp = np.abs(np.fft.rfft(signal))
	freq = np.linspace(0, Fs/2, len(amp))
	return amp, freq

######################
# Processing signals #
######################

def cross_correlation_using_fft(x, y):
	from numpy.fft import fft, ifft, fftshift
	f1 = fft(x)
	f2 = fft(np.flipud(y))
	cc = np.real(ifft(f1 * f2))
	return fftshift(cc)

def compute_shift(x, y):
	"""
	shift > 0 means that x starts 'shift' time steps after y
	shift < 0 means that y starts 'shift' time steps before x 
	"""
	assert len(x) == len(y)
	c = cross_correlation_using_fft(x, y)
	assert len(c) == len(x)
	zero_index = int(len(x) / 2) - 1
	shift = zero_index - np.argmax(c)
	return shift

def compute_shift_with_limit(x, y, limit_secs = 120, Fs = 10):
	"""
	limit sec is individually set for both the sides
	thus, limit_secs = 120 means (+- 120) shift allowed
	shift > 0 means that x starts 'shift' time steps after y
	shift < 0 means that y starts 'shift' time steps before x 
	"""
	assert len(x) == len(y)
	c = cross_correlation_using_fft(x, y)
	assert len(c) == len(x)
	zero_index = int(len(x) / 2) - 1

	# neutralize the value of c for all out of range. So that it doesnt detect in argmax
	c_mean = np.mean(c)
	if zero_index - limit_secs*Fs > 0:
		c[:zero_index - limit_secs*Fs] = c_mean
	if zero_index + limit_secs*Fs < len(c):
		c[zero_index + limit_secs*Fs:] = c_mean

	shift = zero_index - np.argmax(c)
	return shift

def delay_correction(a, b, negative_relationship = True):
	"""
	Correct delay with signals with same length
	Args:
		a: 1st signal
		b: 2nd signal
		negative_relationship: True when the signals are like CO2 and Resp (negative relationship)
			In other words, if negative relationship is set as true, it will invert signal b
	Returns:
		corrected signal a, b and the delay between them
	"""
	if negative_relationship:
		shift =  compute_shift(a, -b)
	else:
		shift =  compute_shift(a, b)
	
	if shift==0:
		return a, b, 0
	elif shift>0:
		b = b[shift:]
		a = a[:-shift]
	else: #shift is negative
		a = a[abs(shift):]
		b = b[:-abs(shift)]
	return a, b, shift

def delay_correction_with_limit(a, b, limit_secs = 120, Fs = 10, negative_relationship = True):
	"""
	Correct delay with signals with same length. LIMIT to delay is applied
	Args:
		a: 1st signal
		b: 2nd signal
		negative_relationship: True when the signals are like CO2 and Resp (negative relationship)
			In other words, if negative relationship is set as true, it will invert signal b
	Returns:
		corrected signal a, b and the delay between them
	"""
	if negative_relationship:
		shift =  compute_shift_with_limit(a, -b, limit_secs, Fs)
	else:
		shift =  compute_shift_with_limit(a, b, limit_secs, Fs)
	
	if shift==0:
		return a, b, 0
	elif shift>0:
		b = b[shift:]
		a = a[:-shift]
	else: #shift is negative
		a = a[abs(shift):]
		b = b[:-abs(shift)]
	return a, b, shift

######## DEPRECATED: because it doesn't consider negative delay. 
## Though I can include negative delay code too, but since I found another better solution (above function), I am using it instead
# def delay_correction(co2, resp):
#   """ 
#   removed the delay between 2 signals with negative relationship
#   It translated the second signal forward (or we can say 1st signal backwards)
#   input = co2, resp signal argmaxray
#   output = co2_corrected, resp_corrected, delay
#   """
#   ## padding: because fft method does circular conv
#   co2_pad = np.append((co2), np.zeros_like(resp))
#   resp_pad = np.append((resp), np.zeros_like(co2))
	
#   delay = np.argmax(abs(np.fft.ifft(np.fft.fft(co2_pad) * np.conj(np.fft.fft(-resp_pad))))[:len(resp)])
#   if delay==0:
#       return co2, resp, 0
#   co2_corrected = co2[delay:]
#   resp_corrected = resp[:-delay]

#   return co2_corrected, resp_corrected, delay

def get_peaks(signal, Fs, thres=0.3):
	"""
	Useful for getting PETCO2 or in general peak of a signal
	"""
	peak_index = peakutils.indexes(signal, thres, min_dist=2*Fs)
	peak_amplitude = signal[peak_index]

	## Making starting and ending as mean of the data for many different reasons
	## like consistent length of RVT and PETCO2 after interpolation
	peak_index = np.array([0, *peak_index, len(signal)])
	peak_amplitude = np.array([np.mean(peak_amplitude), *peak_amplitude, np.mean(peak_amplitude)])

	return peak_index, peak_amplitude

def get_troughs(signal, Fs, thres=0.3):
	peak_index, peak_amplitude = get_peaks(-signal, Fs, thres)
	return peak_index, -peak_amplitude

def get_rvt(resp, Fs, thres=0.2):
    ## Finding Peaks
    indexes_rvt_peaks = peakutils.indexes(resp, thres, min_dist=2.5*Fs)
    rvt_peaks = resp[indexes_rvt_peaks]

    ## Finding Troughs
    indexes_rvt_troughs = peakutils.indexes(-resp, thres, min_dist=2.5*Fs)
    rvt_troughs = resp[indexes_rvt_troughs]

    ## Getting RVT
    ## RVT = (peak - trough) / time exhale
    ## putting rvt pts in mid of peak and trough =>     rvt_index = int(peak_index + next_peak_index)/2)
    rvt = []
    rvt_index = []
    for i, peak_index in enumerate(indexes_rvt_peaks[:-1]): ## Not including last peak as we want complete cycle
        next_peak_index = indexes_rvt_peaks[i + 1]
        if np.nonzero(indexes_rvt_troughs > peak_index)[0].size:
            trough_index = indexes_rvt_troughs[np.nonzero(indexes_rvt_troughs > peak_index)[0][0]]
            if trough_index < next_peak_index:
                temp = (resp[peak_index] - resp[trough_index]) / (next_peak_index - peak_index)
                if temp > 0:
                    rvt.append(temp)
                    rvt_index.append(int((trough_index + peak_index)/2))
    # rvt = utils.median_translate(np.array(rvt))
    rvt_index = np.array(rvt_index)

    ## Making starting and ending as mean of the data for many different reasons
    ## like consistent length of RVT and PETCO2 after interpolation
    rvt_index = np.array([0, *rvt_index, len(resp)])
    rvt = np.array([np.mean(rvt), *rvt, np.mean(rvt)])

    return rvt_index, rvt, (indexes_rvt_peaks, rvt_peaks, indexes_rvt_troughs, rvt_troughs)

def get_rvt_method2(resp, Fs, thres=0.2):
	"""
	Method given in Birn et al. 2008 (or Chang 2009)

	returns interpolated rvt
	"""
	indexes_rvt_peaks_ = peakutils.indexes(resp, thres, min_dist=2.5*Fs)
	indexes_rvt_peaks = np.array([0, *indexes_rvt_peaks_, len(resp)-1])
	rvt_peaks_ = resp[indexes_rvt_peaks_]
	rvt_peaks = np.array([np.mean(rvt_peaks_), *rvt_peaks_, np.mean(rvt_peaks_)])
	rvt_peaks_interpolated = interpolate_signal(indexes_rvt_peaks, rvt_peaks, len(resp))

	## Finding Troughs
	indexes_rvt_troughs_ = peakutils.indexes(-resp, thres, min_dist=2.5*Fs)
	indexes_rvt_troughs = np.array([0, *indexes_rvt_troughs_, len(resp)-1])
	rvt_troughs_ = resp[indexes_rvt_troughs_]
	rvt_troughs = np.array([np.mean(rvt_troughs_), *rvt_troughs_, np.mean(rvt_troughs_)])
	rvt_troughs_interpolated = interpolate_signal(indexes_rvt_troughs, rvt_troughs, len(resp))

	tr1 = indexes_rvt_peaks_[0:-1]
	tr2 = indexes_rvt_peaks_[1:]
	tr_ = tr2 - tr1
	tr = np.array([np.mean(tr_), *tr_, np.mean(tr_)])
	indexes_tr_ = (tr1 + tr2) / 2
	indexes_tr = np.array([0, *indexes_tr_, len(resp)-1])

	tr_interpolated = interpolate_signal(indexes_tr, tr, len(resp))

	rvt = (rvt_peaks_interpolated - rvt_troughs_interpolated) / tr_interpolated

	return rvt

def get_petco2_interpolated(co2, Fs, len_interpolated = None, thres=0.3):
	"""
	Calculated PETCO2 for a given CO2 signal and interpolates it
	Args:
		co2: input co2 signal, array or a list
		len_interpolated: it can be the length of input co2 signal or in a case different length is desired we can use it
	"""
	if not len_interpolated:
		len_interpolated = len(co2)
	
	petco2_index, petco2_amplitude = get_peaks(co2, Fs, thres=thres)
	## Upsampling
	petco2_interpolated = interpolate_signal(petco2_index, petco2_amplitude, len_interpolated)
	return petco2_interpolated


###############
### Filters ###
###############

def moving_average(x, w = 10, mode = 'valid'):
	return np.convolve(x, np.ones(w), mode) / w

def butter_bandpass_filter(data, lowcut, highcut, Fs, order=5):
	"""Butterworth bandpass filter """
	data_mean = np.mean(data)
	data = data - data_mean #to prevent any distrotions due to dc component
	nyq = 0.5 * Fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], btype='band',analog=False,output='sos')
	y = sosfilt(sos, data)
	y = y + data_mean
	return y

def butter_bandstop_filter(data, lowcut, highcut, Fs, order=5):
	"""Butterworth bandstop filter"""
	data_mean = np.mean(data)
	data = data - data_mean #to prevent any distrotions due to dc component
	nyq = 0.5 * Fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], btype='bandstop',analog=False,output='sos')
	y = sosfilt(sos, data)
	y = y + data_mean
	return y

def butter_highpass_filter(data, cutoff, Fs, order=2):
	"""Butterworth highpass filter """
	data_mean = np.mean(data)
	data = data - data_mean #to prevent any distrotions due to dc component
	nyq = 0.5 * Fs
	normal_cutoff = cutoff / nyq
	sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
	y = sosfilt(sos, data)
	y = y + data_mean
	return y

def butter_lowpass_filter(data, cutoff, Fs, order=2):
	"""Butterworth lowpass filter """
	data_mean = np.mean(data)
	data = data - data_mean #to prevent any distrotions due to dc component
	nyq = 0.5 * Fs
	normal_cutoff = cutoff / nyq
	sos = butter(order, normal_cutoff, btype='low', analog=False, output = 'sos')
	y = sosfilt(sos, data)
	y = y + data_mean
	return y

##### DEPRECATED
# def outlier(in_data, z_threshold = 3):
#   """Remove Outlier """
#   in_data = (in_data - in_data.mean()) / in_data.std()
#   in_data_mean = in_data.mean()
#   out_data = in_data.copy()
	
#   for pos, val in enumerate(in_data): 
#       if val > z_threshold:
#           out_data[pos] = z_threshold
#       elif val < -z_threshold:
#           out_data[pos] = -z_threshold

#   return out_data

def remove_outlier(in_data, z_threshold = 3):
	"""Remove Outlier without any rescaling"""
	# in_data = (in_data - in_data.mean()) / in_data.std()
	out_data = in_data.copy()
	
	val_max = in_data.mean() + z_threshold*in_data.std()
	val_min = in_data.mean() - z_threshold*in_data.std()

	idx_max = in_data > val_max
	idx_min =  in_data < val_min

	out_data[idx_max] = val_max
	out_data[idx_min] = val_min

	return out_data

def interpolate_signal(indexes, amplitude, len_interpolated):
	"""
	interpolate_signal the signal given index, respective amplitudes and interpolation length
	Note: I initially named it upsample_signal(indexes, amplitude, upsample_len)
	"""
	f = interp1d(indexes, amplitude, kind = 'linear')
	amplitude_interpolated = f(np.linspace(indexes[0],indexes[-1],num = len_interpolated))
	return amplitude_interpolated

def resample_signal(signal, Fs, resampled_Fs = 10):
	"""
	Resample the signal given the signal (amplitudes at equally spaced timepoints), Fs and resampled_Fs (may increase or decrease the sampling)
	Note: I initially named it downsample_signal(signal, Fs, downsampled_Fs = 10)
	"""
	len_signal = len(signal)
	resample_len = int((len_signal/Fs) * resampled_Fs)
	f = interp1d(np.arange(0,len_signal), signal, kind = 'linear')
	resampled = f(np.linspace(0, len_signal-1, num = resample_len))
	return resampled