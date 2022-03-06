"""
Convert ACQ files to CSV 
Extracts CO2 and RESP Data 
Calculate PetCO2 and RVT 
Saves everything 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import peakutils
from tqdm import tqdm

from scipy.stats.stats import pearsonr 

import utils
import preprocess_utils
import logging  

plt.rcParams["figure.figsize"] = [16,8]

#######################
### User Parameters ###
#######################
# dir requirements

### Baycrest biopac data
data_name = 'baycrest_biopac'
root_dir = os.path.abspath('../data/raw_physio_backup-biopac-20180417/')
raw_files_dir = os.path.join(root_dir, 'acq')


# folders to be created
output_dir = os.path.join(root_dir, 'preprocessed')
petco2_dir = os.path.join(output_dir, 'petco2')
rvt_dir = os.path.join(output_dir, 'rvt')
co2_dir = os.path.join(output_dir, 'co2')
resp_dir = os.path.join(output_dir, 'resp')
plots_dir = os.path.join(output_dir, 'plots')
stats_dir = os.path.join(output_dir, 'stats')
log_pth = os.path.join(output_dir, 'logs.txt')
downsample_fs = 10 #hz
apply_normalization = False

# export data or not
save_stats = True
save_data = True
save_plot = False

# Filter requirements.
order = 2
cutoff_high = 1 #Hz
cutoff_low = 0.05 #Hz

def make_output_dirs(output_dir):
	if save_data:
		os.makedirs(petco2_dir, exist_ok = True)
		os.makedirs(rvt_dir, exist_ok = True)
		os.makedirs(co2_dir, exist_ok = True)
		os.makedirs(resp_dir, exist_ok = True)
	if save_plot:
		os.makedirs(plots_dir, exist_ok = True)
	if save_stats:
		os.makedirs(stats_dir, exist_ok = True)

def main():
	df = preprocess_utils.parse_co2_resp(os.path.join(root_dir,raw_files_dir), data_name)

	corr_resp_co2_list = []
	corr_rvt_petco2_list = []
	name_list = []
	delay_list = []
	trend_reject = 0
	less_corr_reject = 0
	less_length_reject = 0
	logging.info('----------- Main Calculation --------------')
	for index in tqdm(df.index):
		
		name = df.iloc[index]['name'].split('.')[0]
		Fs = df.iloc[index]['Fs']
		
		##########################
		### Initial preprocess ###
		##########################
		co2 = df.iloc[index]['raw_co2']
		resp = df.iloc[index]['raw_resp']

		# # we will subract the mean now and will add again afterwards 
		# # if we don't do this step, there will be errors in midway in the code
		mean_co2 = np.mean(co2)
		mean_resp = np.mean(resp) 

		## Butterworth
		## Note: de_mean of input is done to remove DC component. 
		## If DC component is not removed, resulted signal has distortions in the start
		## this might be relevant: https://dsp.stackexchange.com/a/8981
		## I have tried with other examples too, DON'T USE this filter WIHTOUT DE-MEANED INPUT
		co2  = utils.butter_lowpass_filter(co2 - mean_co2, cutoff_high, Fs, order)
		# resp = utils.butter_lowpass_filter(resp - mean_resp, cutoff, Fs)
		resp = utils.butter_bandpass_filter(resp - mean_resp, cutoff_low, cutoff_high, Fs, order)



		## Check Trend and Reject if trend is present
		slope_resp = utils.trendline(np.arange(len(resp)),resp) * len(resp)/(np.max(resp) - np.min(resp))
		slope_co2 = utils.trendline(np.arange(len(co2)),co2) * len(co2)/(np.max(co2) - np.min(co2))
		if abs(slope_co2) > 0.4 or abs(slope_resp) > 0.4: 
			logging.info('\n ------Removed because of trend---- \n Name: %s \n Index: %d', name, index)
			trend_reject = trend_reject + 1
			continue
			
		## Removing the Delay
		co2, resp, delay = utils.delay_correction(co2, resp, negative_relationship = True)
		delay_secs = delay/Fs

		corr = pearsonr(-resp,co2)[0]
		if corr < 0.4:
			logging.info('\n ------Removed because of LESS CORRELATION---- \n Name: %s \n Index: %d \n pearson_corr: %f', name, index, corr)
			less_corr_reject = less_corr_reject + 1
			continue
			
		# if len(co2) < Fs*180: ## less than 3 minutes of the data
		# 	logging.info('\n ------Removed because of TIME LENGTH---- \n Name: %s \n Index: %d', name, index)
		# 	less_length_reject = less_length_reject + 1
		# 	print(len(co2)/(Fs*60))
		# 	continue

		if apply_normalization:
			co2 = utils.std_normalise(co2)
			resp = utils.std_normalise(resp)
		else:
			# Undo the de-mean done during the lowpass filter Normalization
			co2 = co2 + mean_co2
			resp = resp + mean_resp

		## Outlier Removal
		# co2 = utils.remove_outlier(co2)
		# resp = utils.remove_outlier(resp)

		## Downsampling to 10Hz for CNN
		co2 = utils.resample_signal(co2, Fs, resampled_Fs = 10)
		resp = utils.resample_signal(resp, Fs, resampled_Fs = 10)
		Fs = 10 #updating sampling freq to downsampled_fs
		
		######################
		### petco2 and rvt ###
		######################
		petco2_index, petco2 = utils.get_peaks(co2, Fs, thres = 0.3)
		rvt_index, rvt, peak_trough_tuple = utils.get_rvt(resp, Fs, thres=0.2)
		indexes_rvt_peaks, rvt_peaks, indexes_rvt_troughs, rvt_troughs = peak_trough_tuple
		if save_plot:
			# t_e = int(len(co2)/(Fs*60))
			index_val = np.arange(len(co2))/(60*Fs)
			plt.figure()
			plt.subplot(211)
			plt.plot(petco2_index/(60*Fs),petco2, label = 'petco2')
			plt.plot(index_val, co2, label = 'co2')
			plt.legend()
			plt.xlabel('Time [mins]')
			plt.ylabel('Amplitude [V]')
			plt.title('CO2 peaks: ' + str(len(petco2_index)))
			
			plt.subplot(212)
			plt.plot(indexes_rvt_peaks/(60*Fs), rvt_peaks, label = 'resp_peaks')
			plt.plot(index_val, resp, label = 'resp')
			plt.plot(indexes_rvt_troughs/(60*Fs), rvt_troughs, label = 'resp_troughs') 
			plt.xlabel('Time [mins]')
			plt.ylabel('Amplitude [V]')
			plt.title('RESP peaks: ' + str(len(rvt_peaks)) + ';RESP troughs: ' + str(len(rvt_troughs)))
			plt.legend()

			plt.savefig(os.path.join(plots_dir, name + "_plot.png"), dpi = 250)
			plt.close() 


		# Upsample: Interpolation to imaging Tr #
		rvt_upsampled = utils.interpolate_signal(rvt_index, rvt, len_interpolated = len(co2))
		petco2_upsampled = utils.interpolate_signal(petco2_index, petco2, len_interpolated = len(co2))

		#####################
		### STATS and CSV ###
		#####################
		name_list.append(name)
		corr_rvt_petco2_list.append(pearsonr(rvt_upsampled,petco2_upsampled)[0])
		corr_resp_co2_list.append(-corr)
		delay_list.append(delay_secs)
		
		if save_data:
			pd.DataFrame(petco2_upsampled).to_csv(os.path.join(petco2_dir, name + "_petco2.csv"), index=False,header=False)
			pd.DataFrame(rvt_upsampled).to_csv(os.path.join(rvt_dir, name + "_rvt.csv"), index=False,header=False)       
			pd.DataFrame(co2).to_csv(os.path.join(co2_dir, name + "_co2.csv"), index=False,header=False)
			pd.DataFrame(resp).to_csv(os.path.join(resp_dir, name + "_resp.csv"), index=False,header=False)

	if save_stats:
		logging.info('saving stats')
		corr_df = pd.DataFrame()
		corr_df['name'] = name_list
		corr_df['resp_co2'] = corr_resp_co2_list
		corr_df['rvt_petco2'] = corr_rvt_petco2_list
		corr_df['delay in secs between RESP and CO2'] = delay_list
		corr_df.to_csv(os.path.join(stats_dir, "correlation_stats.csv"), index =True)

	logging.info('trend_rejection: %d', trend_reject)
	logging.info('less_corr_rejection: %d', less_corr_reject)
	logging.info('less_length_rejection: %d', less_length_reject)
	logging.info('Total number of recordings selected: %d', len(name_list))
	#finding unique subjects
	unique_subjects_list = set([i.split('-')[0] for i in name_list])
	logging.info('Total number of unique subjects selected: %d', len(unique_subjects_list))
	logging.info('--------COMPLETED!!!!!----------')


if __name__=='__main__':
	make_output_dirs(output_dir)
	logging.basicConfig(filename=log_pth, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
	main()
