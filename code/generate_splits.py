"""
Given input root directory it generates the k folds splits and save the locations in form of txt

"""
import os
import numpy as np
import glob

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils import write_txt

# User input
# Note": root_dir shouldn't contain 'co2' or 'resp' as string in it, the code will fail, because it replaces resp with co2 in loc in 'get_loc_from_indexes' function
root_dir = os.path.abspath('../data/raw_physio_backup-biopac-20180417/preprocessed') 
split_dir = os.path.join(root_dir, 'splits_train_test')
subject_separater = '-' #the character which separate the subject name from scan number
number_of_splits = 5 #We cant keep it more than the number of subject itself
include_val_split = False

def split_subjects(resp_paths, subject_separater = '-'):
	"""
	Separates the data according to different subjects
	parameters
		resp_paths: sorted list containing all the respiration data paths
		subject_separater: the character which separate the subject name
	returns
		list containing the splits of the subjects
		e.g. [array_subject1([loc_s1_1, loc_s1_2, loc_s1_3...]),
				array_subject2([loc_s2_1, loc_s2_2, loc_s2_3...]),
				...]
	"""
	split_index = []
	for i in range(len(resp_paths[:-1])):
		if resp_paths[i].split('/')[-1].split(subject_separater)[0] != resp_paths[i+1].split('/')[-1].split(subject_separater)[0]:
			split_index.append(i+1)
	return np.split(resp_paths, split_index)

def get_loc_from_indexes(subjects, indexes):
	"""
	Gets location of all the scans for multiple subjects
	parameters
		subjects: output of funtion split subjects
				list containing the splits of the subjects
				e.g. [array_subject1([loc_s1_1, loc_s1_2, loc_s1_3...]),
					array_subject2([loc_s2_1, loc_s2_2, loc_s2_3...]),
					...]
		indexes: all the index

	"""
	resp_paths = []
	co2_paths = []
	for index in indexes:
		for in_loc in subjects[index]:
			resp_paths.append(in_loc)
			co2_paths.append(in_loc.replace('resp','co2'))
	return resp_paths, co2_paths

def save_txt(i, paths, file_name):
	os.makedirs(os.path.join(split_dir, str(i)), exist_ok = True)
	write_txt(os.path.join(split_dir, str(i), file_name), paths)

def generate_splits_train_val_test(root_dir, split_dir, number_of_splits, subject_separater='-'):
	resp_paths = sorted(glob.glob(os.path.join(root_dir, 'resp/*.csv')))
	co2_paths = sorted(glob.glob(os.path.join(root_dir, 'co2/*.csv')))
	assert(len(resp_paths)==len(co2_paths))
	subjects = split_subjects(resp_paths, subject_separater = subject_separater)

	kf = KFold(n_splits=number_of_splits)
	for i, splits in enumerate(kf.split(subjects)):
		train_val_indexes = splits[0]
		test_indexes = splits[1]
		train_indexes, val_indexes = train_test_split(train_val_indexes, test_size = 0.1, random_state=24)

		resp_train_paths, co2_train_paths = get_loc_from_indexes(subjects, train_indexes)
		resp_val_paths, co2_val_paths = get_loc_from_indexes(subjects, val_indexes)
		resp_test_paths, co2_test_paths = get_loc_from_indexes(subjects, test_indexes)

		save_txt(i, resp_train_paths, 'resp_train.txt')
		save_txt(i, resp_val_paths, 'resp_val.txt')
		save_txt(i, resp_test_paths, 'resp_test.txt')

		save_txt(i, co2_train_paths, 'co2_train.txt')
		save_txt(i, co2_val_paths, 'co2_val.txt')
		save_txt(i, co2_test_paths, 'co2_test.txt')

	write_txt(os.path.join(split_dir, 'resp_all_filenames.txt'), resp_paths) # saving a list containing all the filesnames
	write_txt(os.path.join(split_dir, 'co2_all_filenames.txt'), co2_paths) # saving a list containing all the filesnames
	print("COMPLETED!!!")

def generate_splits_train_test(root_dir, split_dir, number_of_splits, subject_separater='-'):
	resp_paths = sorted(glob.glob(os.path.join(root_dir, 'resp/*.csv')))
	co2_paths = sorted(glob.glob(os.path.join(root_dir, 'co2/*.csv')))
	assert(len(resp_paths)==len(co2_paths))
	subjects = split_subjects(resp_paths, subject_separater = subject_separater)
	print(f'Number of subjects: {len(subjects)}')
	kf = KFold(n_splits=number_of_splits)
	for i, splits in enumerate(kf.split(subjects)):
		train_indexes = splits[0]
		test_indexes = splits[1]

		resp_train_paths, co2_train_paths = get_loc_from_indexes(subjects, train_indexes)
		resp_test_paths, co2_test_paths = get_loc_from_indexes(subjects, test_indexes)

		save_txt(i, resp_train_paths, 'resp_train.txt')
		save_txt(i, resp_test_paths, 'resp_test.txt')

		save_txt(i, co2_train_paths, 'co2_train.txt')
		save_txt(i, co2_test_paths, 'co2_test.txt')

		print(f'fold({i}) --- train_filenumbers = {len(resp_train_paths)}, test_filenumbers = {len(resp_test_paths)}')

	write_txt(os.path.join(split_dir, 'resp_all_filenames.txt'), resp_paths) # saving a list containing all the filesnames
	write_txt(os.path.join(split_dir, 'co2_all_filenames.txt'), co2_paths) # saving a list containing all the filesnames
	print("COMPLETED!!!")

def main():
	if include_val_split:
		generate_splits_train_val_test(root_dir, split_dir, number_of_splits,subject_separater)
	else:
		generate_splits_train_test(root_dir, split_dir, number_of_splits,subject_separater)

if __name__ == '__main__':
	main()
