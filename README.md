# RESPCO

This repository implements deep learning models to reconstruct CO2 and PETCO2 data from respiration waveforms in the resting state. 

### 1. Dependencies
Please install essential dependencies.

```
jupyter==1.0.0
notebook==5.7.8
numpy==1.16.2
scipy==1.2.1
pandas==0.24.2
torch==1.0.0
torchvision==0.2.2
tqdm==4.31.1
peakutils==1.3.2
seaborn==0.11.1
```


### 2. Folder Structure

> ./code consists : all source codes

> > ./code/notebooks : jupyter notebooks for training and testing

> ./data : all datasets

> > ./data/DATASET_NAME : directory for a particular dataset (DATASET_NAME)

> > > ./data/DATASET_NAME/acq : place all raw data files (in .acq format) inside this folder




### 3. Pipeline

1. Preprocess the raw files using [code/preprocess_data.py](code/preprocess_data.py), this exports processed CO2 and Resp files in .csv format
2. Generate the splits for Test and Train using [code/generate_splits.py](code/generate_splits.py)
3. Train the DL models using [code/notebooks/train.ipynb](code/notebooks/train.ipynb)
4. Test the DL models using [code/notebooks/test_qualitative.ipynb](code/notebooks/test_quantitative.ipynb) and [code/notebooks/test_quantitative.ipynb](code/notebooks/test_quantitative.ipynb)


### 4. Data pre-processing 

Please note depending on your raw dataset format, you might need to edit [code/preprocess_utils.py](code/preprocess_utils.py). Current code supports .acq format.

1. Put all the raw files inside ./data/DATASET_NAME/acq folder. The naming format supported by the code is {SUB_NAME}-{SCAN_NUM}.acq, where SUB_NAME is the subject ID, and SCAN_NUM is the scan number for that particular subject.
1. In [code/preprocess_data.py](code/preprocess_data.py), set root_dir to dataset directory path (data/DATASET_NAME) 
2. Executing [code/preprocess_data.py](code/preprocess_data.py) will create folders containing preprocessed data inside data/DATASET_NAME directory


### 5. Generating Splits

1. Given dataset directory, [code/generate_splits.py](code/generate_splits.py) generates k-fold splits 
2. It writes the output to .txt files, which contains the list of the path of preprocessed files

### 6. Training and Testing

1. The pytorch models are provided in [code/models_new.py](code/models_new.py)
2. Dataset loader code is provided in [code/datasets.py](code/datasets.py)
2. For main training and testing code, please refer to [code/notebooks](code/notebooks)

### 7. Publication

Manusript is under review.