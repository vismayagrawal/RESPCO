# Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging

## Abstract

Introduction: In the context of functional magnetic resonance imaging (fMRI), carbon dioxide (CO2) is a well-known vasodilator that has been widely used to monitor and interrogate vascular physiology. Moreover, spontaneous fluctuations in end-tidal carbon dioxide (PETCO2) reflects changes in arterial CO2 and has been demonstrated as the largest physiological noise source for denoising the low-frequency range of the resting-state fMRI (rs-fMRI) signal. However, the majority of rs-fMRI studies do not involve CO2 recordings, and most often only heart rate and respiration are recorded. While the intrinsic link between these latter metrics and CO2 led to suggested possible analytical models, they have not been widely applied.

Methods: In this proof-of-concept study, we propose a deep-learning (DL) approach to reconstruct CO2 and PETCO2 data from respiration waveforms in the resting state.

Results: We demonstrate that the one-to-one mapping between respiration and CO2 recordings can be well predicted using fully convolutional networks (FCNs), achieving a Pearson correlation coefficient (r) of 0.946 ± 0.056 with the ground truth CO2. Moreover, dynamic PETCO2 can be successfully derived from the predicted CO2, achieving r of 0.512 ± 0.269 with the ground truth. Importantly, the FCN-based methods outperform previously proposed analytical methods. In addition, we provide guidelines for quality assurance of respiration recordings for the purposes of CO2 prediction.

Discussion: Our results demonstrate that dynamic CO2 can be obtained from respiration-volume using neural networks, complementing the still few reports in DL of physiological fMRI signals, and paving the way for further research in DL based bio-signal processing.

## How to Use this Repository

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

## Citation

If you use some of our work, please cite our paper:

Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539

Alternatively, use the following Bibtex code:
~~~
@ARTICLE{10.3389/fnimg.2023.1119539,
AUTHOR={Agrawal, Vismay  and Zhong, Xiaole Z.  and Chen, J. Jean },
TITLE={Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging},
JOURNAL={Frontiers in Neuroimaging},
VOLUME={2},
YEAR={2023},
URL={https://www.frontiersin.org/journals/neuroimaging/articles/10.3389/fnimg.2023.1119539},
DOI={10.3389/fnimg.2023.1119539},
ISSN={2813-1193},
}
~~~

## Contact

* Vismay Agrawal (vismay.iitm@gmail.com)
