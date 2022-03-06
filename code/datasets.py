"""
Functions to load data during training or testing
"""
import os
import numpy as np
import glob
import pandas as pd
import utils
# from torch.utils.data.dataset import Dataset

def read_txt(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def load_csv(loc):
    """ 
    loads the input csv data
    loc: string with the location of csv file
    """
    return np.squeeze(pd.read_csv(loc, header = None).values)

def preprocess_resp(resp):
    resp = utils.remove_outlier(resp, z_threshold = 3)
    resp = utils.std_normalise(resp)
    return resp

class resp_co2_dataset():
    def __init__(self, txt_dir, 					
                    resp_filename = 'resp_train.txt', 
                    co2_filename = 'co2_train.txt',
                    Fs = 10,
                    apply_co2_norm = True,
                    return_sub_name = False):
        """
        arguments
            txt_dir: root dir containing all the 6 text files with locations of data(test, val, train of co2 and resp)
        """
        self.txt_dir = txt_dir
        self.resp_list = read_txt(os.path.join(txt_dir, resp_filename))
        self.co2_list = read_txt(os.path.join(txt_dir, co2_filename))
        assert(len(self.resp_list) == len(self.co2_list))
        self.Fs = Fs
        self.apply_co2_norm = apply_co2_norm
        self.return_sub_name = return_sub_name

    def __getitem__(self, index):
        resp = load_csv(self.resp_list[index])
        resp = preprocess_resp(resp)
        co2 = load_csv(self.co2_list[index])
        if self.apply_co2_norm:
            co2 = utils.std_normalise(co2)
        assert(len(resp) == len(co2))
        if len(resp)%16 != 0: # This is for the neural network consistent input output
            extra = len(resp)%16
            resp = resp[:-extra]
            co2 = co2[:-extra]
        resp = np.reshape(resp, (1,-1)) #(channel, width)
        co2 = np.reshape(co2, (1,-1))
        if self.return_sub_name:
            name = self.resp_list[index].split('/')[-1].split('_resp.')[0]
            return resp, co2, name
        else:
            return resp, co2

    def getitem_tanh(self, index):
        resp = np.tanh(load_csv(self.resp_list[index]))
        co2 = np.tanh(load_csv(self.co2_list[index]))
        assert(len(resp) == len(co2))
        if len(resp)%16 != 0: # This is for the neural network consistent input output
            extra = len(resp)%16
            resp = resp[:-extra]
            co2 = co2[:-extra]
        resp = np.reshape(resp, (1,1,-1))
        co2 = np.reshape(co2, (1,1,-1))
        return resp, co2

    def get_resp_fileloc(self, index):
        return self.resp_list[index]

    def __len__(self):
        return len(self.resp_list)