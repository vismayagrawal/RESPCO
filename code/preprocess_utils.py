import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

def parse_co2_resp_from_acq(acq_all):
    """
    Input: list containing location of all the acq files
    Output: pandas dataframe
            column - name, Fs, raw_co2, raw_resp
            rows - data of different files
    """
    import bioread
    name_list = []
    Fs_list = []
    raw_co2_list = []
    raw_resp_list = []
    logging.info('--------checking acq files which contains both CO2 and RESP-----------')
    for acq in tqdm(acq_all):
        data = bioread.read_file(acq)
        Fs  = int(data.samples_per_second) # sampling freq
        length_data = int(data.time_index[-1] * Fs)

        co2_status  = False
        resp_status = False
        for chan in data.channels:
            if "CO2100C" in chan.name:
                raw_co2 = chan.data #list of all co2 data points
                co2_status = True
            elif "RSP100C" in chan.name:
                raw_resp = chan.data #list of all resp data points
                resp_status = True

        if co2_status and resp_status:
            name_list.append(acq.split('/')[-1])
            Fs_list.append(Fs)
            raw_co2_list.append(raw_co2)
            raw_resp_list.append(raw_resp)

    return name_list, Fs_list, raw_co2_list, raw_resp_list 

def parse_co2_resp(raw_dir, data_name):
    if data_name == 'baycrest_biopac':
        logging.info("Baycrest biopac data (raw_physio_backup-biopac-20180417) selected")
        file_paths = sorted(glob.glob(os.path.join(raw_dir, '*.acq'), recursive=False))
        name_list, Fs_list, raw_co2_list, raw_resp_list = parse_co2_resp_from_acq(file_paths)

    else:
        raise Exception('Data Not in the list')

    df = pd.DataFrame()
    df['name'] = name_list
    df['Fs'] = Fs_list
    df['raw_co2'] = raw_co2_list
    df['raw_resp'] = raw_resp_list
    logging.info('%d files found', len(df.index))
    return df