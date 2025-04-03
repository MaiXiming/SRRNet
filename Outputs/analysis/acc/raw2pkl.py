"""
Run from base repo
Raw data: Statistics_raw
Functionality: raw data (.csv / .mat) --> .pickle for plot & CSV (significant test)
"""

import os
import numpy as np
import pickle
import csv
import sys 
sys.path.append("..") 
# from utils import *
import scipy.io
import h5py

path_file_current = os.path.abspath(__file__)
path_folder_current = os.path.dirname(path_file_current)

## Input
method_folder = 'srrnet' # reg igzsl sst tlcca fbcca trca srrv2 srrnet

## Fixed Params
subjs = {'benchmark': 35, 'beta': 70}
datasets = ['benchmark', 'beta']
unseens = [8, 20, 32]
times = [0.4, 0.6, 0.8, 1.0, 1.2]
results_path = os.path.join(path_folder_current, '../data/accs_raw/') #'Analysis/Acc/Statistics_raw/'

method = 'rescnn_lstm' if method_folder == 'reg' else method_folder
suffix = '.mat' if method in ['sst', 'tlcca', 'fbcca'] else '.csv' # .csv .mat
if method_folder == 'reg' or method_folder == 'srrv2' or method_folder == 'srrnet':
    prep = '-trca' # -trca -tdca -none
else:
    prep = ''
# note: '-' is necessary


def main():
    results = {}
    for dataset in datasets:
        ## Get raw data 
        subjects = subjs[dataset]
        csv_filenames = get_files_in_folder(results_path + method_folder, suffix=suffix)
        ## Format accs into np array
        accs_3d = np.zeros((len(unseens), len(times), subjects))
        for uii in range(len(unseens)):
            for tii in range(len(times)):
                unseen, time = unseens[uii], times[tii]
                condition = dataset + '-u' + str(unseen) + '-t' + str(time) + 's' + prep + '-' + method
                accs = load_condition_acc(csv_filenames, condition, subjects)
                accs_3d[uii,tii,:] = accs
        results[dataset] = accs_3d

    ## Save accs into .pickle
    dict_pkl = {'results':results, 'datasets':datasets, 
                'unseens': unseens, 'times': times}
    # with open('Analysis/Acc/acc-pkls/'+method_folder+'.pkl', 'wb') as pickle_file: # from base
    with open(os.path.join(path_folder_current,'acc-pkls/'+method_folder+'.pkl'), 'wb') as pickle_file: # from base
        pickle.dump(dict_pkl, pickle_file)
    
                
## Functions

def load_condition_acc(csv_filenames, condition, subjects):
    target_filename = None
    for filename in csv_filenames:
        if filename[:len(condition)] == condition:
            target_filename = filename
            break
    if target_filename is None:
        raise ValueError('condition '+condition+'not found!')
    else:
        if target_filename[-4:] == '.mat':
            accs = load_acc_mat(results_path+method_folder+'/'+target_filename, subjects)
        else:
            accs = load_subjects_acc(results_path+method_folder+'/'+target_filename, subjects)
        return accs


# def load_acc_mat(filepath, subjects):
#     contents = scipy.io.loadmat(filepath)
#     accs = contents['accs']
#     assert accs.shape[0] == subjects
#     return accs

def load_acc_mat(filepath, subjects):
    with h5py.File(filepath, 'r') as file:
        accs = file['accs'][:]
        accs = np.squeeze(accs)

    assert accs.shape[0] == subjects
    return accs


def load_subjects_acc(csv_path, subjects):
    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    accs = np.zeros((subjects))
    for sii in range(subjects):
        accs[sii] = data[sii][0]

    return accs

def get_files_in_folder(folder_path, suffix='.csv'):
    # 获取文件夹中的所有文件和目录名
    entries = os.listdir(folder_path)

    # 筛选出所有以.csv结尾的文件
    csv_files = [file for file in entries if file.endswith(suffix)]

    return csv_files

def dict2pickle(dict, path=method+'.pkl'):
    with open(path, 'wb') as pickle_file:
        pickle.dump(dict, pickle_file)

def load_pickle(path):
    with open(path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data

if __name__ == '__main__':
    main()