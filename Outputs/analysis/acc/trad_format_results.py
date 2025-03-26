import os
import numpy as np
import pickle
import csv
import sys
sys.path.append('..')
from utils import *

results_path = '../../Performances/'
method_folder = 'TRCA' # TRCA TDCA
method = 'trca' # trca tdca
subjs = {'benchmark': 35, 'beta': 70}

datasets = ['benchmark', 'beta']
# unseens = [8, 32]
trainblocks = [1,2,3,4,5]
times = [0.6, 0.8, 1.0]

def main():
    
    results = {}
    for dataset in datasets:
        csv_filenames = get_csv_filenames(results_path+method_folder)
        subjects = subjs[dataset]
        
        accs_3d = np.zeros((len(trainblocks), len(times), subjects))
        for uii in range(len(trainblocks)):
            for tii in range(len(times)):
                Nb, time = trainblocks[uii], times[tii]
                is_Nb_exceed = Nb > 3 and dataset == 'beta'
                is_tdca_nb_1 = Nb == 1 and method == 'tdca'
                if is_Nb_exceed or is_tdca_nb_1:
                    continue
                target_suffix = dataset + '-' + method + '-t' + str(time) + 's-Nb' + str(Nb)
                accs = extract_target_condition_acc(csv_filenames, target_suffix, subjects)
                accs_3d[uii,tii,:] = accs
        results[dataset] = accs_3d

    # dict2pickle(dict, path=method_folder+'.pkl')
    dict_pkl = {'results':results, 'datasets':datasets, 
                'unseens': trainblocks, 'times': times}
    ensure_path('acc-pkls/')
    with open('acc-pkls/'+method_folder+'.pkl', 'wb') as pickle_file:
        pickle.dump(dict_pkl, pickle_file)
    
                

def extract_target_condition_acc(csv_filenames, target_suffix, subjects):
    target_filename = ''
    for filename in csv_filenames:
        if filename[:len(target_suffix)] == target_suffix:
            target_filename = filename
            break
    if target_filename == '':

        raise ValueError(f'target_suffix {target_suffix} not found!')
    else:
        accs = load_subjects_acc(results_path+method_folder+'/'+target_filename, subjects)
        return accs

def load_subjects_acc(csv_path, subjects):
    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    accs = np.zeros((subjects))
    for sii in range(subjects):
        accs[sii] = data[sii][0]

    return accs

def get_csv_filenames(folder_path):
    # 获取文件夹中的所有文件和目录名
    entries = os.listdir(folder_path)

    # 筛选出所有以.csv结尾的文件
    csv_files = [file for file in entries if file.endswith('.csv')]

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