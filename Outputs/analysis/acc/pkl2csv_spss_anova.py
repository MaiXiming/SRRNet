"""
for significant test
"""

import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
from utils import *

path_file = os.path.abspath(__file__)
path_folder = os.path.dirname(path_file)

class_num = 40
datasets = ['benchmark', 'beta']
unseens = [8, 20, 32]

methods = ['srrv2', 'reg', 'igzsl', 'sst','trca',] # tlCCA fbcca no need, acc too low, significant for sure
# subjects = {'benchmark':35, 'beta': 70}
windows = np.linspace(0.4, 1.2, 5)

def main(dataset, unseen):
    subjects = 35 if dataset=='benchmark' else 70
    subjects = list(range(1, subjects+1))

    subject_num = len(subjects)
    acc3d, itr3d = np.zeros((subject_num, len(methods), len(windows))), np.zeros((subject_num, len(methods), len(windows)))
    for mii in range(len(methods)):
        method = methods[mii]
        times, accs, itrs = get_results_pkl(method, dataset, unseen)
        assert len(times) == len(windows)
        acc3d[:,mii,:] = np.transpose(accs)
        itr3d[:, mii, :] = np.transpose(itrs)

    # stat_anal(acc3d)
    ## Output results
    output_csv(acc3d, subjects, 'acc')
    output_csv(itr3d, subjects, 'itr')
    
def output_csv(data, subjects, label='acc'):
    flattened_data = []

    # for subject in range(subjects):
    for sii, subject in enumerate(subjects):
        subject_data = {}
        for tii, time_window in enumerate(windows):
            for mii, method in enumerate(methods):
                column_name = f"{method}{time_window}"
                subject_data[column_name] = data[sii, mii, tii]
        flattened_data.append(subject_data)

    df = pd.DataFrame(flattened_data)

    # Add subject identifiers
    df.insert(0, 'Subject', subjects)

    # Save DataFrame to CSV
    ensure_path(os.path.join(path_folder, 'csv-spss'))
    csv_filename = os.path.join(path_folder, 'csv-spss/'+f'{dataset}_u{unseen}_{label}.csv')
    df.to_csv(csv_filename, index=False)

    print(f"Data has been saved to {csv_filename}")


def stat_anal(results):
    # Create a dataframe to hold the data
    data = []
    for sii, subject in enumerate(subjects):
        for mii, method in enumerate(methods):
            for tii, time_window in enumerate(windows):
                # result = np.random.rand()  # Replace with actual data
                result = results[sii, mii, tii]
                data.append([str(subject), str(method), str(time_window), result])

    df = pd.DataFrame(data, columns=['Subject', 'Method', 'TimeWindow', 'Result'])

    # Perform two-way repeated measures ANOVA
    aovrm = AnovaRM(df, 'Result', 'Subject', within=['Method', 'TimeWindow'])
    res = aovrm.fit()
    print(res)

    ## post hoc test
    df['Method_TimeWindow'] = df['Method'] + "_" + df['TimeWindow']
    tukey_interaction = pairwise_tukeyhsd(endog=df['Result'], groups=df['Method_TimeWindow'], alpha=0.05)
    print("\nPost Hoc Test for Interaction (Method_TimeWindow):")
    print(tukey_interaction)


def get_results_pkl(method, dataset, unseen):
    # pth = 'acc-pkls/'+method+'.pkl'
    pth = os.path.join(path_folder, 'acc-pkls/'+method+'.pkl')
    with open(pth, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    times = data['times']
    unseens = data['unseens']
    us_idx = unseens.index(unseen)
    accs = data['results'][dataset][us_idx, :, :] # t*subjs
    itrs = np.zeros_like(accs)
    for tii in range(itrs.shape[0]):
        for sii in range(itrs.shape[1]):
            time = times[tii]
            itrs[tii,sii] = cal_itr(accs[tii, sii], time)
    # acc_mean = np.mean(accs, axis=-1)
    # itr_mean = np.mean
    # sem = np.std(accs, axis=-1) / (subjs[dataset]**0.5)
    return times, accs, itrs

def cal_itr(acc, time, gst=0.5):
    if acc == 1:
        return 60 * (np.log2(class_num)+acc*np.log2(acc)) / (gst+time)
    elif acc < (1/class_num):
        return 0
    else:
        return 60 * (np.log2(class_num)+acc*np.log2(acc)+(1-acc)*np.log2((1-acc)/(class_num-1))) / (gst+time)

if __name__ =='__main__':
    for dataset in datasets:
        for unseen in unseens:
            main(dataset, unseen)
    # main()