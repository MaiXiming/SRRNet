import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
# from utils import *
import pickle
import os
import scipy

subjs_dict = {'benchmark': 35, 'beta': 70}
blks_dict = {'benchmark': 6, 'beta': 4}
class_num = 40

# dataset = 'benchmark'
# unseen = 32
datasets = ['benchmark', 'beta', ]
unseens = [8, 20, 32]
unseens_labels = {8:list(range(16, 24)), 20: list(range(10, 30)), 32: list(range(4, 36))}
# datasets = ['beta']
# unseens = [32]
t = 1.2
algo = 'reg'
prep = 'trca'
method = 'rescnn_lstm'
fb = 5

def plot_confuse(dataset='beta', unseen=32):
    subjects, blocks = subjs_dict[dataset], blks_dict[dataset]

    # filepath = f'../../Results/Records/{algo}/{dataset}-u{unseen}-t{t}s-{prep}-{method}-fb{fb}.pickle'
    filepath = f'Results/Records/{algo}/{dataset}-u{unseen}-t{t}s-{prep}-{method}-fb{fb}.pickle'
    with open(filepath, 'rb') as file:
        confusemat_subjs = pickle.load(file) # subjs, Nclass-true, Nclass-predict

    confusemat_sum = np.sum(confusemat_subjs, 0) # y==true; x==predict; val=0~subjs*blks
    confusemat_percent = confusemat_sum / (subjects*blocks)
    str_cmap = 'binary' # Spectral binary
    plt.figure(figsize=(5,5))
    plt.imshow(confusemat_percent, cmap=str_cmap, aspect='equal', vmin=0, vmax=1) # Reds viridis norm='linear', 
    # plt.colorbar()  # 添加颜色条
    plt.title('')
    # plt.xlabel('predict')
    # plt.ylabel('true')
    plt.xticks(np.arange(0, class_num, 8))
    plt.yticks(np.arange(0, class_num, 8))
    if not os.path.exists('fig_confuse'):
        os.makedirs('fig_confuse')
    # ensure_path('fig_confuse')
    plt.savefig(f'Analysis/ConfusionMatrix/fig_confuse/{dataset}-u{unseen}-t{t}s-{prep}-{method}-{str_cmap}.svg')


def output_acc_seen_unseen(dataset='beta', unseen=32):
    subjects, blocks = subjs_dict[dataset], blks_dict[dataset]
    # filepath = f'../../Results/Records/{algo}/{dataset}-u{unseen}-t{t}s-{prep}-{method}-fb{fb}.pickle'
    filepath = f'Results/Records/{algo}/{dataset}-u{unseen}-t{t}s-{prep}-{method}-fb{fb}.pickle'
    with open(filepath, 'rb') as file:
        confusemat_subjs = pickle.load(file) # subjs, Nclass-true, Nclass-predict

    accs = np.zeros((subjects, 2)) # seen & unseen
    for subj in range(subjects):
        cmat = confusemat_subjs[subj,:,:] / blocks
        if dataset == 'benchmark':
            unseen_label = unseens_labels[unseen]
        elif dataset == 'beta':
            unseen_label = unseens_labels[unseen]
            label_bm2bt = find_label_bm2bt()
            unseen_label = np.array(label_bm2bt)[unseen_label]
        else:
            raise ValueError
        
        seen_label = [label for label in range(40) if label not in unseen_label]

        acc_unseen = np.mean(cmat[unseen_label, unseen_label])
        acc_seen = np.mean(cmat[seen_label, seen_label])

        accs[subj,:] = np.array([acc_seen, acc_unseen])

    with open(f'Analysis/ConfusionMatrix/accs/{dataset}-u{unseen}-t{t}s-{prep}-{method}.pkl', 'wb') as pickle_file:
        pickle.dump(accs, pickle_file)

    ## for statistical analysis (t-test)
    ttest = scipy.stats.ttest_rel(accs[:,0], accs[:, 1])
    print(f'###{dataset}-{unseen}: Acc_seen={np.mean(accs[:, 0])}, Acc_unseen={np.mean(accs[:, 1])}.')
    print(f'ttest: t({ttest.df})={np.abs(ttest.statistic):.4f}, p={ttest.pvalue:.4f}')
    print(unseen_label, '\n')

def find_label_bm2bt():
    freqs_benchmark = [ 8. , 9. , 10. , 11. , 12. , 13. , 14. , 15. , 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
    freqs_beta = [ 8.6, 8.8, 9. , 9.2, 9.4, 9.6, 9.8, 10. , 10.2, 10.4, 10.6, 10.8, 11. , 11.2, 11.4, 11.6, 11.8, 12. , 12.2, 12.4, 12.6, 12.8, 13. , 13.2, 13.4, 13.6, 13.8, 14. , 14.2, 14.4, 14.6, 14.8, 15. , 15.2, 15.4, 15.6, 15.8, 8. , 8.2, 8.4]

    labels_bm2bt = []
    for idx_bm in range(len(freqs_benchmark)):
        freq_bm = freqs_benchmark[idx_bm]
        for idx_bt in range(len(freqs_beta)):
            freq_bt = freqs_beta[idx_bt]
            if abs(freq_bm - freq_bt) < 0.01:
                labels_bm2bt.append(idx_bt)
                break
        
    return labels_bm2bt

if __name__ == '__main__':
    for dataset in datasets:
        for unseen in unseens:
            plot_confuse(dataset, unseen)
            output_acc_seen_unseen(dataset, unseen)
    