import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('WebAgg')
import pickle

## Figure settings
colors = [[0,0,0], [191,29,45], [1,138,103], [24, 104, 178], [243, 163, 50]]
markers = ['.', 'o', 's', '^', 'v'] # ['o', 'v', '^', '<', '>', '1', '2', '3']
methods = ['reg', 'igzsl', 'sst', 'tlcca', 'fbcca']
proposed = 'reg'

## Condition setting
# unseen2Nbs = {
#     'benchmark': {8:int(5*(40-8)/40), 20: int(5*(40-20)/40), 32: int(5*(40-32)/40)},
#     'beta': {8:int(3*(40-8)/40), 20: int(3*(40-20)/40), 32: 1},
#     }
subjs = {'benchmark': 35, 'beta': 70}
datasets = ['benchmark', 'beta']
unseens = [8, 20, 32]
class_num = 40



def main(dataset='benchmark', unseen=8):

    condition = 'Acc'
    plt.figure(figsize=(5,5))
    for mii in range(len(methods)):
        method = methods[mii]
        times, accs, itrs = get_results_pkl(method, dataset, unseen)
        mmean, ssem = cal_stat(accs)

        color = np.array(colors[mii]) / 255
        marker = markers[mii]
        line = '-'
        label = method+'(proposed)' if method==proposed else method

        plt.errorbar(times, mmean, yerr=ssem, marker=marker, linestyle=line, color=color, label=label)
        # plt.legend(loc='lower right')
        # plt.xlabel('Time (s)')
        # plt.ylabel(condition)
        plt.ylim([0,1])
        plt.xlim([0.3, 1.3])
        plt.xticks([0.4,0.6,0.8,1.0,1.2], labels=[])
        plt.yticks([0.25, 0.5, 0.75], labels=[])
    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    plt.show()
    # ensure_path('figs')
    plt.savefig('Analysis/Acc/figs/' + dataset+'-u' + str(unseen) + '_'+condition+'.svg')
    plt.savefig('Analysis/Acc/figs/' + dataset+'-u' + str(unseen) + '_'+condition+'.png')


    condition = 'ITR'
    plt.figure(figsize=(5,5))
    for mii in range(len(methods)):
        method = methods[mii]
        times, accs, itrs = get_results_pkl(method, dataset, unseen)
        mmean, ssem = cal_stat(itrs)

        color = np.array(colors[mii]) / 255
        marker = markers[mii]
        line = '-'
        label = method+'(proposed)' if method==proposed else method

        plt.errorbar(times, mmean, yerr=ssem, marker=marker, linestyle=line, color=color, label=label)
        # plt.xlabel('Time (s)')
        # plt.ylabel(condition)
        # plt.legend(loc='lower right')
        plt.ylim([0,250])
        plt.xlim([0.3, 1.3])
        plt.xticks([0.4,0.6,0.8,1.0,1.2], labels=[])
        plt.yticks([100, 200], labels=[])
        print(dataset, unseen, 'ITR', method, mmean)
    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    plt.show()
    # ensure_path('figs')
    plt.savefig('Analysis/Acc/figs/' + dataset+'-u' + str(unseen) + '_'+condition+'.svg')
    plt.savefig('Analysis/Acc/figs/' + dataset+'-u' + str(unseen) + '_'+condition+'.png')



# def get_accs(method, dataset, unseen):
#     if method[-5:] == 'paper':
#         times, acc_mean, sem = get_accs_paper(method, dataset, unseen)
#     elif method in ['TRCA', 'TDCA']:
#         times, acc_mean, sem = get_accs_trad(method, dataset, unseen)
#     else:
#         times, acc_mean, sem = get_results_pkl(method, dataset, unseen)
#     return times, acc_mean, sem

# def get_accs_trad(method, dataset, unseen):
#     Nb = unseen2Nbs[dataset][unseen]
#     pth = 'Results/Performances/zz-pkls/'+method+'.pkl'
#     with open(pth, 'rb') as pickle_file:
#         data = pickle.load(pickle_file)
#     times = data['times']
#     trainblocks = data['unseens']
#     us_idx = trainblocks.index(Nb)
#     accs = data['results'][dataset][us_idx, :, :]
#     acc_mean = np.mean(accs, axis=-1)
#     sem = np.std(accs, axis=-1) / (subjs[dataset]**0.5)
#     return times, acc_mean, sem

# def get_accs_paper(method, dataset, unseen):
#     pth = 'acc-pkls/'+method+'.pkl'
#     with open(pth, 'rb') as pickle_file:
#         data = pickle.load(pickle_file)
#     times = data['times']
#     unseens = data['unseens']
#     us_idx = unseens.index(unseen)
#     accs = data['results'][dataset][us_idx, :]
#     # acc_mean = np.mean(accs, axis=-1)
#     # sem = np.std(accs, axis=-1) / (subjs[dataset]**0.5)
#     sem = np.ones(len(times)) * 0.01
#     return times, accs, sem


def get_results_pkl(method, dataset, unseen):
    # pth = 'acc-pkls/'+method+'.pkl'
    pth = 'Analysis/Acc/acc-pkls/'+method+'.pkl'
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

def cal_stat(metrics):
    m_mean = np.mean(metrics, axis=-1)
    m_sem = np.std(metrics, axis=-1) / (subjs[dataset]**0.5)
    return m_mean, m_sem

def cal_itr(acc, time, gst=0.5):
    if acc == 1:
        return 60 * (np.log2(class_num)+acc*np.log2(acc)) / (gst+time)
    elif acc < (1/class_num):
        return 0
    else:
        return 60 * (np.log2(class_num)+acc*np.log2(acc)+(1-acc)*np.log2((1-acc)/(class_num-1))) / (gst+time)


if __name__ == '__main__':
    for dataset in datasets:
        for unseen in unseens:
            main(dataset, unseen)

# benchmark_u8 = {
#     'igzsl': {'t': [0.6, 0.8, 1.0],
#               'acc': [0.6996,0.8369,0.8998],
#               'std': np.array([0.1998,0.1641,0.1243])/35**0.5},
#     'igzsl-paper': {'t': [0.6, 0.8, 1.0],
#                     'acc': [0.89639,0.924188,0.941877],
#                     'std': [0.01, 0.01, 0.01, ]},
#             }


# plt.figure(figsize=(5,5))
# for method in benchmark_u8.keys():
#     time = benchmark_u8[method]['t']
#     acc = benchmark_u8[method]['acc']
#     std = benchmark_u8[method]['std']

#     plt.errorbar(time, acc, yerr=std, fmt='-o', label=method)

#     plt.xlabel('Time (s)')
#     plt.ylabel('Acc')
#     plt.legend()
#     plt.ylim([0,1])
#     plt.xlim([0, 1.2])

# plt.show()

# ensure_path('figs/Accs')
# plt.savefig('figs/Accs/benchmark_u8.svg')