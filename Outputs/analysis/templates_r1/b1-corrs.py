
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_corrcoef(arr1, arr2):
    """
    计算两个一维 numpy 数组的相关系数
    
    参数：
    - arr1: 第一组数据 (一维数组或形状为 (1, Nt))
    - arr2: 第二组数据 (一维数组或形状为 (1, Nt))
    
    返回：
    - 两个数组的相关系数（标量）
    """
    # 如果数组是 (1, Nt) 的形状，则压缩成一维数组
    arr1 = np.squeeze(arr1)
    arr2 = np.squeeze(arr2)
    
    # 利用 np.corrcoef 计算相关系数矩阵，取[0, 1]处的值
    corr_matrix = np.corrcoef(arr1, arr2)
    return corr_matrix[0, 1]

def compute_mse(arr1, arr2):
    """
    计算两个一维 numpy 数组的均方误差（MSE）。

    参数：
    - arr1: 第一组数据（形状为 (1, Nt) 或 (Nt,)）
    - arr2: 第二组数据（形状为 (1, Nt) 或 (Nt,)）

    返回：
    - 均方误差（标量）
    """
    # 如果输入数组是 (1, Nt) 的形状，则压缩成一维数组
    arr1 = np.squeeze(arr1)
    arr2 = np.squeeze(arr2)
    
    # 检查两个数组长度是否一致
    if arr1.shape != arr2.shape:
        raise ValueError("输入数组的形状不一致，无法计算 MSE。")
    
    # 计算均方误差
    mse = np.mean((arr1 - arr2) ** 2)
    return mse

def get_corrs(dataset, unseen, window, channel_idx=-2, fbii=0, mean_condition='unseen'):
    """
    Inputs:
        mean_condition: all == mean all classes; unseen == mean only on unseen classes (args.label_unseen)
    Returns:
        {corrs, accs, corr_detail}
        corr_detail: (unseen, subjects, ) # u1, u2, ..., u8 each has a correlation
    """
    subjects = 35 if dataset=='benchmark' else 70
    # corrs = np.zeros((subjects, unseen, ))
    corrs_2d = np.array([])
    accs = np.zeros((subjects, ))

    for subject in range(subjects):
        ## Load data
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)

        filename = f"Details/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
        # filename = f"Details-unseen/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
        filename = os.path.join(folder_path, filename)
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        labels = np.arange(40) if mean_condition=='all' else np.array(data['args'].label_unseen)

        # for ui in range(unseen):
        corrs = np.zeros((labels.shape[0], ))
        for idx, ui in enumerate(labels):

            # all class templates (Details)
            recon_t = data['recon_templates'][ui, fbii, channel_idx, :]
            true_t = data['true_templates'][ui, fbii, channel_idx, :]
            
            # unseen class templates (Details-unseen)
            # recon_t = data['recon_templates'][idx, fbii, channel_idx, :]
            # true_t = data['true_templates'][idx, fbii, channel_idx, :]

            corr = compute_corrcoef(recon_t, true_t)
            corrs[idx] = corr

        corrs_2d = np.concatenate((corrs_2d, corrs))
        accs[subject] = data['acc']
    
    corrs_2d = np.reshape(corrs_2d, (labels.shape[0], subjects))
    corrs = np.mean(corrs_2d, 0)
        
    return {'corrs': corrs, 
            'accs': accs,
            'corr_2d': corrs_2d,
    }
        



if __name__ == '__main__':
    # ## Single conditions
    # data = get_corrs('benchmark', 32, 1) # Oz indeed highest
    # print('Corrs: ', data['corrs'])
    # print('Accs: ', data['accs'])
    # # print('Corr detail (unseen*subjects): ', data['corr_detail'])
    # corrs, accs = data['corrs'], data['accs']

    ## All
    mean_condition = 'unseen' # all unseen
    corrs, accs = np.array([]), np.array([])
    for dataset in ['benchmark', 'beta']:
        for unseen in [8, 20, 32]:
        # for unseen in [8]:
            data = get_corrs(dataset, unseen, 1, mean_condition=mean_condition)
            corrs = np.concatenate((corrs, data['corrs']))
            accs = np.concatenate((accs, data['accs']))
            print(f"Unseen={unseen}", "Mean Corr: ", np.mean(corrs))

    
    plt.scatter(corrs, accs, color='blue', marker='o')
    plt.xlabel("Corrs")
    plt.ylabel("Accs)")
    plt.title("")
    # plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)  # 添加网格便于观察数据分布

    plt.show()



    # # for subject in range(35):
    # dataset = 'benchmark'
    # unseen = 32
    # window = 1.0
    # # subject = 31

    # file_path = os.path.abspath(__file__)
    # folder_path = os.path.dirname(file_path)

    # filename = f"Details/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
    # filename = os.path.join(folder_path, filename)
    # # filename = os.path.join('/data3/mxm/Code/gzsl_reg_simple/Outputs/SaveForAnalysis/Templates-R1/', filename)
    # with open(filename, 'rb') as file:
    #     data = pickle.load(file)

    # # print(data)

    
    # ## Pearson correlation
    # for fbii in range(5):
    #     # fbii = 0
    #     unseen_idx = 0
    #     channel_idx = -2 # Oz

    #     recon_templates = data['recon_templates'][unseen_idx, fbii, channel_idx, :]
    #     true_templates = data['true_templates'][unseen_idx, fbii, channel_idx, :]
    #     print(compute_corrcoef(recon_templates, true_templates), end=' ')
    
    # print()