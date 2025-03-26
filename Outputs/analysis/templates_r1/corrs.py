
import pickle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.io import savemat

## Locate current path
path_file_current = os.path.abspath(__file__)
path_folder_current = os.path.dirname(path_file_current)

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

def compute_acc_su(confusemat, labels, block_num):
    total = len(labels) * block_num
    count = 0
    for idx in labels:
        count += confusemat[idx, idx]
    return count / total

def get_data(dataset, unseen, window, subject, folder=os.path.join(path_folder_current, '../data/details-norm0')):
    """
    Returns: 
        data['recon_templates'][bii, ui, fbii, channel_idx, :]
        data['args']
        data['acc']
        data['confusemat']
    """
    # filename = os.path.join(folder, f"{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle")
    filename = f"{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
    # filename = f"Details/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
    # filename = f"Details-unseen/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
    filename = os.path.join(folder, filename)
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_results(dataset, unseen, window, channel_idx=-2, fbii=0, mean_condition='unseen', metric='corr'):
    """
    Inputs:
        mean_condition: all == mean all classes; unseen == mean only on unseen classes (args.label_unseen)
        metric: corr, mse
    Returns:
        {corrs, accs, corr_detail}
        corr_detail: (unseen, subjects, ) # u1, u2, ..., u8 each has a correlation
    """
    subjects = 35 if dataset=='benchmark' else 70
    corrs_nd = []
    accs = np.zeros((subjects, ))
    accs_seen = np.zeros((subjects, ))
    accs_unseen = np.zeros((subjects, ))

    for subject in range(subjects):
        ## Load data
        data = get_data(dataset, unseen, window, subject)
        # filename = f"Details/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
        # # filename = f"Details-unseen/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
        # filename = os.path.join(path_folder_current, filename)
        # with open(filename, 'rb') as file:
        #     data = pickle.load(file)

        ## Calculate correlation
        labels = np.arange(40) if mean_condition=='all' else np.array(data['args'].label_unseen)
        block_num = data['args'].block_num
        corrs = []
        for bii in range(block_num):
            for idx, ui in enumerate(labels):

                # all class templates (Details)
                recon_t = data['recon_templates'][bii, ui, fbii, channel_idx, :]
                true_t = data['true_templates'][bii, ui, fbii, channel_idx, :]
                
                # unseen class templates (Details-unseen)
                # recon_t = data['recon_templates'][idx, fbii, channel_idx, :]
                # true_t = data['true_templates'][idx, fbii, channel_idx, :]

                if metric == 'corr':
                    corr = compute_corrcoef(recon_t, true_t)
                else:
                    corr = compute_mse(recon_t, true_t)

                # corrs[bii, idx] = corr
                corrs.append(corr)

        corrs_nd.append(corrs)

        ## Acc
        accs[subject] = data['acc']
        accs_seen[subject] = compute_acc_su(data['confusemat'], data['args'].label_seen, data['args'].block_num)
        accs_unseen[subject] = compute_acc_su(data['confusemat'], data['args'].label_unseen, data['args'].block_num)
    
    # corrs_2d = np.reshape(corrs_2d, (labels.shape[0], subjects))
    # corrs = np.mean(corrs_2d, 0)
    # corrs_nd = np.stack(corrs_nd, axis=2)
    # corrs = np.mean(corrs_nd, (0, 1))
    corrs_nd = np.stack(corrs_nd, axis=1)
    corrs = np.mean(corrs_nd, 0)
        
    return {'corrs': corrs, 
            'corr_nd': corrs_nd,
            'accs': accs,
            'accs_seen': accs_seen,
            'accs_unseen': accs_unseen,
            'args': data['args'],
    }
        

def fit_linear(corrs, accs):
    # 将 corrs 变为二维列向量，以符合 sklearn 的输入格式
    X = corrs.reshape(-1, 1)
    y = accs

    # 建立并训练线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # 得到回归参数
    a_linear = linear_model.intercept_    # 截距
    b_linear = linear_model.coef_[0]      # 斜率

    # 预测
    y_pred_linear = linear_model.predict(X)

    print("线性回归模型: y = {:.4f} + {:.4f} * x".format(a_linear, b_linear))

    r2 = r2_score(y, y_pred_linear)
    mse = mean_squared_error(y, y_pred_linear)

    print("线性回归 R²: {:.4f}".format(r2))
    print("线性回归 MSE: {:.4f}".format(mse))

    return y_pred_linear, linear_model

def fit_log(corrs, accs):
    X_log = np.log(corrs).reshape(-1, 1)
    y     = accs

    # 建立线性回归模型
    log_model = LinearRegression()
    log_model.fit(X_log, y)

    # 拟合参数：截距 a, 以及系数 b
    a = log_model.intercept_
    b = log_model.coef_[0]

    print(f"对数函数关系: y = {a:.4f} + {b:.4f} * ln(x)")

    # 在训练数据上预测
    y_pred_log = log_model.predict(X_log)

    # 计算 R²、MSE
    r2  = r2_score(y, y_pred_log)
    mse = mean_squared_error(y, y_pred_log)

    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    return y_pred_log, log_model

def regression(corrs, accs):

    # # 将 corrs 变为二维列向量，以符合 sklearn 的输入格式
    # X = corrs.reshape(-1, 1)
    # y = accs

    # # 建立并训练线性回归模型
    # linear_model = LinearRegression()
    # linear_model.fit(X, y)

    # # 得到回归参数
    # a_linear = linear_model.intercept_    # 截距
    # b_linear = linear_model.coef_[0]      # 斜率

    # # 预测
    # y_pred_linear = linear_model.predict(X)

    # print("线性回归模型: y = {:.4f} + {:.4f} * x".format(a_linear, b_linear))

    # r2 = r2_score(y, y_pred_linear)
    # mse = mean_squared_error(y, y_pred_linear)

    # print("线性回归 R²: {:.4f}".format(r2))
    # print("线性回归 MSE: {:.4f}".format(mse))

    # X_log = np.log(corrs).reshape(-1, 1)
    # y     = accs

    # # 建立线性回归模型
    # log_model = LinearRegression()
    # log_model.fit(X_log, y)

    # # 拟合参数：截距 a, 以及系数 b
    # a = log_model.intercept_
    # b = log_model.coef_[0]

    # print(f"对数函数关系: y = {a:.4f} + {b:.4f} * ln(x)")

    # # 在训练数据上预测
    # y_pred_log = log_model.predict(X_log)

    # # 计算 R²、MSE
    # r2  = r2_score(y, y_pred_log)
    # mse = mean_squared_error(y, y_pred_log)

    # print(f"R²: {r2:.4f}")
    # print(f"MSE: {mse:.4f}")

    y_pred_linear, linear_model = fit_linear(corrs, accs)
    y_pred_log, log_model = fit_log(corrs, accs)

    return y_pred_linear, y_pred_log, linear_model, log_model



if __name__ == '__main__':
    # ## Single conditions
    # data = get_corrs('benchmark', 32, 1) # Oz indeed highest
    # print('Corrs: ', data['corrs'])
    # print('Accs: ', data['accs'])
    # # print('Corr detail (unseen*subjects): ', data['corr_detail'])
    # corrs, accs = data['corrs'], data['accs']

    ## All
    mean_condition = 'all' # all unseen
    metric = 'corr' # corr mse
    acc_condition = 'accs_unseen' # accs accs_unseen accs_seen for answer comment1, comment2
    datasets = ['benchmark', 'beta']
    # datasets = ['benchmark']
    # datasets = ['beta']
    unseens = [8, 20, 32]
    window = 1 # 1 0.8
    corrs, accs = np.array([]), np.array([])
    accs_all, accs_seen, accs_unseen = np.array([]), np.array([]), np.array([])
    for dataset in datasets:
        for unseen in unseens:
            data = get_results(dataset, unseen, window, mean_condition=mean_condition, metric=metric)
            corrs = np.concatenate((corrs, data['corrs']))
            # accs = np.concatenate((accs, data[acc_condition]))
            accs_all = np.concatenate((accs_all, data['accs']))
            accs_seen = np.concatenate((accs_seen, data['accs_seen']))
            accs_unseen = np.concatenate((accs_unseen, data['accs_unseen']))
            print(f"Metric={metric}, Dataset={dataset}, Unseen={unseen}", "Mean Corr: ", np.mean(data['corrs']), "STD: ", np.std(data['corrs']), 'SEM: ', np.std(data['corrs'])/data['args'].subject**0.5)

    ## Save to .mat
    data = {'corrs': corrs, 
            'accs': accs,
            }
    savemat(os.path.join(path_folder_current, 'test.mat'), data)

    if acc_condition == 'accs':
        accs = accs_all
    elif acc_condition == 'accs_unseen':
        accs = accs_unseen
    else:
        accs = accs_seen

    accs_lr, accs_exp, linear_model, log_model = regression(corrs, accs)

    ## Plot
    # matplotlib.rc('font', family='Times New Roman', size=8)
    plt.figure(figsize=(5, 3))
    xmin, xmax = 0, 1.05

    plt.scatter(corrs,
            accs,
            s=20,              # 散点大小，可根据需求调整，如 10, 20, 30等
            facecolors='none', # 不填充内部
            edgecolors='blue',# 边缘颜色
            marker='o',         # 圆形标记
            label='data',
            )        
    # plt.scatter(corrs, accs_lr, color='black', marker='.')
    # plt.scatter(corrs, accs_exp, color='red', marker='x')
    # Log y
    # X = np.linspace(np.exp(xmin), np.exp(xmax), 1000).reshape(-1, 1)
    X = np.linspace(xmin+0.001, xmax, 1000)
    X1 = np.log(X).reshape(-1, 1)
    # X = np.log(X)
    ylog = log_model.predict(X1)
    plt.plot(X, ylog, linestyle='-', color='black', label='Log fit')

    # plt.xlabel("Corrs")
    # plt.ylabel(acc_condition)
    plt.title("")
    plt.xlim([xmin, xmax])
    plt.ylim([0, 1.05])
    # plt.xticks(np.linspace(0.1, 1.0, 10))
    # plt.yticks(np.linspace(0.1, 1.0, 10))
    plt.xticks(np.linspace(0.1, 1.0, 10), labels=[])
    plt.yticks(np.linspace(0.1, 1.0, 10), labels=[])
    # plt.legend()
    # plt.grid(True)  # 添加网格便于观察数据分布

    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    
    # 紧凑布局，让标签与图边缘不重叠
    plt.tight_layout()

    # 保存为高分辨率图片，适合投稿使用
    # plt.savefig('scatter_ieee.png', dpi=600)
    plt.savefig(os.path.join(path_folder_current, f'{metric}_vs_{acc_condition}.svg'), dpi=600)

    plt.show()

    ## Plot acc vs acc_unseen
    y_pred_linear, _ = fit_linear(accs_unseen, accs_all)
    plt.figure(figsize=(5,5))
    plt.scatter(accs_unseen, accs_all)
    plt.scatter(accs_unseen, y_pred_linear, color='black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('unseen vs all')
    plt.show()

    ## Plot acc vs acc_seen
    y_pred_linear, _ = fit_linear(accs_seen, accs_all)
    plt.figure(figsize=(5,5))
    plt.scatter(accs_seen, accs_all)
    plt.scatter(accs_seen, y_pred_linear, color='black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('seen vs all')
    plt.show()

    ## Plot acc_seen vs acc_unseen
    y_pred_linear, _ = fit_linear(accs_seen, accs_unseen)
    plt.figure(figsize=(5,5))
    plt.scatter(accs_seen, accs_unseen)
    plt.scatter(accs_seen, y_pred_linear, color='black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('seen vs unseen')
    plt.show()