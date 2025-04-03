from Outputs.analysis.templates_r1.corrs import *
import numpy as np
import matplotlib.pyplot as plt
import os

path_file = os.path.abspath(__file__)
path_folder = os.path.dirname(path_file)

datasets = ['benchmark', 'beta']
unseens = [8, 20, 32]
window = 1.0

def main(dataset, unseen, window):

    data = get_results(dataset, unseen, window, mean_condition='all', metric='corr', path_folder='../data/details-norm0')
    # data = get_results(dataset, unseen, window, mean_condition='all', metric='corr', path_folder='../data/details-alltime')
    accs = data['accs']
    accs_seen = np.expand_dims(data['accs_seen'], 0)
    accs_unseen = np.expand_dims(data['accs_unseen'], 0)


    data = np.concatenate((accs_seen, accs_unseen), axis=0)
    # data = np.concatenate((data, np.expand_dims(np.mean(data, axis=1), 0)), axis=1)
    # 设置图形
    plt.figure(figsize=(15, 6))  # 设置图形大小，因为要显示35个人，可能需要宽一些

    fs = 20
    plt.rcParams.update({
    'font.size': fs,           # 全局字体大小
    'axes.titlesize': 16,      # 标题字体大小
    'axes.labelsize': fs,      # x/y轴标签字体大小
    'xtick.labelsize': fs,     # x轴刻度标签字体大小
    'ytick.labelsize': fs,     # y轴刻度标签字体大小
    'legend.fontsize': fs      # 图例字体大小
    })

    # 设置柱状图的位置
    n_people = data.shape[1]  # 人数
    bar_width = 0.35  # 柱子的宽度
    index = np.arange(n_people)+1  # 每个人的位置

    # 绘制第一组数据（第一行）
    plt.bar(index - bar_width/2, data[0], bar_width, 
            label='Seen', color='b', alpha=0.7)

    # 绘制第二组数据（第二行）
    plt.bar(index + bar_width/2, data[1], bar_width, 
            label='Unseen', color='r', alpha=0.7)

    # 添加图例和标签
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    # plt.title('Comparison of Two Data Points for 35 People')
    plt.yticks(np.linspace(0, 1, 5))
    # plt.xticks(index, rotation=45)
    plt.xticks(np.arange(1, n_people+1, 4))
    # plt.xticks(index, [f'Person {i+1}' for i in range(n_people)], rotation=45)
    plt.legend(loc='lower right')

    # 调整布局，防止标签被截断
    plt.tight_layout()

    plt.savefig(os.path.join(path_folder, f"{dataset}-u{unseen}-t{window:.1f}.svg"))

    # 显示图形
    plt.show()

    print(f"{dataset}-u{unseen}-t{window:.1f}", np.mean(data, axis=1))


if __name__ == '__main__':
    for dataset in datasets:
        for unseen in unseens:
            main(dataset, unseen, window)