import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import numpy as np
import pickle

blocks = 6
class_num = 40

# Load  data
dataset, unseen, t = 'benchmark', 8, 1
prep, model = 'trca', 'rescnn_lstm'
subject = 31

data_raw, data_trca, data_corr = [], [], []
for bii in range(blocks):
    with open(f'data/{dataset}-u{unseen}-t{t}-{prep}-{model}-fb1-s{subject}-b[{bii}].pickle', 'rb') as file:
        outputs = pickle.load(file)
    data_raw.append(outputs['raw'])
    data_trca.append(outputs['trca'])
    data_corr.append(outputs['predict'])
data_raw = np.stack(data_raw, axis=0) # block, samples, chnl, tp
data_trca = np.stack(data_trca, axis=0)
data_corr = np.stack(data_corr, axis=0)
labels = [np.arange(0, 40) for _ in range(blocks)]
labels = np.reshape(labels, (-1,))

data_raw = np.reshape(data_raw, (data_raw.shape[0]*data_raw.shape[1], -1))
data_trca = np.reshape(data_trca, (data_trca.shape[0]*data_trca.shape[1], -1))
data_corr = np.reshape(data_corr, (data_corr.shape[0]*data_corr.shape[1], -1))

def reduce_tsne(data, label, condition):
    tsne = TSNE(n_components=2, random_state=42)
    x_reduced = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(10,9.3), constrained_layout=True)
    # colors = plt.cm.jet(np.arange(0, class_num, 1))
    colors = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
         list(plt.cm.tab20b(np.linspace(0, 1, 20))) + \
         list(plt.cm.tab20c(np.linspace(0, 1, 20)))

    for cii in range(class_num):
        indices = labels == cii
        plt.scatter(x_reduced[indices,0], x_reduced[indices, 1], 
                    color=colors[cii], label=cii)
    # plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=label.astype(int), cmap='jet', alpha=0.6)
    # plt.colorbar()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(f'{condition}.svg')
    print(f"{condition} done")

# reduce_tsne(data_raw, labels, 'raw')
# reduce_tsne(data_trca, labels, 'trca')
reduce_tsne(data_corr, labels, 'corr')
# ## Raw
# tsne = TSNE(n_components=2, random_state=42)
# x_reduced = tsne.fit_transform(data_raw)

# # Plot the result
# plt.figure(figsize=(13,10))
# plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels.astype(int), cmap='jet', alpha=0.6)
# plt.colorbar()
# # plt.show()
# plt.savefig('raw.png')

# ## TRCA
# tsne = TSNE(n_components=2, random_state=42)
# x_reduced = tsne.fit_transform(data_trca)

# # Plot the result
# plt.figure(figsize=(13,10))
# plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels.astype(int), cmap='jet', alpha=0.6)
# plt.colorbar()
# # plt.show()
# plt.savefig('trca.png')

# ## Corr
# tsne = TSNE(n_components=2, random_state=42)
# x_reduced = tsne.fit_transform(data_corr)

# # Plot the result
# plt.figure(figsize=(13,10))
# plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels.astype(int), cmap='jet', alpha=0.6)
# plt.colorbar()
# # plt.show()
# plt.savefig('corr.png')


