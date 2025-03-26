from corrs import *

import pickle
import numpy as np
import os

# for subject in range(35):
dataset = 'benchmark'
unseen = 32
window = 1.0
subject = 5

file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(file_path)

filename = f"Details/{dataset}-u{unseen}-t{window:.1f}-s{subject}.pickle"
filename = os.path.join(folder_path, filename)
# filename = os.path.join('/data3/mxm/Code/gzsl_reg_simple/Outputs/SaveForAnalysis/Templates-R1/', filename)
with open(filename, 'rb') as file:
    data = pickle.load(file)

# print(data)


## Pearson correlation
for fbii in range(5):
    # fbii = 0
    unseen_idx = 0
    channel_idx = -2 # Oz

    recon_templates = data['recon_templates'][unseen_idx, fbii, channel_idx, :]
    true_templates = data['true_templates'][unseen_idx, fbii, channel_idx, :]
    print(compute_corrcoef(recon_templates, true_templates), end=' ')

print()