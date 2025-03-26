## from iGZSL paper

import numpy as np
import pickle

datasets = ['benchmark', 'beta']
unseens = [8, 32]
times = [0.4, 0.6, 0.8, 1.0, 1.2]

method = 'igzsl-paper'
results = {}
# 2d matrix
results['benchmark'] = np.array([[0.812996,0.89639,0.924188,0.941877,0.964621],
                                 [0.657034,0.78365,0.845247,0.879468,0.903422]])
results['beta'] = np.array([[0.544829,0.653227,0.728813,0.759868,0.79561],
                                 [0.418773,0.534296,0.609386,0.675812,0.710469]])
dict_pkl = {'results': results, 'datasets': datasets, 'unseens': unseens, 'times': times}
with open('acc-pkls/'+method+'.pkl', 'wb') as pickle_file:
    pickle.dump(dict_pkl, pickle_file)

method = 'sst-paper'
results = {}
# 2d matrix
results['benchmark'] = np.array([[0.370758,0.560289,0.724549,0.805415,0.843321],
                                 [0.373004,0.547529,0.704943,0.790494,0.831559]])
results['beta'] = np.array([[0.331547,0.458696,0.567094,0.647368,0.699516],
                                 [0.335018,0.462094,0.560289,0.635379,0.684477]])
dict_pkl = {'results': results, 'datasets': datasets, 'unseens': unseens, 'times': times}
with open('acc-pkls/'+method+'.pkl', 'wb') as pickle_file:
    pickle.dump(dict_pkl, pickle_file)
