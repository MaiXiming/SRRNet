from Outputs.SaveForAnalysis.templates_r1.corrs import *
import os

path_file_current = os.path.abspath(__file__)
path_folder_current = os.path.dirname(path_file_current)

# print(get_data('benchmark', 8, 1.0, 31, 'Details-norm1'))

def get_accs(dataset, unseen, window, data_folder):
    subjects = 35 if dataset=='benchmark' else 70
    accs = np.zeros((subjects))
    for sii in range(subjects):
        data = get_data(dataset, unseen, window, sii, data_folder)
        accs[sii] = data['acc']
    
    return np.mean(accs), accs


datasets = ['benchmark', 'beta']
unseens = [8, 32, 20, ]
window = 1.0

for dataset in datasets:
    for unseen in unseens:
        acc, _ = get_accs(dataset, unseen, window, os.path.join(path_folder_current, './../templates_r1/Details-norm1/'))
        print(f'{dataset} u{unseen}, with harmonic phase, Acc=', acc)
        acc, _ = get_accs(dataset, unseen, window, os.path.join(path_folder_current,'./../../Results/Details/'))
        print(f'{dataset} u{unseen}, without harmonic phase, Acc=', acc)
        

