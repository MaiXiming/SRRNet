import scipy
import numpy as np
import platform
import os
from scipy import io

parameters_required = ['path', 'sampling_rate', 'frequencies', 'phases', 'channels', 'channels_decode', 'channels_decode_indices', 'trial_second', 'trial_timepoint', 'delay_timepoint', 'onset_timepoint', 'block_num']
system_os = platform.system().lower()

folder_data = '../../data/'

def load_dataset_parameters(dataset, server='default'):
    """
    Load dataset information.
    Parameters
    ----------
    dataset (str): dataset name 

    Returns
    ----------
    parameters (dictionary): dataset parameters 
    """
    
    if dataset == 'ucsd':
        parameters_dataset = load_ucsd_parameters(server)
    elif dataset == 'benchmark':
        parameters_dataset = load_benchmark_parameters(server)
    elif dataset == 'beta':
        parameters_dataset = load_beta_parameters(server)
    else:
        raise ValueError("Dataset not found!\n")
    
    is_parameters_exist(parameters_required, parameters_dataset)
    return parameters_dataset
    
    # calculate parameters for update_args: implement in that function, not here


def is_parameters_exist(parameters_required, parameters_dataset):
    for parameter in parameters_required:
        if parameter not in parameters_dataset.keys():
            raise ValueError(f"Required parameters <{parameter} >not contained in dataset's parameters!")
        
    
def load_ucsd_parameters(server='default'):
    parameters = {}
    # '/data3/mxm/data/ssvep_ucsd_2015/'
    # path_linux = os.path.expanduser('~/scratch/data/ssvep_ucsd_2015/')
    if server == 'pc':
        parameters['path'] = 'C:\\Users\\maixi\OneDrive - sjtu.edu.cn\\1Work\\1Lab\\Researches\\code_analysis\\data\\ssvep_ucsd_2015\\'
    else:
        parameters['path'] = folder_data + 'ssvep_ucsd_2015/'# '../../data/ssvep_ucsd_2015/'

    # inforaw = scipy.io.loadmat(parameters['path'] + 'info')
    inforaw = io.loadmat(parameters['path'] + 'info')
    parameters['sampling_rate'] = float(inforaw['fs'])
    parameters['frequencies'] = inforaw['freqs'][0]
    parameters['phases'] = inforaw['phases'][0]
    parameters['channels'] = inforaw['channels']
    parameters['channels'] = list(parameters['channels'][0])
    parameters['channels_decode'] = parameters['channels']
    parameters['channels_decode_indices'] = np.arange(len(parameters['channels'])).astype(int)

    parameters['trial_second'] = 4
    parameters['trial_timepoint'] = int(parameters['trial_second'] * parameters['sampling_rate'])
    parameters['delay_timepoint'] = 0.14 * parameters['sampling_rate']
    parameters['onset_timepoint'] = inforaw['smpl_stimonset'] # 38
    parameters['block_num'] = 15
    parameters['class_num'] = parameters['frequencies'].shape[1]

    return parameters


def load_benchmark_parameters(server='default'):
    parameters = {}
    if server == 'pc':
        parameters['path'] = '/home/mxm/Research/Datasets/ssvep_benchmark_2016/data_sum/'
    else:
        parameters['path'] = folder_data + 'ssvep_benchmark_2016/data_sum/'
    inforaw = io.loadmat(parameters['path'] + 'Freq_Phase')
    parameters['frequencies'] = inforaw['freqs'][0]
    parameters['phases'] = inforaw['phases'][0]

    chnls = []
    with open(parameters['path'] + '64-channels.txt', 'r') as f:
        tmp = f.readlines()
        for line in tmp:
            if line.strip():
                chnl = line[-4:]
                chnls.append(chnl.strip())
    # parameters['channels = np.array(chnls) # (64, )
    # parameters['channels = np.transpose(parameters['channels[:, np.newaxis]) # (1, 64)
    parameters['channels'] = chnls
    parameters['channels_decode'] = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    parameters['channels_decode_indices'] = [parameters['channels'].index(element) for element in parameters['channels_decode']]
    parameters['channels_decode_indices'] = np.array(parameters['channels_decode_indices']).astype(int)

    parameters['sampling_rate'] = 250
    parameters['trial_second'] = 5
    parameters['trial_timepoint'] = int(parameters['trial_second'] * parameters['sampling_rate'])
    parameters['delay_timepoint'] = 0.14 * parameters['sampling_rate']
    parameters['onset_timepoint'] = 0.5 * parameters['sampling_rate']
    parameters['block_num'] = 6
    parameters['class_num'] = parameters['frequencies'].shape[0]

    return parameters


def load_beta_parameters(server='default'):
    parameters = {}
    if server == 'pc':
        parameters['path'] = 'C:\\Users\\maixi\\OneDrive - sjtu.edu.cn\\1Work\\1Lab\Researches\\code_analysis\\data\\ssvep_beta\\data\\'
    else:
        parameters['path'] = folder_data + 'ssvep_beta/data/'
    # load struct from .mat: https://blog.csdn.net/weixin_43537379/article/details/119857729
    # inforaw = scipy.io.loadmat(parameters['path'] + 'S1')['data'][0][0]['suppl_info'] # each subject has suppl_info
    inforaw = io.loadmat(parameters['path'] + 'S1')['data'][0][0]['suppl_info'] # each subject has suppl_info
    parameters['frequencies'] = inforaw[0][0]['freqs'][0]
    parameters['phases'] = inforaw[0][0]['phases'][0]
    parameters['sampling_rate'] = float(inforaw[0][0]['srate'])

    channels_raw = list(inforaw[0][0]['chan'][:, 3])
    parameters['channels'] = [tmp[0] for tmp in channels_raw]
    parameters['channels_decode'] = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    parameters['channels_decode_indices'] = [parameters['channels'].index(element) for element in parameters['channels_decode']]
    parameters['channels_decode_indices'] = np.array(parameters['channels_decode_indices']).astype(int)

    parameters['trial_second'] = 2
    parameters['trial_timepoint'] = int(parameters['trial_second'] * parameters['sampling_rate'])
    parameters['delay_timepoint'] = 0.13 * parameters['sampling_rate']
    parameters['onset_timepoint'] = 0.5 * parameters['sampling_rate']
    parameters['block_num'] = 4
    parameters['class_num'] = parameters['frequencies'].shape[0]

    return parameters