"""
After ctrl-c, need to kill program one by one:
    $ nvidia-smi # get pids
    $ kill pid
"""

import subprocess
import re
import time
# import os
import numpy as np
# import math
from datetime import datetime
import pynvml

dataset_subjects = {'ucsd': 10, 'benchmark': 35, 'beta': 70}

GPUS_ASSIGN = [0,1,2,3] ## ALWAYS LEAVE SOME GPUS FOR OTHERS :)
UTIL_LIMIT = 75
MEMO_REQUIRED = 12000 # 12000 *** based on onefold.py pretrain:20000
ASSIGN_FROZEN_TIME = 45 # 45 # *** 100-pretrain; wait for prev job to load data and steadily run
MEAN_STATUS_TIME = 5 # 5; 45-pretrain

TIMER = np.zeros(8) # timer for each gpu

## Params
datasets = ['beta', 'benchmark', ] # 
unseens = [32, 8, 20, ] # 20, 
windows = [0.4, 0.6, 0.8, 1.0, 1.2, ] # 0.4, 0.6, 0.8, 1.0, 1.2, 
step = 25 ## (0,1): second; [1,100]: win%
epoch_num = 200
batch_size = 64 # 32-gzsl; 64-Reg
Nh = 5 # 3-gzsl; 5-reg
is_losscurve = 0

loss = 'mse' # corr mse

models = ['srrnet'] # srrnet, igzsl, srrv2
fb_num = 5 # 5; gzsl-1; 
spatialfilter = 'trca' # none trca tdca ecca

is_cudnn_fixed = 1 # 0 for gzsl
is_pretrain = 0 # 1-igzsl; 
is_earlystop = 1

## Fixed
opt = 'adam' # adam sgd
dp = 0.5
lr, wd = 1e-3, 1e-5
is_tmpl_trueseen = 1

def main():
    pynvml.nvmlInit()

    # subjects_select = [0, 27,28,29,30,31]
    for dataset in datasets:
        subjects = dataset_subjects[dataset]
        for unseen_num in unseens:
            # processes = []
            for window_second in windows:
                for model in models:
                    timenow_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    
                    # for fold in subjects_select:
                    for fold in range(subjects):
                        # is_first_fold = 1 if fold == 0 else 0
                        job = f"python main_fold1.py " + \
                        f"--dataset {dataset} " + \
                        f"--subjects {subjects} " + \
                        f"--unseen-num {unseen_num} " + \
                        f"--step {step} " + \
                        f"--fb-num {fb_num} " + \
                        f"--model {model} " + \
                        f"--spatialfilter {spatialfilter} " + \
                        f"--subject {fold} " + \
                        f"--is-pretrain {is_pretrain} " + \
                        f"--is-cudnn-fixed {is_cudnn_fixed} " + \
                        f"--window {window_second} " + \
                        f"--epoch-num {epoch_num} " + \
                        f"--batch-size {batch_size} " + \
                        f"--harmonic-num {Nh} " + \
                        f"--lr {lr} " + \
                        f"--opt {opt} " + \
                        f"--loss {loss} " + \
                        f"--dropout {dp} " + \
                        f"--weight-decay {wd} " + \
                        f"--is-earlystopping {is_earlystop} " + \
                        f"--es-patience {100} " + \
                        f"--is-normalize {1} " + \
                        f"--is-phase-harmonic {1} " + \
                        f"--is-tmpl-trueseen {is_tmpl_trueseen} " + \
                        f"--is-plot-template {0} " + \
                        f"--is-plot-weights {0} " + \
                        f"--is-csv-output {1} " + \
                        f"--is-losscurve {is_losscurve} " + \
                        f"--is-plot-template {0} " + \
                        f"--timenow {timenow_str} " + \
                        f"--nfolds {1} " + \
                        f"> Tmp/logs/fold{fold}_log"
                        process = submit_pyjob(job)

    pynvml.nvmlShutdown()
    

def get_gpu_status(gpu_index):
    unit_per_gb = 1024**2 # GB=1024MB=1024**2KB=1024**3bytes
    # Initialize NVML
    # pynvml.nvmlInit()

    # Get handle for the GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    
    # Get memory info
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Extract total memory and used memory
    total_memory = memory_info.total
    used_memory = memory_info.used
    free_memory = (total_memory - used_memory)
    
    # Get GPU utilization info
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_utilization_percentage = utilization.gpu
    
    # Shutdown NVML
    # pynvml.nvmlShutdown()
    
    # memo unit: MB
    return free_memory/unit_per_gb, gpu_utilization_percentage, total_memory/unit_per_gb, used_memory/unit_per_gb

def submit_pyjob(cmd):
    t0 = time.time()
    gpu_assign = -1
    while gpu_assign == -1:
        for gpu in GPUS_ASSIGN:
            if is_gpu_avail(gpu):
                gpu_assign = gpu
                break
            # time.sleep(MEAN_STATUS_TIME)
        if gpu_assign == -1:
            dt = time.time() - t0
            print(f'\rGPU_ASSIGN are occupied... please wait ({dt:2f}sec)', end='')
        
    # assign job
    cmd = f"CUDA_VISIBLE_DEVICES='{gpu_assign}'" + " " + cmd
    print(yellow('\r' + cmd))
    # os.system(cmd)
    process = subprocess.Popen(cmd, shell=True)
    TIMER[gpu_assign] = time.time()
    return process

def get_gpu_mean_status(gpu, mean_time=MEAN_STATUS_TIME):
    free_memos, utils = [], []
    t0 = time.time()
    while (time.time() - t0) < mean_time:
        m, u, _, _ = get_gpu_status(gpu)
        free_memos.append(m)
        utils.append(u)

    return np.min(free_memos), np.max(utils)

def is_gpu_avail(gpu):
    is_newjob_time = False
    if time.time() - TIMER[gpu] >= ASSIGN_FROZEN_TIME:
        TIMER[gpu] = 0
        is_newjob_time = True
        
    is_gpu_idle = False
    # free_memo, util, _, _ = get_gpu_status(gpu)
    free_memo, util = get_gpu_mean_status(gpu)
    if (free_memo > (MEMO_REQUIRED+500)) and (util <= UTIL_LIMIT):
        is_gpu_idle = True
        
    return (is_gpu_idle and is_newjob_time)


# # Example usage for GPU 0
# gpu_index = 0
# total_memory, used_memory, gpu_utilization_percentage = get_gpu_status(gpu_index)
# print(f"Total Memory (GPU {gpu_index}): {total_memory / (1024**3):.2f} GB")
# print(f"Used Memory (GPU {gpu_index}): {used_memory / (1024**3):.2f} GB")
# print(f"GPU Utilization (GPU {gpu_index}): {gpu_utilization_percentage:.2f}%")


# def wait_dataloading(t):
#     for count in range(t):
#         time.sleep(1)
#         print(f"\rwaiting for data loading ... {count} / {t} ", end='')
            

def yellow(x):
    return '\033[93m' + str(x) + '\033[0m'


# def get_gpu_jobpid(gpu_id):
#     output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader', '--id='+str(gpu_id)]).decode()
#     pids = [line.strip() for line in output.strip().split('\n') if line.strip()]
#     return pids

# def get_pid_user(pid):
#     try:
#         user = subprocess.check_output(['ps', '-o', 'user=', '-p', str(pid)]).decode().strip()
#         return user
#     except subprocess.CalledProcessError:
#         return 'Unknown'

# def is_gpu_room_for_newjob(gpu_id, upperlimit=60):
#     ## check for 15s
#     gpu_uses = []
#     for _ in range(90):
#         gpu_percent = int(get_gpu_usage(gpu_id))
#         if gpu_percent >= upperlimit:
#             return False
#         else:
#             gpu_uses.append(gpu_percent)
#             time.sleep(0.5)
#     gpu_usage = np.max(gpu_uses)
#     # gpu_usage = get_gpu_usage(gpu_id)
#     return gpu_usage <= upperlimit

# def is_user_using_gpu_any(gpus, user):
#     flag = False
#     for gpu in gpus:
#         if is_user_using_gpu(gpu, user):
#             flag = True
#             break
#     return flag

# def is_user_using_gpu(gpu_id, user):
#     is_user_using = False

#     pids = get_gpu_jobpid(gpu_id)
#     if pids:
#         for pid in pids:
#             user_now = get_pid_user(pid)
#             if user == user_now:
#                 is_user_using = True
    
#     return is_user_using

# def is_anyone_using_gpu(gpu_id):
#     if get_gpu_jobpid(gpu_id):
#         return True
#     else:
#         return False


# def get_gpu_usage(gpu_id):
#     # 执行 nvidia-smi 命令
#     nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv']).decode()

#     # print(nvidia_smi_output)
#     # 解析输出以获取GPU使用率
#     gpu_usages = []
#     for line in nvidia_smi_output.split('\n'):
#         # print('line:', line)
#         match = re.search(r'([0-9]+).*%', line)
#         if match:
#             usage = int(match.group(1))
#             gpu_usages.append(usage)

#     return gpu_usages[gpu_id]


if __name__ == "__main__":
    main()


