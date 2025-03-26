import multiprocessing
import subprocess
import signal
import sys
import os
import time
from datetime import datetime
"""
Improve: using cpu_num < subjects + lock cpu by .lock. 
"""

dataset = 'beta'
# wind_sec = 0.6
windows = [1.0, 0.8, 0.6, ] # 
fb_num = 5
model = 'tdca'
# trad_trainblocks = 4
trainblocks = [2,3,] # 4,5,
# unseen_num = 20
# is_tmpl_trueseen = 1

datasets = {'ucsd': 10, 'benchmark': 35, 'beta': 70}
subjects = datasets[dataset]

timenow = datetime.now()
timenow_str = timenow.strftime("%Y%m%d-%H%M%S")


# def run_program(fold, cpu_affinity):
def run_program(fold, cpu_affinity, Nb, win):
    try:
        # 设置进程的 CPU 亲和性
        if cpu_affinity:
            os.sched_setaffinity(0, cpu_affinity)

        is_first_fold = 1 if fold == 0 else 0

        cmd = f"python onefold.py " + \
        f"--dataset {dataset} " + \
        f"--subjects {subjects} " + \
        f"--trad-trainblocks {Nb} " + \
        f"--fb-num {fb_num} " + \
        f"--model {model} " + \
        f"--subject {fold} " + \
        f"--window {win} " + \
        f"--is-normalize {1} " + \
        f"--is-csv-output {1} " + \
        f"--timenow {timenow_str} " + \
        f"> Tmp/logs/fold{fold}_log"
        
        # 在这里执行你的程序，可以使用subprocess模块或其他合适的方式
        # subprocess.run(command, shell=True)
        process = subprocess.Popen(cmd, shell=True)

        if is_first_fold:
            time.sleep(3)

    except Exception as e:
        print(f"Error in run_program: {e}")


if __name__ == "__main__":
    num_cpus_to_use = subjects

    # 获取CPU核心数
    num_cpus_available = multiprocessing.cpu_count()

    # 确保要使用的CPU数不超过实际可用的CPU数
    num_cpus_to_use = min(num_cpus_to_use, num_cpus_available)

    # 指定每个进程绑定到特定的CPU
    cpu_affinities = [(i % num_cpus_available,) for i in range(num_cpus_to_use)]

    # 创建进程列表
    processes = []

    def signal_handler(sig, frame):
        print("Ctrl-C pressed. Stopping all processes.")
        for process in processes:
            process.terminate()
        sys.exit(0)

    # 注册Ctrl-C信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    for Nb in trainblocks:
        for win in windows:
            try:
                for i, cpu_affinity in enumerate(cpu_affinities):
                    # process = multiprocessing.Process(target=run_program, args=(i, cpu_affinity))
                    process = multiprocessing.Process(target=run_program, args=(i, cpu_affinity, Nb, win))
                    processes.append(process)
                    process.start()

                # 等待所有进程完成
                for i, process in enumerate(processes):
                    process.join()
                    print(f'Process {i} started.')

            except KeyboardInterrupt:
                print("Ctrl-C pressed. Stopping all processes.")
                for process in processes:
                    process.terminate()
