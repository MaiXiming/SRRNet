from corrs import *

import numpy as np
import matplotlib.pyplot as plt
import os
import math

## Locate current path
file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(file_path)

"""
figure: grid on. 
metrics: correlation; amp; 
"""

## Parameters
dataset = 'benchmark'
unseen = 32
freq = 14.4 # seen: 9, 14.8; unseen: 9.4; 14.4;
window = 1.0
subject = 31 # 31(s32) for figures
fs = 250
L = 1024

def get_1freq_amp(dataset, unseen, window, subject, freq, blkii=0, fbii=0, chii=-2, fs=250, folder='../data/details-norm0', is_normalize=1):
    ## Output: [blocks, freqs, (reg, true), (f, 2f)]: percentage calculate in other func
    
    data = get_data(dataset, unseen, window, subject, folder=folder)
    freqs = data['args'].frequencies
    freq_idx = np.where(abs(freqs==freq)<1e-3)[0][0]
    signal = data['recon_templates'][blkii,freq_idx,fbii, chii, :] # (250, )
    signal_true = data['true_templates'][blkii,freq_idx,fbii, chii, :] # (250, )

    # fs = 250    # 采样频率 (Hz)
    t_max = window    # 信号持续时间 (秒)
    t = np.arange(0, t_max, 1/fs)
    N = len(t)     # 样本点数


    ## Normalize
    if is_normalize:
        if math.isnan(np.std(signal)) or abs(np.std(signal)) < 1e-5:
            a = 1
        signal = (signal - np.mean(signal)) / np.std(signal)
        # signal = (signal - np.mean(signal)) / (np.max(signal) - np.min(signal))
        signal_true = (signal_true - np.mean(signal_true)) / np.std(signal_true)


    ## Zero-padding
    
    signal_pad = np.zeros((L))
    signal_pad[:N] = signal
    signal_true_pad = np.zeros((L))
    signal_true_pad[:N] = signal_true


    ## FFT
    Y = np.fft.fft(signal_pad)       # 复数谱
    Y1 = np.fft.fft(signal_true_pad)       # 复数谱
    freqs = np.fft.fftfreq(L, d=1.0/fs)  # 对应频率刻度 (可能包含正负频率)

    # === 3. 计算幅度谱并只取正频率部分(如果信号是实信号，一般只关注0~fs/2) ===
    amplitude = np.abs(Y)        # 每个频率点的幅度
    amplitude1 = np.abs(Y1)        # 每个频率点的幅度
    # 只取 0 ~ fs/2 的范围
    idx = np.where(freqs >= 0)   # 选择非负频率索引
    freqs_pos = freqs[idx]
    amplitude_pos = amplitude[idx]
    amplitude_pos1 = amplitude1[idx]


    ## Normalized
    amplitude_pos[1:-1] = amplitude_pos[1:-1] * 2
    amplitude_pos1[1:-1] = amplitude_pos1[1:-1] * 2
    amplitude_pos = amplitude_pos / L
    amplitude_pos1 = amplitude_pos1 / L

    
    freq_amps = np.zeros((2, 2)) # (reg, true) * (f, 2f)

    stim_idx = np.where(abs(freqs_pos-freq)<=0.5*fs/L)[0][0]
    freq_amps[0, 0] = max(amplitude_pos[stim_idx-1:stim_idx+2])
    freq_amps[1, 0] = max(amplitude_pos1[stim_idx-1:stim_idx+2])
    
    stim_idx = np.where(abs(freqs_pos-2*freq)<=0.5*fs/L)[0][0]
    freq_amps[0, 1] = max(amplitude_pos[stim_idx-1:stim_idx+2])
    freq_amps[1, 1] = max(amplitude_pos1[stim_idx-1:stim_idx+2])
    # print('f: Recon amp: ', max(amplitude_pos[stim_idx-1:stim_idx+2]))
    # print('f: True amp: ', max(amplitude_pos1[stim_idx-1:stim_idx+2]))

    # stim_idx = np.where(abs(freqs_pos-2*freq)<=0.5*fs/L)[0][0]
    # print('2f: Recon amp: ', max(amplitude_pos[stim_idx-1:stim_idx+2]))
    # print('2f: True amp: ', max(amplitude_pos1[stim_idx-1:stim_idx+2]))

    signals = {
        't': t, 
        'signal': signal,
        'signal_true': signal_true,
        'freqs_pos': freqs_pos, 
        'amplitude_pos': amplitude_pos,
        'amplitude_pos1': amplitude_pos1,
    }

    return freq_amps, signals

# data = get_data(dataset, unseen, window, subject)
# freqs = data['args'].frequencies
# freq_idx = np.where(freqs==freq)[0][0]
# signal = data['recon_templates'][0,freq_idx,0, -2, :] # (250, )
# signal_true = data['true_templates'][0,freq_idx,0, -2, :] # (250, )

# fs = 250    # 采样频率 (Hz)
# t_max = 1.0    # 信号持续时间 (秒)
# t = np.arange(0, t_max, 1/fs)
# N = len(t)     # 样本点数


# ## Normalize
# signal = (signal - np.mean(signal)) / np.std(signal)
# # signal = (signal - np.mean(signal)) / (np.max(signal) - np.min(signal))
# signal_true = (signal_true - np.mean(signal_true)) / np.std(signal_true)


# ## Zero-padding
# L = 1024
# signal_pad = np.zeros((L))
# signal_pad[:N] = signal
# signal_true_pad = np.zeros((L))
# signal_true_pad[:N] = signal_true


# ## FFT
# Y = np.fft.fft(signal_pad)       # 复数谱
# Y1 = np.fft.fft(signal_true_pad)       # 复数谱
# freqs = np.fft.fftfreq(L, d=1.0/fs)  # 对应频率刻度 (可能包含正负频率)

# # === 3. 计算幅度谱并只取正频率部分(如果信号是实信号，一般只关注0~fs/2) ===
# amplitude = np.abs(Y)        # 每个频率点的幅度
# amplitude1 = np.abs(Y1)        # 每个频率点的幅度
# # 只取 0 ~ fs/2 的范围
# idx = np.where(freqs >= 0)   # 选择非负频率索引
# freqs_pos = freqs[idx]
# amplitude_pos = amplitude[idx]
# amplitude_pos1 = amplitude1[idx]


# ## Normalized
# amplitude_pos[1:-1] = amplitude_pos[1:-1] * 2
# amplitude_pos1[1:-1] = amplitude_pos1[1:-1] * 2
# amplitude_pos = amplitude_pos / L
# amplitude_pos1 = amplitude_pos1 / L

# print(signal_true[:10])

if __name__ == '__main__':
    freq_amps, signals = get_1freq_amp(dataset, unseen, window, subject, freq)

    t = signals['t']
    signal = signals['signal']
    signal_true = signals['signal_true']
    freqs_pos = signals['freqs_pos']
    amplitude_pos = signals['amplitude_pos']
    amplitude_pos1 = signals['amplitude_pos1']

    ## Plot
    lw = 5
    lw_grid = 0.2
    true_linestyle = 'dashed'
    plt.figure(figsize=(6, 2))

    # 原始信号时域波形
    # plt.subplot(2,1,1)
    # plt.plot(t, signal, linewidth=lw, color=np.array([222,88,43])/255)
    # plt.plot(t, signal_true, linewidth=lw, color=np.array([24, 104, 178])/255, linestyle='--')
    plt.plot(t, signal, linewidth=lw, color=np.array([191,29,45])/255)
    plt.plot(t, signal_true, linewidth=lw-1, color='black', linestyle=true_linestyle)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    plt.ylim((-3, 3))
    plt.xlim((0, 1.1))
    plt.yticks([-2, 0, 2], labels=[])
    plt.xticks(np.linspace(0.1, 1.0, 10), labels=[])
    # plt.title('Time-Domain Signal')
    plt.tight_layout()
    plt.grid(
        True,           # 开启网格
        axis='both',    # 对哪条坐标轴显示网格
        which='major',  # 显示主刻度网格
        linestyle='-', # 网格线型
        linewidth=lw_grid   # 网格线宽
    )
    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    plt.savefig(os.path.join(folder_path, f'{dataset}-s{subject}-u{unseen}-{freq:.1f}Hz-temporal.svg'))
    # plt.savefig(os.path.join(folder_path, f'{dataset}-s{subject}-u{unseen}-{freq:2f}Hz-temporal.png'), dpi=600))
    plt.show()

    # 幅度频谱
    # plt.subplot(2,1,2)
    plt.figure(figsize=(6, 2))
    plt.plot(freqs_pos, amplitude_pos, linewidth=lw, color=np.array([191,29,45])/255)
    plt.plot(freqs_pos, amplitude_pos1, linewidth=lw-1, color='black', linestyle=true_linestyle)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    plt.xticks(np.linspace(0, 70, 8), labels=[])
    plt.yticks([0, 0.25, 0.5], labels=[])
    plt.ylim((0, 0.55))
    plt.xlim((0, 80))
    # plt.title('Frequency Spectrum (Single-sided)')
    plt.tight_layout()
    plt.grid(
        True,           # 开启网格
        axis='both',    # 对哪条坐标轴显示网格
        which='major',  # 显示主刻度网格
        linestyle='-', # 网格线型
        linewidth=lw_grid   # 网格线宽
    )
    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    plt.savefig(os.path.join(folder_path, f'{dataset}-s{subject}-u{unseen}-{freq:.1f}Hz-freq.svg'))
    # plt.savefig(os.path.join(folder_path, f'{dataset}-s{subject}-u{unseen}-{freq:2f}Hz-freq.png'), dpi=600))
    plt.show()

    ## Compute metrics
    print('Correlation: ', compute_corrcoef(signal, signal_true))

    stim_idx = np.where(abs(freqs_pos-freq)<=0.5*fs/L)[0][0]
    print('f: Recon amp: ', max(amplitude_pos[stim_idx-1:stim_idx+2]))
    print('f: True amp: ', max(amplitude_pos1[stim_idx-1:stim_idx+2]))

    stim_idx = np.where(abs(freqs_pos-2*freq)<=0.5*fs/L)[0][0]
    print('2f: Recon amp: ', max(amplitude_pos[stim_idx-1:stim_idx+2]))
    print('2f: True amp: ', max(amplitude_pos1[stim_idx-1:stim_idx+2]))

    a = 1

