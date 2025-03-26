import torch
import torch.nn as nn


## Linear models
"""
使用线性回归模型，利用sine-cosine模版回归出时域平均模版。
实现思路：
- 对EEG信号进行平铺flatten，然后线性回归（参数量大）
- 直接对EEG信号通道进行组合（参数量小）
实现方法：
nn.Linear (nn.Conv1d也可以，需要对数据维度顺序调换transpose)

Inputs:
    x: (samples, 1, harmonic, tp) # need to add dim=1
Returns:
    out: (samples, chnl, tp)

"""

class LinregFlatten(nn.Module):
    def __init__(self, harmonic_num, channel_num, timepoint_num):
        super(LinregFlatten, self).__init__()
        self.harmonic_num = harmonic_num
        self.channel_num = channel_num
        self.timepoint_num = timepoint_num
        self.input_size = harmonic_num * timepoint_num
        self.output_size = channel_num * timepoint_num
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.output_size)
        )

    
    def forward(self, x):
        ## nn.Flatten(start_dim=1), ignore batch dim.
        out = self.layer(x)
        out = out.reshape(-1, self.channel_num, self.timepoint_num)
        return out


class LinregChannel(nn.Module):
    def __init__(self, harmonic_num, channel_num):
        super(LinregChannel, self).__init__()
        self.harmonic_num = harmonic_num
        self.channel_num = channel_num

        self.layer = nn.Sequential(
            nn.Linear(harmonic_num, channel_num, bias=True)
        )
    
    def forward(self, x):
        ## x: (batch, 1, harmonic, tp)
        x = x.squeeze(1)
        x = x.transpose(1, 2) # (batch, tp, harmonic)
        out = self.layer(x)
        return out.transpose(1, 2) # (batch, chnl, tp)


class LinearRegChnl(nn.Module):
    def __init__(self, harmonic_num, channel_num):
        super(LinearRegChnl, self).__init__()
        
        self.harmonic_num = harmonic_num
        self.channel_num = channel_num

        self.linear = nn.Conv1d(in_channels=self.harmonic_num, out_channels=self.channel_num, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        return self.linear(x)
    
