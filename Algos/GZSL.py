import torch
import torch.nn as nn
from Algos.tcn import *

"""
Ref: Wang et al. 2023TNSRE A Generalized Zero-Shot Learning Scheme for SSVEP-Based BCI System
"""
class ExtractionNet(nn.Module):
    def __init__(self, electrode_num):
        super().__init__()
        self.electrode_num = electrode_num

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 9), padding='same'),
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            nn.Conv2d(1, 1, (self.electrode_num, 1), padding='valid'),
        )

    def forward(self, x):
        ## x: EEG data (batch, 1, Nel, Ns)
        return self.net(x)
    

class ElectrodeCombNet(nn.Module):
    def __init__(self, electrode_num):
        super().__init__()
        self.electrode_num = electrode_num

        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(1, 1, kernel_size=(self.electrode_num, 1), padding='valid'),
        )
    
    def forward(self, x):
        ## x: averaged template: (batch, 1, Nel, Ns)
        return self.net(x)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels=[24]*6, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GenerationNet(nn.Module):
    def __init__(self, in_ch, channels=[24]*6, group_num=8, frequencies=[]):
        super().__init__()
        self.in_ch = in_ch
        self.channels = channels
        self.group_num = group_num
        self.frequencies = frequencies

        assert frequencies.shape[0] % group_num == 0
        self.class_per_group = frequencies.shape[0] // group_num

        ## Sort&back idxs (freqs-->[8, 8.2, 8.4, ..., 15.8]
        self.freq_sorted, self.idx_sorted = torch.sort(self.frequencies)
        self.idx_back = [(self.freq_sorted==element).nonzero(as_tuple=True)[0] for element in self.frequencies]

        ## Group Nets
        self.nets = nn.ModuleList()
        for _ in range(self.group_num):
            net_sub = nn.Sequential(
                TemporalConvNet(self.in_ch, channels, kernel_size=3, dropout=0.2),
                nn.Conv2d(channels[-1], 1, kernel_size=(1,3), padding='same')
                ) # the author said kernel_size=(1, 3), not 1 stated in GZSL paper.
            self.nets.append(net_sub)
    
    def forward(self, x):
        # x: (batch, N2h, Nf, Ns)

        ## Sort Nf dim into 8-15.8Hz
        x_sorted = x[:, :, self.idx_sorted, :]

        ## Split into groups
        out_groups = []
        for ii in range(self.group_num):
            x_ii = x_sorted[:, :, (ii*self.class_per_group):((ii+1)*self.class_per_group),:]
            out_ii = self.nets[ii](x_ii) # (batch, 1, Nf/group, Ns)
            out_groups.append(out_ii)

        out = torch.cat(out_groups, dim=2) # (batch, 1, Nf, Ns)

        ## Order back
        return out[:, :, self.idx_back, :]
    

    