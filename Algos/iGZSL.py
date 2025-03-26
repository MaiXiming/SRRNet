import torch
import torch.nn as nn
from Algos.tcn import *
import numpy as np
"""
Ref: Wang et al. 2023TNSRE Improving Generalized Zero-Shot Learning SSVEP Classification Performance From Data-Efficient Perspective
"""
class TransformerNet(nn.Module):
    def __init__(self, electrode_num=9, electrode_embed_num=64, timepoint_num=250, kernel=7, num_heads_mha=8, dp=0.1):
        super().__init__()
        ## Input: (batch, Nch, Ns)
        self.block_cnn_embed = nn.Conv1d(electrode_num, electrode_embed_num, kernel_size=kernel, padding='same')

        self.block_atten_encoder = nn.Sequential(
            EncoderBlock(electrode_embed_num, timepoint_num, num_heads_mha, dp),
            EncoderBlock(electrode_embed_num, timepoint_num, num_heads_mha, dp),
        )

        self.block_fc_wave = nn.Sequential(
            nn.LayerNorm((timepoint_num, electrode_embed_num)),
            nn.Linear(electrode_embed_num, 1)
        )


    def forward(self, x):
        ## x: (batch, 1, Nel, Ns)
        x = torch.squeeze(x, dim=1)
        x_embed = self.block_cnn_embed(x)
        x_embed = torch.transpose(x_embed, -2, -1) # (batch, Ns, Nel)

        x_encode = self.block_atten_encoder(x_embed) # (batch, Ns, Nel)
        out = self.block_fc_wave(x_encode) # (batch, Ns, 1)
        out = torch.transpose(out, -2, -1) # (batch, 1, Ns)
        out = torch.unsqueeze(out, 1) # (batch, 1, 1, Ns)
        return out, x_encode


class EncoderBlock(nn.Module):
    """
    Solved:
    1. channel attention: num_heads=Ns//10, not 8 (from Xietian's answer). (AssertError: embed_dim(125/250) % num_heads(8) != 0 
    2. FeedForward: 1 MLP, not 1 linear as paper stated.
    3. FeedForward after CA: still on channels, not on timepoints (should be right bcz the author didn't point out this issue after reading my code)
    """
    def __init__(self, electrode_embed_num=64, timepoint_num=250, num_heads_mha=8, dp=0.1):
        super().__init__()
        self.electrode_embed_num = electrode_embed_num
        self.timepoint_num = timepoint_num
        assert self.timepoint_num % 10 == 0
        num_heads_ca = int(self.timepoint_num // 10)

        self.ln1 = nn.LayerNorm((timepoint_num, electrode_embed_num))
        self.mha = nn.MultiheadAttention(embed_dim=electrode_embed_num, num_heads=num_heads_mha, batch_first=True)
        self.dp1 = nn.Dropout(dp)

        self.ln2 = nn.LayerNorm((timepoint_num, electrode_embed_num))
        # self.ffn1 = nn.Linear(electrode_embed_num, electrode_embed_num)
        self.ffn1 = nn.Sequential(
            nn.Linear(electrode_embed_num, electrode_embed_num),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(electrode_embed_num, electrode_embed_num),
        )
        # self.dp2 = nn.Dropout(dp) # contained in ffn

        ## Transpose dim for MHA on channel
        self.ln3 = nn.LayerNorm((electrode_embed_num, timepoint_num))
        self.ca = nn.MultiheadAttention(embed_dim=timepoint_num, num_heads=num_heads_ca, batch_first=True)
        self.dp3 = nn.Dropout(dp)
        ## Transpose back for linear comb on embed channels
        self.ln4 = nn.LayerNorm((timepoint_num, electrode_embed_num))
        # self.ffn2 = nn.Linear(electrode_embed_num, electrode_embed_num)
        self.ffn2 = nn.Sequential(
            nn.Linear(electrode_embed_num, electrode_embed_num),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(electrode_embed_num, electrode_embed_num),
        )
        # self.dp4 = nn.Dropout(dp)


    def forward(self, x):
        ## x: (batch, tp, embed) after conv embedding
        x = self.ln1(x)
        out1, _ = self.mha(x, x, x)
        out1 = self.dp1(out1)
        out1 += x # residual

        # out2 = self.dp2(self.ffn1(self.ln2(out1)))
        out2 = self.ffn1(self.ln2(out1))
        out2 += out1 # residual

        out2 = torch.transpose(out2, 1, 2) # (batch, embed, tp)
        out3 = self.ln3(out2)
        out3, _ = self.ca(out3, out3, out3)
        out3 = self.dp3(out3)
        out3 += out2 # residual

        out3 = torch.transpose(out3, 1, 2) # (batch, tp, embed)
        # out4 = self.dp4(self.ffn2(self.ln4(out3)))
        out4 = self.ffn2(self.ln4(out3))
        out4 += out3 # residual

        return out4


class ClassificationNet(nn.Module):
    def __init__(self, electrode_embed_num=64, timepoint_num=250, dp=0.4, seen_num=20):
        super().__init__()

        self.conv_ch_comb = nn.Sequential(
            nn.Conv2d(1, 20, (electrode_embed_num, 1), padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        self.conv_tp_half = nn.Sequential(
            nn.Conv2d(20, 20, (1, 2), stride=(1, 2), padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(dp),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20*int(timepoint_num//2), seen_num),
        )

    def forward(self, x):
        ## x: (batch, tp, ele_embed)
        x = torch.unsqueeze(torch.transpose(x, -2, -1), 1) # (batch, 1, ele_embed, tp)
        x = self.conv_ch_comb(x)
        x_half = self.conv_tp_half(x)
        return self.fc(x_half)



if __name__ == '__main__':

    print("Test Encoder block:")
    fs = 250
    ts = np.linspace(0.2, 1, 5)
    for tt in ts:
        tp_num = int(tt*fs)
        print(tp_num, end='')
        ch_num = 64
        x = torch.rand((64, tp_num, ch_num))
        net = EncoderBlock(ch_num, tp_num)
        out = net(x)
        print(out.shape)

    print("Test Transformer Net:")
    electrode_embed_num = 64
    for tt in ts:
        tp_num = int(tt*fs)
        print(tp_num, end='')
        ch_num = 9
        x = torch.rand((64, 1, ch_num, tp_num))
        net = TransformerNet(ch_num, electrode_embed_num, tp_num)
        cnet = ClassificationNet(electrode_embed_num, tp_num)
        out, out_encoder = net(x)
        out2 = cnet(out_encoder)
        print(out.shape, out2.shape)