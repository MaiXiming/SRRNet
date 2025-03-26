import torch
import torch.nn as nn
from Algos.Conv2dWithConstraint import *


class SRRNet(nn.Module):
    def __init__(self, N2h=10, Nel=9, kernel_sf=6, kernel_t=10, dp=0.5, ch_expand=16):
        super().__init__()
        assert N2h == 10

        # self.cnn_expand = nn.Sequential(
        #     Conv2dWithConstraint(1, ch_expand, kernel_size=(N2h, 1), padding='same'),
        #     nn.BatchNorm2d(num_features=ch_expand),
        #     nn.PReLU(),
        #     nn.Dropout(dp),
        # ) # (ch_expand, N2h, tp)

        self.cnn_expand = nn.Sequential(
            Conv2dWithConstraint(1, ch_expand, kernel_size=(1, 10), padding='same'),
            Conv2dWithConstraint(ch_expand, ch_expand, kernel_size=(N2h, 1), padding='same'),
            nn.BatchNorm2d(num_features=ch_expand),
            nn.PReLU(),
            nn.Dropout(dp),
        ) # (ch_expand, N2h, tp)

        self.convs1 = nn.Sequential(
            nn.Conv2d(ch_expand, ch_expand, (1, kernel_t), padding='same'),
            nn.PReLU(),
            nn.Conv2d(ch_expand, ch_expand*2, (kernel_sf, 1), stride=(4, 1), padding=(kernel_sf//2, 0)),
            nn.BatchNorm2d(2*ch_expand),
        ) # (ch_expand*2, 3, tp)
        self.convs1_1x1 = nn.Conv2d(ch_expand, ch_expand*2, 1, stride=(4, 1))
        self.act1 = nn.PReLU()
        self.dp1 = nn.Dropout(dp)

        self.convs2 = nn.Sequential(
            nn.Conv2d(ch_expand*2, ch_expand*2, (1, kernel_t), padding='same'),
            nn.PReLU(),
            nn.Conv2d(ch_expand*2, ch_expand*4, (3, 1), padding=(1,0), stride=(4, 1)),
            nn.BatchNorm2d(4*ch_expand),
        ) # (ch_expand*4, 1, tp)
        self.convs2_1x1 = nn.Conv2d(ch_expand*2, ch_expand*4, 1, stride=(4,1))
        self.act2 = nn.PReLU()
        self.dp2 = nn.Dropout(dp)

        self.fc = nn.Sequential(
            nn.Conv2d(ch_expand*4, Nel, (1, kernel_t), padding='same'),
            nn.BatchNorm2d(Nel),
            nn.PReLU(),
            nn.Dropout(dp),
        )

        self.time = nn.LSTM(Nel, Nel, num_layers=1, batch_first=True, bidirectional=True)
        self.fc2 = nn.Conv1d(Nel*2, Nel, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.cnn_expand(x)

        y = self.convs1(x)
        y += self.convs1_1x1(x)
        y = self.dp1(self.act1(y))

        yy = self.convs2(y)
        yy += self.convs2_1x1(y)
        yy = self.dp2(self.act2(yy))

        yy = self.fc(yy)

        out_1d = torch.transpose(torch.squeeze(yy, 2), -2, -1) # (batch, tp, Nel)
        out_1d, _ = self.time(out_1d)

        out = self.fc2(torch.transpose(out_1d, -2, -1))
        return out



class RegResCNN(nn.Module):
    def __init__(self, N2h=10, Nel=9, kernel_sf=6, kernel_t=10, dp=0.5, ch_expand=16):
        super().__init__()
        assert N2h == 10

        self.cnn_expand = nn.Sequential(
            Conv2dWithConstraint(1, ch_expand, kernel_size=(N2h, 1), padding='same'),
            nn.BatchNorm2d(num_features=ch_expand),
            nn.PReLU(),
            nn.Dropout(dp),
        ) # (ch_expand, N2h, tp)

        self.convs1 = nn.Sequential(
            nn.Conv2d(ch_expand, ch_expand, (1, kernel_t), padding='same'),
            nn.PReLU(),
            nn.Conv2d(ch_expand, ch_expand*2, (kernel_sf, 1), stride=(4, 1), padding=(kernel_sf//2, 0)),
            nn.BatchNorm2d(2*ch_expand),
        ) # (ch_expand*2, 3, tp)
        self.convs1_1x1 = nn.Conv2d(ch_expand, ch_expand*2, 1, stride=(4, 1))
        self.act1 = nn.PReLU()
        self.dp1 = nn.Dropout(dp)

        self.convs2 = nn.Sequential(
            nn.Conv2d(ch_expand*2, ch_expand*2, (1, kernel_t), padding='same'),
            nn.PReLU(),
            nn.Conv2d(ch_expand*2, ch_expand*4, (3, 1), padding=(1,0), stride=(4, 1)),
            nn.BatchNorm2d(4*ch_expand),
        ) # (ch_expand*4, 1, tp)
        self.convs2_1x1 = nn.Conv2d(ch_expand*2, ch_expand*4, 1, stride=(4,1))
        self.act2 = nn.PReLU()
        self.dp2 = nn.Dropout(dp)

        self.fc = nn.Sequential(
            nn.Conv2d(ch_expand*4, Nel, (1, kernel_t), padding='same'),
            nn.BatchNorm2d(Nel),
            nn.PReLU(),
            nn.Dropout(dp),
        )

        # self.time = nn.LSTM(Nel, Nel, num_layers=1, batch_first=True, bidirectional=True)
        # self.fc2 = nn.Conv1d(Nel*2, Nel, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.cnn_expand(x)

        y = self.convs1(x)
        y += self.convs1_1x1(x)
        y = self.dp1(self.act1(y))

        yy = self.convs2(y)
        yy += self.convs2_1x1(y)
        yy = self.dp2(self.act2(yy))

        yy = self.fc(yy)
        out = torch.squeeze(yy, -2)

        # out_1d = torch.transpose(torch.squeeze(yy, 2), -2, -1) # (batch, tp, Nel)
        # out_1d, _ = self.time(out_1d)

        # out = self.fc2(torch.transpose(out_1d, -2, -1))
        return out