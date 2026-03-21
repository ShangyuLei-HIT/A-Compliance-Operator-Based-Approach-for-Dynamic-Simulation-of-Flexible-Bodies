"""
Author: 雷尚谕
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SigmoidTanh(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * torch.tanh(x)

class MLP(nn.Module):
    def __init__(self,use_dropout,dropout_p,
                 hidden_dim,block_num):
        super().__init__()

        self.use_residual=True
        self.use_dropout=use_dropout
        # hidden_dim=128

        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.sigma1 = nn.Parameter(torch.tensor(0.0))          # 可学习标量
        self.shift1_x = nn.Parameter(torch.tensor(0.0))
        self.shift1_y = nn.Parameter(torch.tensor(0.0))
        self.shift1_z = nn.Parameter(torch.tensor(0.0))

        self.input_layer = nn.Sequential(
            nn.Linear(9, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),  # Swish 激活
            # Sin(),
            # SigmoidTanh(),
            # nn.GELU(),
            nn.Dropout(p=dropout_p) if use_dropout else nn.Identity()
        )

        self.res_blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout_p)
            for _ in range(block_num)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)

    def _make_block(self, dim, dropout_p):
        return nn.Sequential(
            nn.Linear(dim, dim),
            # nn.BatchNorm1d(dim),
            nn.SiLU(),
            # nn.LeakyReLU(),
            # Sin(),
            # SigmoidTanh(),
            # nn.GELU(),
            nn.Dropout(p=dropout_p) if self.use_dropout else nn.Identity()
        )

    def forward(self,x):
        dx = x[:,6]
        dy = x[:,7]
        dz = x[:,8]

        x = self.input_layer(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            if self.use_residual:
                x = x + residual
        x = self.output_layer(x)

        sigma1 = F.softplus(self.sigma1) * 1e-1 * 3
        r1 = (dx + self.shift1_x)**2 + (dy + self.shift1_y)**2 + (dz + self.shift1_z)**2
        k1 = torch.exp(-r1 / (2 * sigma1**2))
        k1 = k1.unsqueeze(-1)

        x = (torch.exp(self.w1) * k1 + 1.0) * x

        # if torch.isnan(self.sigma1) or torch.isnan(x).any():
        #     print("Detected NaN in sigma or x!")
        #     sys.exit()
        return x




