import torch
from torch import nn
import yaml

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Depth_Model(nn.Module):
    def __init__(self, state_dim=10, output_dim=128):
        super().__init__()
        self.dim_obs = state_dim         # quat, anglular rate, linear acceleration
        self.output_dim = output_dim         # ctbr   
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(self.dim_obs, 192)   
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, self.output_dim, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)   # activate

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))     
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, hx
