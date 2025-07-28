import torch
from torch import nn
import yaml
from tensordict import TensorDict


def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Depth(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()

        # input dim, dict[str, tuple[int]]
        self.state_dim = input_dim["state"][0]      # quat, anglular rate, linear acceleration, last action
        self.img_dim = input_dim["depth"]           # depth image

        # input dim, scaler
        self.output_dim = output_dim                # ctbr   

        self.hx = None
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),    # 32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),   # 64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(self.state_dim, 192)   
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, self.output_dim, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)   # activate

    def reset(self):
        pass

    def forward(self, obs):
        img_feat = self.stem(obs["depth"])
        x = self.act(img_feat + self.v_proj(obs["state"]))
        # h_new = self.gru(x, hidden_state)
        act = self.fc(self.act(x))
        return act