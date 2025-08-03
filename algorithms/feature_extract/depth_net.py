# import torch
# from torch import nn
# import yaml
# from tensordict import TensorDict
# import torch.nn.utils.rnn as rnn_utils

# def g_decay(x, alpha):
#     return x * alpha + x.detach() * (1 - alpha)

# class Depth(nn.Module):
#     def __init__(self, input_dim, output_dim=4, num_layers=1):
#         super().__init__()

#         # input dim, dict[str, tuple[int]]
#         self.num_envs = input_dim["state"][0]
#         self.state_dim = input_dim["state"][1:]      # quat, anglular rate, linear acceleration, last action
#         self.img_dim = input_dim["depth"][1:]          # depth image, 12*16

#         self.output_dim = output_dim                # ctbr   
        
#         self.num_layers = num_layers
#         self.hidden_states = torch.zeros((self.num_layers, self.num_envs, 192), device="cuda")
#         self.stem = nn.Sequential(
#             nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
#             nn.LeakyReLU(0.05),
#             nn.Conv2d(32, 64, 3, bias=False),    # 32, 6, 8 -> 64, 4, 6
#             nn.LeakyReLU(0.05),
#             nn.Conv2d(64, 128, 3, bias=False),   # 64, 4, 6 -> 128, 2, 4
#             nn.LeakyReLU(0.05),
#             nn.Flatten(),
#             nn.Linear(128*2*4, 192, bias=False),
#         )
#         self.v_proj = nn.Linear(*self.state_dim, 192)   
#         self.v_proj.weight.data.mul_(0.5)

#         # self.gru = nn.GRUCell(192, 192)
#         self.gru = nn.GRU(192*2, 192, num_layers=self.num_layers)
#         self.fc = nn.Linear(192, self.output_dim, bias=False)
#         self.fc.weight.data.mul_(0.01)
#         self.act = nn.LeakyReLU(0.05)   # activate

#     def reset(self, dones=None, hidden_states=None):
#         if dones is None:  # reset all hidden states
#             if hidden_states is None:
#                 self.hidden_states = None
#             else:
#                 self.hidden_states = hidden_states
#         elif self.hidden_states is not None:  # reset hidden states of done environments
#             if hidden_states is None:
#                 if isinstance(self.hidden_states, tuple):  
#                     for hidden_states in self.hidden_states:
#                         hidden_states[..., dones == 1, :] = 0.0
#                 else:
#                     self.hidden_states[..., dones == 1, :] = 0.0
#             else:
#                 NotImplementedError(
#                     "Resetting hidden states of done environments with custom hidden states is not implemented"
#                 )

#     def forward(self, obs, hidden_states=None, masks=None):
#         batch_mode = masks is not None
#         # batch mode (train)
#         if batch_mode:
#             T, B, C, H, W = obs["depth"].shape
#             depth_reshape = obs["depth"].permute(1, 0, 2, 3, 4).reshape(B * T, C, H, W)
#             state_reshape = obs["state"].permute(1, 0, 2).reshape(B * T, -1)

#             img_feat = self.stem(depth_reshape)                         # (B*T, 192)
#             state_proj = self.v_proj(state_reshape)                     # (B*T, 192)
#             # x = self.act(torch.cat([img_feat, state_proj], dim=-1))     # (B*T, 192*2)
#             x_seq = torch.cat([img_feat, state_proj], dim=-1).reshape(T, B, -1)                                 # (B, T, 192*2)
#             out_seq, h_new = self.gru(x_seq, hidden_states)             # (T, B, hidden_dim)
#             final_hidden = h_new[-1]                                    # (B, hidden_dim)
#             act = self.fc(self.act(final_hidden))                       # (B, num_actions)

#         # inference mode (rollout)
#         else:
#             img_feat = self.stem(obs["depth"])
#             x_tem = self.act(torch.cat([img_feat, self.v_proj(obs["state"])], dim=-1))     
#             _, self.hidden_states = self.gru(x_tem.unsqueeze(0), self.hidden_states)
#             act = self.fc(self.act(self.hidden_states[-1])).squeeze(0)

#         return torch.tanh(act)

import torch
from torch import nn
import yaml
from tensordict import TensorDict
import torch.nn.utils.rnn as rnn_utils

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Actor_net(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_layers=1):
        super().__init__()

        # input dim, dict[str, tuple[int]]
        self.num_envs = input_dim["state"][0]
        self.state_dim = input_dim["state"][1:]      # quat, anglular rate, linear acceleration, last action
        self.img_dim = input_dim["depth"][1:]          # depth image, 12*16

        self.output_dim = output_dim                # ctbr   
        
        self.num_layers = num_layers
        self.hidden_states = torch.zeros((self.num_layers, self.num_envs, 192), device="cuda")
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),    # 32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),   # 64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*8*12, 192, bias=False),
        )

        self.v_proj = nn.Linear(*self.state_dim, 192)   
        self.fc = nn.Linear(192*2, self.output_dim, bias=False)

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self, obs):
        img_feat = self.stem(obs["depth"])
        x_tem = torch.cat([img_feat, self.v_proj(obs["state"])], dim=-1)  
        act = self.fc(x_tem)
        act = torch.tanh(act)
        return act

class Critic_net(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_layers=1):
        super().__init__()

        # input dim, dict[str, tuple[int]]
        self.num_envs = input_dim["state"][0]
        self.state_dim = input_dim["state"][1:]      # quat, anglular rate, linear acceleration, last action
        self.img_dim = input_dim["depth"][1:]          # depth image, 12*16
        
        self.privileged_obs_dim = input_dim["privileged"][1:]
        self.output_dim = output_dim                # ctbr   
        
        self.num_layers = num_layers
        self.hidden_states = torch.zeros((self.num_layers, self.num_envs, 192), device="cuda")
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),    # 32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),   # 64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*8*12, 192, bias=False),
        )

        self.v_proj = nn.Linear(self.state_dim[0] + self.privileged_obs_dim[0], 192)     
        self.fc = nn.Linear(192*2, self.output_dim, bias=False)

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self, obs):
        img_feat = self.stem(obs["depth"])
        x_tem = torch.cat([img_feat, self.v_proj(torch.cat([obs["state"], obs["privileged"]], dim=-1))], dim=-1)  
        act = self.fc(x_tem)
        act = torch.tanh(act)
        return act