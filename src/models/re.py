import torch
import torch.nn as nn

class RE(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RE, self).__init__()
        self.w_h = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
        self.w_t = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
        self.bilinear = nn.Bilinear(input_dim // 4, input_dim // 4, output_dim)


    def forward(self, h, t):
        h_rep = self.w_h(h)
        t_rep = self.w_t(t)
        out = self.bilinear(h_rep, t_rep)
        return out
