import torch
import torch.nn as nn

class RE(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RE, self).__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, input_dim // 2)
        self.re = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, output_dim)
        )
    def forward(self, h, t):
        out = torch.tanh(self.bilinear(h, t))
        out = self.re(out)
        return out
