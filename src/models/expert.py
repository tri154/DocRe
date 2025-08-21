import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, in1_features, in2_features, out_features):
        super().__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features)

    def forward(self, h_rep, t_rep):
        output = self.bilinear(h_rep, t_rep)
        return output
