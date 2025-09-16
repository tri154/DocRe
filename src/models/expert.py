import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)
        # self.bilinear = nn.Bilinear(in1_features, in2_features, out_features)

    def forward(self, pair_rep):
        output = self.ln(pair_rep)
        return output
