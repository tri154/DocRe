import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):

    def __init__(self, in1_features, in2_features, out_features):
        super().__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features)

    def forward(self, h_rep, t_rep):
        output = self.bilinear(h_rep, t_rep)
        return output


class MixtureOfExperts(nn.Module):

    def __init__(self, num_experts, in1_features, in2_features, out_features):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in1_features, in2_features, out_features) for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(in1_features + in2_features, num_experts),
        )

    def forward(self, h_rep, t_rep):
        expert_out = torch.stack([expert(h_rep, t_rep) for expert in self.experts], dim=1) # num_rel, 4, 2

        rel_rep = torch.cat([h_rep, t_rep], dim=-1)
        gate_logits = self.gate(rel_rep)
        gate_out = F.softmax(gate_logits, dim=1) # 14 ,4

        out = torch.bmm(gate_out.unsqueeze(1), expert_out).squeeze(1)
        return out
