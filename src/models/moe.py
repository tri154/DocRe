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

    def __init__(self, cfg, num_experts, in1_features, in2_features, out_features):
        super().__init__()
        self.cfg = cfg
        self.num_experts = num_experts
        self.more_logging = False

        self.stats = torch.zeros(num_experts)
        self.experts = nn.ModuleList([
            Expert(in1_features, in2_features, out_features) for _ in range(num_experts)
        ])

        self.gate = nn.Bilinear(in1_features, in2_features, num_experts)

    def reset_stats(self):
        self.stats = torch.zeros(self.num_experts)

    def set_more_logging(self, value):
        self.more_logging = value

    def forward(self, h_rep, t_rep):
        expert_out = torch.stack([expert(h_rep, t_rep) for expert in self.experts], dim=1) # num_rel, 4, 2

        gate_logits = self.gate(h_rep, t_rep)
        gate_out = F.softmax(gate_logits, dim=1) # 14 ,4

        if self.more_logging:
            self.cfg.logging(f"{gate_out} ")
        temp = F.one_hot(torch.argmax(gate_out.detach(), dim=-1), num_classes=gate_out.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()

        out = torch.bmm(gate_out.unsqueeze(1), expert_out).squeeze(1)
        return out
