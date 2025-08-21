import torch
import torch.nn as nn
import torch.nn.functional as F
from models.expert import Expert

class MoeSparse(nn.Module):
    # TODO: remove uniform noise if normal noise works well

    def __init__(self, cfg, num_experts, topk, noise_scale, in1_features, in2_features, out_features):
        super().__init__()
        self.cfg = cfg
        self.num_experts = num_experts
        self.topk = topk
        self.noise_scale = noise_scale
        self.more_logging = False

        self.stats = torch.zeros(num_experts)
        self.experts = nn.ModuleList([
            Expert(in1_features, in2_features, out_features) for _ in range(num_experts)
        ])

        self.gate = nn.Bilinear(in1_features, in2_features, num_experts)
        self.noise = nn.Bilinear(in1_features, in2_features, num_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.noise.weight)
        nn.init.zeros_(self.noise.bias)

    def forward(self, h_rep ,t_rep, is_training=True, noise_epsilon=1e-2):
        # NOTE: be carefull when modify.
        return self.forward_not_dispatch(h_rep, t_rep, is_training=is_training)


    def forward_not_dispatch(self, h_rep ,t_rep, is_training=True, noise_epsilon=1e-2):
        # NOTE: add load balance loss, if need.
        gate_logits = self.gate(h_rep, t_rep)
        if is_training:
            noise_stddev = F.softplus(self.noise(h_rep, t_rep)) + noise_epsilon
            noise_logits = torch.randn(gate_logits.shape).to(gate_logits.device) * noise_stddev
            gate_logits = gate_logits + noise_logits

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_indices = gate_probs.topk(self.topk, dim=-1)
        if self.topk > 1:
            top_logits = top_logits / top_logits.sum(dim=-1, keepdim=True)

        zeros = torch.zeros_like(gate_probs, requires_grad=True, device=gate_probs.device)
        gates = zeros.scatter(dim=1, index=top_indices, src=top_logits)

        if self.more_logging:
            self.cfg.logging(f"{gate_probs }")
        temp = F.one_hot(torch.argmax(gates.detach(), dim=-1), num_classes=gates.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()

        n_sample = h_rep.shape[0]
        out = list()
        for sample_id in range(n_sample):
            expert_ids = top_indices[sample_id]
            sample_gate = top_logits[sample_id]
            sample_out = torch.stack([self.experts[i](h_rep[sample_id], t_rep[sample_id]) for i in expert_ids])
            out.append(sample_gate @ sample_out)
        out = torch.stack(out)

        return out, gates


















    def forward_dispatch(self, h_rep ,t_rep, is_training=True, noise_epsilon=1e-2):
        # NOTE: add load balance loss, if needed.
        # NOTE: currently not in use, stick with brute force for reliable.
        gate_logits = self.gate(h_rep, t_rep)
        if is_training:
            noise_stddev = F.softplus(self.noise(h_rep, t_rep)) + noise_epsilon
            noise_logits = torch.randn(gate_logits.shape).to(gate_logits.device) * noise_stddev
            gate_logits = gate_logits + noise_logits

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_indices = gate_probs.topk(self.topk, dim=-1)
        top_logits = top_logits / top_logits.sum(dim=-1, keepdim=True)

        zeros = torch.zeros_like(gate_probs, requires_grad=True, device=gate_probs.device)
        gates = zeros.scatter(dim=1, index=top_indices, src=top_logits)

        # STATS=====
        if self.more_logging:
            self.cfg.logging(f"{gate_probs }")
        temp = F.one_hot(torch.argmax(gates.detach(), dim=-1), num_classes=gates.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()
        self.stats = self.stats.int()
        # STATS

        row, col = (gates != 0).T.nonzero(as_tuple=True)
        expert_ids, counts = torch.unique_consecutive(row, return_counts=True)
        sample_per_expert = torch.split(col, counts.tolist())

        # dict to dispatch
        exp2sample = dict()
        for index, exp_id in enumerate(expert_ids.tolist()):
            exp2sample[exp_id] = sample_per_expert[index]

        # revert to combine
        revert = list()
        for exp_id in range(self.num_experts):
            if exp_id not in exp2sample:
                revert.append(None)
                continue
            list_sample = exp2sample[exp_id].tolist()
            idx = list(range(len(list_sample)))
            temp = dict(zip(list_sample, idx))
            revert.append(temp)

        # dispatch
        out = list()
        for exp_id in range(self.num_experts):
            if exp_id not in exp2sample:
                out.append(torch.tensor([]))
                continue
            sample_ids = exp2sample[exp_id]
            h = h_rep[sample_ids]
            t = t_rep[sample_ids]
            out.append(self.experts[exp_id](h, t))

        num_sample_per_expert = torch.tensor([0] + [len(i) for i in out]).cumsum(dim=-1)
        out = torch.cat(out, dim=0)

        # combine
        index = list()
        for sample_id, expert_ids in enumerate(top_indices):
            position = list()
            for exid in expert_ids.tolist():
                position.append(num_sample_per_expert[exid] + revert[exid][sample_id])
            index.append(position)
        index = torch.tensor(index)

        temp = out[index]
        res = torch.bmm(top_logits.unsqueeze(1), temp).squeeze(1)

        return res, gates

    def reset_stats(self):
        self.stats = torch.zeros(self.num_experts)

    def set_more_logging(self, value):
        self.more_logging = value
