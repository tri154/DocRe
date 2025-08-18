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


    def reset_stats(self):
        self.stats = torch.zeros(self.num_experts)

    def set_more_logging(self, value):
        self.more_logging = value



    def forward_old(self, h_rep, t_rep):
        self.clean_forward(h_rep, t_rep)
        input("STTTTOOPPP")
        # +===============
        device = h_rep.device

        # normal noise
        gate_logits = self.gate(h_rep, t_rep)
        noise_logits = torch.randn(gate_logits.shape).to(device) * F.softplus(self.noise(h_rep, t_rep))
        gate_logits = gate_logits + noise_logits

        # uniform noise
        # noise = torch.rand_like(gate_logits) * self.noise_scale
        # noise = (torch.rand_like(gate_logits) - 0.5) * 2 * self.noise_scale
        # gate_out = F.softmax(gate_logits, dim=1)
        # gate_out = gate_out + noise
        # gate_out = gate_out / gate_out.sum(dim=-1, keepdim=True)

        # logging
        if self.more_logging:
            self.cfg.logging(f"{F.softmax(gate_logits)} ")
        temp = F.one_hot(torch.argmax(gate_logits.detach(), dim=-1), num_classes=gate_logits.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()
        # end logging

        gate_topk, indices = torch.topk(gate_logits, self.topk, dim=-1)
        gate_out = F.softmax(gate_topk, dim=1) # nsample, topk

        n_sample = h_rep.shape[0]
        out = list()
        for sample_id in range(n_sample):
            expert_ids = indices[sample_id]
            sample_gate = gate_out[sample_id]
            sample_out = torch.stack([self.experts[i](h_rep[sample_id], t_rep[sample_id]) for i in expert_ids])
            out.append(sample_gate @ sample_out)

        out = torch.stack(out)
        print(gate_out.shape)
        print(gate_out)
        input()
        # todo: gate_out modify ???
        return out, gate_out

    def forward(self, h_rep ,t_rep, is_training=True, noise_epsilon=1e-2):
        # TODO: implement sparse moe with importance loss.
        # NOTE: add load balance loss, if needed.
        gate_logits = self.gate(h_rep, t_rep)
        if is_training:
            noise_stddev = F.softplus(self.noise(h_rep, t_rep)) + noise_epsilon
            noise_logits = torch.randn(gate_logits.shape).to(gate_logits.device) * noise_stddev
            gate_logits = gate_logits + noise_logits

        gate_probs = F.softmax(gate_logits)
        top_logits, top_indices = gate_probs.topk(self.topk, dim=-1)
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
