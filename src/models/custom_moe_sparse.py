import torch
import torch.nn as nn
import torch.nn.functional as F
from models.expert import Expert

class CustomMoeSparse(nn.Module):

    def __init__(self, cfg, in1_features, in2_features, out_features):
        super().__init__()
        self.cfg = cfg
        self.num_experts = self.cfg.num_experts
        self.topk = self.cfg.topk
        if self.topk > 1: raise Exception("currently support topk=1")

        self.more_logging = False
        self.stats = torch.zeros(self.num_experts)

        self.experts = nn.ModuleList([
            Expert(in1_features, in2_features, out_features) for _ in range(self.num_experts)
        ])

        self.gate = nn.Bilinear(in1_features, in2_features, self.num_experts)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        if self.cfg.noise_type == 'normal':
            self.noise = nn.Bilinear(in1_features, in2_features, self.num_experts)
            nn.init.zeros_(self.noise.weight)
            nn.init.zeros_(self.noise.bias)

    def __add_noise(self, h_rep, t_rep, gate_logits, cur_epoch=None):
        if self.cfg.noise_type == 'normal':
            return self.__add_trainable_normal_noise(h_rep, t_rep, gate_logits)
        elif self.cfg.noise_type == 'uniform':
            return self.__add_uniform_noise(h_rep, t_rep , gate_logits)
        else:
            raise Exception("not a noise type.")

    def forward(self, h_rep ,t_rep, labels, is_training=True, cur_epoch=None):
        gate_logits = self.gate(h_rep, t_rep)
        if is_training and self.cfg.noise_type is not None:
            gate_logits = self.__add_noise(h_rep, t_rep, gate_logits, cur_epoch=cur_epoch)

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_indices = gate_probs.topk(self.topk, dim=-1)

        zeros = torch.zeros_like(gate_probs, requires_grad=True, device=gate_probs.device)
        gates = zeros.scatter(dim=1, index=top_indices, src=top_logits)

        if self.more_logging:
            self.cfg.another_logging(f"{gate_probs }")
        temp = F.one_hot(torch.argmax(gates.detach(), dim=-1), num_classes=gates.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()

        n_sample = h_rep.shape[0]
        out = list()
        for sample_id in range(n_sample):
            expert_ids = top_indices[sample_id]
            sample_out = torch.stack([self.experts[i](h_rep[sample_id], t_rep[sample_id]) for i in expert_ids])

            sample_gate = top_logits[sample_id]
            sample_gate = sample_gate.detach() # gate weight not involved in main loss.
            sample_gate = sample_gate / sample_gate.sum(dim=-1, keepdim=True)

            sample_out = sample_gate @ sample_out
            out.append(sample_out)
        out = torch.stack(out)

        gate_loss = 0.0
        if is_training and self.cfg.use_gate_loss:
            pred = torch.argmax(out, dim=-1)
            labels = torch.argmax(labels, dim=-1)
            mask = pred == labels

            positive = top_logits[mask]
            positive_loss = - torch.log(positive).sum()

            negative_indices = torch.nonzero(~mask).squeeze(-1)
            negative_hrep = h_rep[negative_indices]
            negative_trep = t_rep[negative_indices]
            negative_labels = labels[negative_indices]
            res = torch.stack([self.experts[idx](negative_hrep, negative_trep) for idx in range(self.num_experts)], dim=1)
            res = torch.softmax(res, dim=-1)

            idx = negative_labels.view(-1, 1, 1).expand(-1, self.num_experts, 1)
            temp = torch.gather(res, dim=2, index=idx).squeeze(-1)
            # _, indices = torch.topk(temp, dim=-1, k=self.topk)
            # indices = indices.squeeze(-1)
            indices = torch.argmax(temp, dim=-1)

            negative = gate_probs[negative_indices, indices]
            negative_loss = - torch.log(negative).sum()

            gate_loss = positive_loss + negative_loss

        return out, gates, gate_loss

    def __add_uniform_noise(self, h_rep, t_rep, gate_logits, noise_epsilon=1e-2):
        noise_logits = torch.rand(gate_logits.shape).to(gate_logits.device)
        return gate_logits + noise_epsilon * noise_logits

    def __add_trainable_normal_noise(self, h_rep, t_rep, gate_logits, noise_epsilon=1e-2, cur_epoch=None):
        noise_stddev = F.softplus(self.noise(h_rep, t_rep)) + noise_epsilon
        noise_logits = torch.randn(gate_logits.shape).to(gate_logits.device) * noise_stddev
        return gate_logits + noise_logits

    def reset_stats(self):
        self.stats = torch.zeros(self.num_experts)

    def set_more_logging(self, value):
        self.more_logging = value

    def toggle_gate_weight(self, value):
        for param in self.gate.parameters():
            param.requires_grad = value
