import torch
import torch.nn as nn
import torch.nn.functional as F
from models.expert import Expert

class CustomMoeSparse(nn.Module):

    def __init__(self, cfg, in_features, out_features):
        super().__init__()
        self.cfg = cfg
        self.num_experts = self.cfg.num_experts
        self.topk = self.cfg.topk
        if self.topk > 1: raise Exception("currently support topk=1")

        self.more_logging = False
        self.stats = torch.zeros(self.num_experts)

        self.experts = nn.ModuleList([
            Expert(in_features, out_features) for _ in range(self.num_experts)
        ])

        self.gate = nn.Linear(in_features, self.num_experts)

        # nn.init.xavier_uniform_(self.gate.weight)
        # nn.init.zeros_(self.gate.weight)
        # nn.init.zeros_(self.gate.bias)

        if self.cfg.noise_type == 'normal':
            self.noise = nn.Linear(in_features, self.num_experts)
            nn.init.zeros_(self.noise.weight)
            nn.init.zeros_(self.noise.bias)

    def __add_noise(self, pair_reps, gate_logits, cur_epoch=None):
        if self.cfg.noise_type == 'normal':
            return self.__add_trainable_normal_noise(pair_reps , gate_logits)
        elif self.cfg.noise_type == 'uniform':
            return self.__add_uniform_noise(pair_reps , gate_logits)
        else:
            raise Exception("not a noise type.")


    def forward(self, pair_reps , labels, is_training=True, cur_epoch=None):
        gate_logits = self.gate(pair_reps)
        if is_training and self.cfg.noise_type is not None:
            gate_logits = self.__add_noise(pair_reps , gate_logits, cur_epoch=cur_epoch)

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_indices = gate_probs.topk(self.topk, dim=-1)

        # zeros = torch.zeros_like(gate_probs, requires_grad=True, device=gate_probs.device)
        # gates = zeros.scatter(dim=1, index=top_indices, src=top_logits)

        if self.more_logging:
            self.cfg.another_logging(f"{gate_probs }")
        temp = F.one_hot(torch.argmax(gate_probs.detach(), dim=-1), num_classes=gate_probs.shape[-1]).int()
        self.stats = self.stats.cpu() + temp.sum(dim=0).cpu()

        out = self.forward_not_dispatch(pair_reps, top_indices, top_logits)
        # out = self.forward_dispatch(h_rep, t_rep, top_indices, top_logits)

        gate_loss = 0.0
        if is_training and self.cfg.use_gate_loss:
           gate_loss = self.compute_gate_loss(pair_reps, out, labels, top_logits, top_indices, gate_probs)

        return out, gate_probs, gate_loss


    def compute_gate_loss(self, pair_reps, out, labels, top_logits, top_indices, gate_probs):
        count = torch.zeros(self.num_experts, device=pair_reps.device)

        # positive loss
        pred = torch.argmax(out, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        positive_indices = pred == labels

        positive = top_logits[positive_indices]

        pos_gate_idx = top_indices[positive_indices].squeeze(-1)
        count_pos_id = F.one_hot(pos_gate_idx, num_classes=self.num_experts).sum(dim=0)
        count = count + count_pos_id

        positive_loss = - torch.log(positive).squeeze(-1)

        # negative loss
        negative_indices = torch.nonzero(~positive_indices).squeeze(-1)
        negative_pair_reps = pair_reps[negative_indices]
        negative_labels = labels[negative_indices]
        res = torch.stack([self.experts[idx](negative_pair_reps) for idx in range(self.num_experts)], dim=1)
        res = torch.softmax(res, dim=-1)

        idx = negative_labels.view(-1, 1, 1).expand(-1, self.num_experts, 1)
        temp = torch.gather(res, dim=2, index=idx).squeeze(-1)
        neg_gate_idx = torch.argmax(temp, dim=-1)

        count_neg_id = F.one_hot(neg_gate_idx, num_classes=self.num_experts).sum(dim=0)
        count = count + count_neg_id

        negative = gate_probs[negative_indices, neg_gate_idx]
        negative_loss = - torch.log(negative)

        # scale
        alpha = torch.zeros_like(count).to(pair_reps.device)
        nonzero_mask = count != 0
        alpha[nonzero_mask] = 1.0 / count[nonzero_mask]

        positive_loss = positive_loss * alpha[pos_gate_idx]
        negative_loss = negative_loss * alpha[neg_gate_idx]

        # final loss
        gate_loss = positive_loss.sum(dim=0) + negative_loss.sum(dim=0)

        return gate_loss


    def forward_not_dispatch(self, pair_reps, top_indices, top_logits):
        n_sample = pair_reps.shape[0]
        out = list()
        for sample_id in range(n_sample):
            expert_ids = top_indices[sample_id]
            sample_out = torch.stack([self.experts[i](pair_reps[sample_id]) for i in expert_ids])

            sample_gate = top_logits[sample_id]
            sample_gate = sample_gate.detach() # gate weight not involved in main loss.
            sample_gate = sample_gate / sample_gate.sum(dim=-1, keepdim=True)

            sample_out = sample_gate @ sample_out
            out.append(sample_out)
        out = torch.stack(out)
        return out

    def forward_dispatch(self, h_rep, t_rep, top_indices, top_logits):
        n_sample = h_rep.shape[0]
        device = h_rep.device

        zeros = torch.zeros((n_sample, self.num_experts), device=device)
        gates = zeros.scatter(dim=1, index=top_indices, src=top_logits)

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
                out.append(torch.tensor([], device=device))
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
        top_logits_norm = top_logits.detach()
        top_logits_norm = top_logits_norm / top_logits_norm.sum(dim=-1, keepdim=True)
        res = torch.bmm(top_logits_norm.unsqueeze(1), temp).squeeze(1)
        return res

    def __add_uniform_noise(self, pair_reps, gate_logits, noise_epsilon=0.1):
        noise_logits = torch.rand_like(gate_logits).to(gate_logits.device)
        return gate_logits + noise_epsilon * noise_logits

    def __add_trainable_normal_noise(self, pair_reps, gate_logits, noise_epsilon=1e-2, cur_epoch=None):
        noise_stddev = F.softplus(self.noise(pair_reps)) + noise_epsilon
        noise_logits = torch.randn(gate_logits.shape).to(gate_logits.device) * noise_stddev
        return gate_logits + noise_logits

    def reset_stats(self):
        self.stats = torch.zeros(self.num_experts)

    def set_more_logging(self, value):
        self.more_logging = value

    def toggle_gate_weight(self, value):
        for param in self.gate.parameters():
            param.requires_grad = value
