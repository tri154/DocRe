import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

        # Wrapper
        if self.cfg.re_loss == 'AT':
            self.cal_loss = self.AT_focal_loss
            self.predict = self.AT_pred
        elif self.cfg.re_loss == 'CE':
            self.cal_loss = self.CE_focal_loss
            self.predict = self.CE_pred
        elif self.cfg.re_loss == 'sigmoidF1':
            self.cal_loss = self.sigmoidF1_loss
            self.predict = self.CE_pred
        elif self.cfg.re_loss == 'softmaxF1':
            self.cal_loss = self.softmaxF1_loss_log
            self.predict = self.CE_pred
        else:
            raise Exception("Define loss function.")



    def AT_loss_original(self, logits, labels):
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, self.cfg.id_rel_thre] = 1.0
        labels[:, self.cfg.id_rel_thre] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss


    def AT_focal_loss(self, logits, labels):
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, self.cfg.id_rel_thre] = 1.0
        labels[:, self.cfg.id_rel_thre] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        log_pred_soft1 = F.log_softmax(logit1, dim=-1)
        loss1 = -(torch.pow(1.0 - log_pred_soft1.exp(), self.cfg.focal_gamma) * log_pred_soft1 * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        log_pred_soft2 = F.log_softmax(logit2, dim=-1)
        loss2 = -(torch.pow(1.0 - log_pred_soft2.exp(), self.cfg.focal_gamma) * log_pred_soft2 * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def AT_pred(self, logits):
        th_logit = logits[:, self.cfg.id_rel_thre].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if self.cfg.topk > 0:
            top_v, _ = torch.topk(logits, self.cfg.topk, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def CE_focal_loss(self, logits, labels):
        device = self.cfg.device

        log_probs = F.log_softmax(logits, dim=-1)
        loss = - torch.pow(1.0 - log_probs.exp(), self.cfg.focal_gamma) * log_probs * labels

        counts = labels.sum(dim=0)
        alpha = torch.zeros_like(counts).to(device)
        nonzero_mask = counts != 0
        alpha[nonzero_mask] = 1.0 / counts[nonzero_mask]
        alpha = alpha / alpha.sum()
        alpha = alpha.unsqueeze(0)

        loss = alpha * loss
        loss = loss.sum(-1).mean()

        return loss

    def CE_pred(self, logits):
        pred = torch.argmax(logits, dim=-1)
        one_hot_pred = F.one_hot(pred, num_classes=self.cfg.num_rel).float()
        return one_hot_pred

    def sigmoidF1_loss(self, logits, labels):
        device = self.cfg.device
        β = self.cfg.β
        η = self.cfg.η

        logits = β * (logits + η)
        sig = torch.sigmoid(logits)

        tp = torch.sum(sig * labels, dim=0)
        fp = torch.sum(sig * (1.0 - labels), dim=0)
        fn = torch.sum((1 - sig) * labels, dim=0)
        sigmoid_f1 = (2 * tp) / (2 * tp + fn + fp + self.cfg.small_positive)

        counts = labels.sum(dim=0)
        alpha = torch.zeros_like(counts).to(device)
        nonzero_mask = counts != 0
        alpha[nonzero_mask] = 1.0 / counts[nonzero_mask]
        alpha = alpha / alpha.sum()
        alpha = alpha.unsqueeze(0)

        return 1.0 - torch.sum(sigmoid_f1 * alpha)

    def softmaxF1_loss_log(self, logits, labels):
        device = self.cfg.device
        T = self.cfg.T

        logits = logits / T
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        tp = torch.sum(log_probs * labels, dim=0)
        fp = torch.sum(log_probs * (1.0 - labels), dim=0)
        fn = torch.sum(torch.log(1.0 - probs) * labels, dim=0)
        softmax_f1 = (2 * tp) / (2 * tp + fn + fp + self.cfg.small_positive)

        counts = labels.sum(dim=0)
        alpha = torch.zeros_like(counts).to(device)
        nonzero_mask = counts != 0
        alpha[nonzero_mask] = 1.0 / counts[nonzero_mask]
        alpha = alpha / alpha.sum()
        alpha = alpha.unsqueeze(0)

        return 1.0 - torch.sum(softmax_f1 * alpha)

    def softmaxF1_loss(self, logits, labels):
        device = self.cfg.device
        T = self.cfg.T

        logits = logits / T
        probs = F.softmax(logits, dim=-1)
        tp = torch.sum(probs* labels, dim=0)
        fp = torch.sum(probs * (1.0 - labels), dim=0)
        fn = torch.sum((1.0 - probs) * labels, dim=0)
        softmax_f1 = (2 * tp) / (2 * tp + fn + fp + self.cfg.small_positive)

        counts = labels.sum(dim=0)
        alpha = torch.zeros_like(counts).to(device)
        nonzero_mask = counts != 0
        alpha[nonzero_mask] = 1.0 / counts[nonzero_mask]
        alpha = alpha / alpha.sum()
        alpha = alpha.unsqueeze(0)

        return 1.0 - torch.sum(softmax_f1 * alpha)

    def PSD_loss(self, logits, teacher_logits, current_epoch):
        current_temp = self.cfg.upper_temp - (self.cfg.upper_temp - self.cfg.lower_temp) * current_epoch / (self.cfg.num_epoch - 1.0)
        current_tradeoff = self.cfg.loss_tradeoff * current_epoch / (self.cfg.num_epoch - 1.0)
        loss = self.kd_loss(F.log_softmax(logits / current_temp, dim=1),
                            F.softmax(teacher_logits / current_temp, dim=1))

        return loss, current_tradeoff


    def SC_loss(self, reps, oh_labels):
        '''
            A new loss function, that only works for single label.
        '''

        ####
        # DEBUG
        # labels = torch.ones(25)
        # labels[5] = 2
        # labels[10] = 3
        # reps = torch.rand(25, 50)
        # n_sample = len(reps)
        ####
        device = self.cfg.device
        reps = F.normalize(reps, p=2, dim=1)
        n_sample = len(reps)
        labels = oh_labels.argmax(dim=-1)
        uniques, counts = torch.unique(labels,return_counts=True)
        val2count = dict(zip(uniques.tolist(), counts.tolist()))
        pairs = torch.triu_indices(n_sample, n_sample, offset=1).to(device)

        anchor_values = labels[pairs[0]].to(device)
        candidate_values = labels[pairs[1]].to(device)
        mask = (anchor_values == candidate_values)

        has_1 = (counts == 1).nonzero().squeeze(-1).to(device)
        has_1 = uniques[has_1].to(device) # unique labels that have only 1 sample.
        if len(has_1) != 0:
            mask = mask & ~torch.isin(anchor_values, has_1)
        pairs = pairs[:, mask]
        if pairs.shape[-1] == 0: return 0
        anchor_values = anchor_values[mask].cpu()
        anchor_values.apply_(val2count.get)
        anchor_values = anchor_values.to(device)

        numerator = torch.exp(torch.sum(reps[pairs[0]] * reps[pairs[1]], dim=-1) / self.cfg.sc_temp).to(device)

        unique_anchor_idx = pairs[0].unique().to(device) # 22

        temp = reps[unique_anchor_idx] # 22, 50

        cached = torch.matmul(temp, reps.T)
        cached = torch.exp(cached / self.cfg.sc_temp)
        own = cached[torch.arange(len(unique_anchor_idx)), unique_anchor_idx] # 22
        cached = torch.sum(cached, dim=1) - own # 22

        cached = {int(i.item()): cached[idx] for idx, i in enumerate(unique_anchor_idx)}

        denominator = torch.stack([cached[k.item()] for k in pairs[0]]).to(device)

        loss = torch.log(numerator / (denominator + 1e-6)) * (-1 / (anchor_values - 1))
        loss = loss.sum()
        return loss
