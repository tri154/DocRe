import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import math
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW


class Trainer:
    def __init__(self, cfg, model, train_set, tester):
        self.cfg = cfg
        self.model = model
        self.train_set = train_set
        self.tester = tester
        self.cur_epoch = 0
        self.warmup_phase = self.cfg.warmup_phase

        self.opt_main, self.sched_main, self.opt_gate, self.sched_gate = self.prepare_optimizer_scheduler()


    # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
    #             'doc_title': doc_title,
    #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
    #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.


    def prepare_optimizer_scheduler(self):
        grouped_params = defaultdict(list)
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                grouped_params['pretrained_lr'].append(param)
            elif 'gate' in name:
                grouped_params['gate_lr'].append(param)
            else:
                grouped_params['new_lr'].append(param)

        grouped_lrs_main = [{'params': grouped_params[group], 'lr': lr} for group, lr in zip(['pretrained_lr', 'new_lr'], [self.cfg.pretrained_lr, self.cfg.new_lr])]
        opt_main = AdamW(grouped_lrs_main, eps=self.cfg.adam_epsilon)

        num_updates = math.ceil(math.ceil(len(self.train_set) / self.cfg.train_batch_size) / self.cfg.update_freq) * self.cfg.num_epoch
        num_warmups = int(num_updates * self.cfg.warmup_ratio)
        sched_main = get_linear_schedule_with_warmup(opt_main, num_warmups, num_updates)

        grouped_lrs_gate = [{'params': grouped_params[group], 'lr': lr} for group, lr in zip(['gate_lr'], [self.cfg.gate_lr])]
        opt_gate = AdamW(grouped_lrs_gate, eps=self.cfg.adam_epsilon)

        # modify logic here if modify training phase logic.
        num_updates = math.ceil(math.ceil(len(self.train_set) / self.cfg.train_batch_size) / self.cfg.update_freq) * (self.cfg.num_epoch - (self.warmup_phase - 1))
        sched_gate = get_linear_schedule_with_warmup(opt_gate, num_warmups, num_updates)

        return opt_main, sched_main, opt_gate, sched_gate


    def prepare_batch(self, batch_size):
        inputs = self.train_set

        num_batch = math.ceil(len(inputs) / batch_size)
        device = self.cfg.device

        for idx_batch in range(num_batch):
            indicies = (idx_batch * batch_size, min((idx_batch + 1) * batch_size, len(inputs)))

            batch_inputs = inputs[indicies[0]:indicies[1]]
            cur_batch_size = len(batch_inputs)

            batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
            batch_titles = list()
            batch_start_mpos = list()
            batch_epair_rels = list()
            batch_sent_pos = list()
            num_sent_per_doc = list()
            batch_mpos2sid = list()
            batch_eid2sid = list()
            batch_mentions_link = list()
            batch_ents_link = list()
            batch_teacher_logits = list()

            for doc_input in batch_inputs:
                batch_titles.append(doc_input['doc_title'])
                batch_token_seqs.append(doc_input['doc_tokens'])
                batch_start_mpos.append(doc_input['doc_start_mpos'])
                batch_sent_pos.append(doc_input['doc_sent_pos'])
                batch_mpos2sid.append(doc_input['doc_mpos2sid'])
                batch_eid2sid.append(doc_input['doc_eid2sid'])
                batch_ents_link.append(doc_input['doc_ents_link'])
                batch_mentions_link.append(doc_input['doc_mentions_link'])
                num_sent_per_doc.append(len(doc_input['doc_sent_pos']))

                if 'teacher_logits' in doc_input:
                    batch_teacher_logits.append(doc_input['teacher_logits'])

                doc_seqs_len = doc_input['doc_tokens'].shape[0]
                batch_token_masks.append(torch.ones(doc_seqs_len))
                doc_tokens_types = torch.zeros(doc_seqs_len)

                for sid in range(len(doc_input['doc_sent_pos'])):
                    start, end = doc_input['doc_sent_pos'][sid][0], doc_input['doc_sent_pos'][sid][1]
                    doc_tokens_types[start:end] = sid % 2
                batch_token_types.append(doc_tokens_types)

                batch_epair_rels.append(doc_input['doc_epair_rels'])

            batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long()
            batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float()
            batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long()

            max_m_n_p_b = max([len(mention_pos) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()])  # max mention number in batch.

            # num_entity_per_doc = torch.tensor([len(doc_start_mpos.values()) for doc_start_mpos in batch_start_mpos]) # Keep it on CPU.
            # batch_start_mpos = torch.stack([F.pad(torch.tensor(list(mention_pos)), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()]).to(self.cfg.device)

            temp = []
            num_entity_per_doc = []
            num_mention_per_doc = [0 for _ in range(cur_batch_size)]
            num_mention_per_entity = []
            for did, doc_start_mpos in enumerate(batch_start_mpos):
                num_entity_per_doc.append(len(doc_start_mpos.values()))
                for eid in sorted(doc_start_mpos): # Keep entity order.
                    mention_pos = doc_start_mpos[eid]
                    num_mention = len(mention_pos)
                    num_mention_per_entity.append(num_mention)
                    num_mention_per_doc[did] += num_mention
                    temp.append(F.pad(torch.tensor(sorted(list(mention_pos))), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1)) # Keep mention order

            batch_start_mpos = torch.stack(temp) #expecting: [sum entity, max_mention]
            num_entity_per_doc = torch.tensor(num_entity_per_doc)
            num_mention_per_doc = torch.tensor(num_mention_per_doc)
            num_sent_per_doc = torch.tensor(num_sent_per_doc)
            num_mention_per_entity = torch.tensor(num_mention_per_entity)
            batch_mpos2sid = torch.cat(batch_mpos2sid, dim=0)
            batch_teacher_logits = torch.cat(batch_teacher_logits, dim=0) if len(batch_teacher_logits) != 0 else None

            num_mentlink_per_doc = torch.tensor([ts.shape[-1] for ts in batch_mentions_link])
            batch_mentions_link = torch.cat(batch_mentions_link, dim=-1)
            num_entlink_per_doc = torch.tensor([ts.shape[-1] for ts in batch_ents_link])
            batch_ents_link = torch.cat(batch_ents_link, dim=-1)

            yield { 'indices': indicies,
                    'batch_titles': np.array(batch_titles),
                    'batch_epair_rels': batch_epair_rels,
                    'batch_sent_pos': batch_sent_pos,
                    'batch_eid2sid': batch_eid2sid,
                    'num_sent_per_doc': num_sent_per_doc.cpu(),
                    'num_entity_per_doc': num_entity_per_doc.cpu(),
                    'num_mention_per_doc': num_mention_per_doc.cpu(),
                    'num_mentlink_per_doc': num_mentlink_per_doc.cpu(),
                    'num_entlink_per_doc': num_entlink_per_doc.cpu(),
                    'num_mention_per_entity': num_mention_per_entity.to(device),
                    'batch_token_seqs': batch_token_seqs.to(device),
                    'batch_token_masks': batch_token_masks.to(device),
                    'batch_token_types': batch_token_types.to(device),
                    'batch_start_mpos': batch_start_mpos.to(device),
                    'batch_mpos2sid': batch_mpos2sid.to(device),
                    'batch_mentions_link': batch_mentions_link.to(device),
                    'batch_ents_link': batch_ents_link.to(device),
                    'batch_teacher_logits': batch_teacher_logits.to(device) if batch_teacher_logits is not None else None,
                    }

    def debug(self):
        for batch_input in self.prepare_batch(self.cfg.train_batch_size):
            self.model.bilinear.reset_stats()
            loss, _ = self.model(batch_input, is_training=True, current_epoch=0)
            print(f"Stats: {self.model.bilinear.stats} ")
            self.cfg.logging(f"Stats: {self.model.bilinear.stats} ")
            print(loss)
            # preds, labels = self.model(batch_input, is_training=False)
            # print(preds)
            input("Stop")

    def PSD_add_logits(self, batch_logits, indicies):
        for did, doc_idx in enumerate(range(indicies[0], indicies[1])):
            self.train_set[doc_idx]['teacher_logits'] = batch_logits[did]

    # new implement
    def train_one_epoch(self, current_epoch, batch_size):
        self.model.train()
        self.opt_main.zero_grad()
        self.opt_gate.zero_grad()

        if current_epoch < self.warmup_phase: self.prepare_warmup()
        else:                                 self.prepare_fitting()

        np.random.shuffle(self.train_set)

        num_batch = math.ceil(len(self.train_set) / batch_size)

        total_loss = 0.0
        for idx_batch, batch_input in enumerate(self.prepare_batch(batch_size)):
            batch_loss, batch_logits = self.model(batch_input, current_epoch=current_epoch, is_training=True)
            # =======================
            # print(batch_loss)
            # input("debug")
            # break
            # =======================
            if self.cfg.use_psd:
                self.PSD_add_logits(batch_logits, batch_input['indices'])
            total_loss += batch_loss.item()
            (batch_loss / self.cfg.update_freq).backward()

            if idx_batch % self.cfg.update_freq == 0 or idx_batch == num_batch - 1:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt_main.step()
                self.opt_main.zero_grad()
                self.opt_gate.zero_grad()
                self.sched_main.step()

        # logging
        self.cfg.logging(f"Stats train (before rerouting): {self.model.bilinear.stats} ", is_printed=True)
        self.model.bilinear.reset_stats()
        # logging

        # warmup phase doesn't have rerouting, only the last warmup epoch or in interval epoch.
        is_warmup = current_epoch < self.warmup_phase - 1
        is_rerouting = (current_epoch == self.warmup_phase - 1) or (current_epoch % self.cfg.rerouting_interval == 0)

        if is_warmup or not is_rerouting:
            return total_loss

        self.prepare_rerouting()
        if current_epoch == self.warmup_phase - 1:
            self.cfg.noise_type = 'uniform'

        for idx_batch, batch_input in enumerate(self.prepare_batch(batch_size)):
            batch_loss, batch_logits = self.model(batch_input, current_epoch=current_epoch, is_training=True)
            (batch_loss / self.cfg.update_freq).backward()

            if idx_batch % self.cfg.update_freq == 0 or idx_batch == num_batch - 1:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt_gate.step()
                self.opt_gate.zero_grad()
                self.opt_main.zero_grad()
                self.sched_gate.step()
        # logging
        self.cfg.logging(f"Stats train (after rerouting): {self.model.bilinear.stats} ", is_printed=True)
        self.model.bilinear.reset_stats()
        # logging

        return total_loss

    def prepare_warmup(self):
        # self.cfg.noise_limit = 1
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        self.cfg.noise_type = None
        self.cfg.use_importance_loss = True
        self.cfg.use_gate_loss = False
        self.cfg.use_sc = False

    def prepare_fitting(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        self.model.bilinear.toggle_gate_weight(False)
        self.cfg.noise_type = None
        self.cfg.use_importance_loss = False
        self.cfg.use_gate_loss = False
        self.cfg.use_sc = True

    def prepare_rerouting(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.bilinear.toggle_gate_weight(True)
        self.cfg.noise_type = None
        self.cfg.use_importance_loss = False
        self.cfg.use_gate_loss = True
        self.cfg.use_sc = False

    def train(self, num_epoches, batch_size, train_set=None):
        if train_set is not None:
            self.train_set = train_set

        self.best_f1_dev = 0
        for idx_epoch in range(num_epoches):
            self.cfg.logging(f'epoch {idx_epoch}/{num_epoches} ' + '=' * 100, is_printed=True)

            self.model.bilinear.reset_stats()
            epoch_loss = self.train_one_epoch(idx_epoch, batch_size)

            # self.cfg.logging(f"Stats train: {self.model.bilinear.stats} ", is_printed=True)

            self.model.bilinear.reset_stats()
            d_tp, d_fp, d_fn, d_presicion, d_recall, d_f1 = self.tester.test(self.model, dataset='dev')
            self.cfg.logging(f"Stats dev: {self.model.bilinear.stats} ", is_printed=True)
            self.cfg.logging(f"epoch: {idx_epoch}, Dev result : loss={epoch_loss}, TP={d_tp}, FP={d_fp}, FN={d_fn}, P={d_presicion:.10f}, R={d_recall:.10f}, F1={d_f1:.10f}.", is_printed=True)

            if d_f1 > self.best_f1_dev:
                self.best_f1_dev = d_f1
                torch.save(self.model.state_dict(), self.cfg.save_path)

            self.cur_epoch += 1

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        self.model.bilinear.reset_stats()
        self.model.bilinear.set_more_logging(True)
        t_tp, t_fp, t_fn, self.precision_test, self.recall_test, self.f1_test = self.tester.test(self.model, dataset='test')
        self.model.bilinear.set_more_logging(False)
        self.cfg.logging(f"Stats test: {self.model.bilinear.stats} ", is_printed=True)
        self.cfg.logging(f"Test result: TP={t_tp}, FP={t_fp}, FN={t_fn}, P={self.precision_test:.10f}, R={self.recall_test:.10f}, F1={self.f1_test:.10f}", is_printed=True)

        return self.best_f1_dev

    # end new

    def train_one_epoch_old(self, current_epoch, batch_size):
        self.model.train()
        self.opt.zero_grad()

        np.random.shuffle(self.train_set)

        num_batch = math.ceil(len(self.train_set) / batch_size)


        total_loss = 0.0
        for idx_batch, batch_input in enumerate(self.prepare_batch(batch_size)):
            batch_loss, batch_logits = self.model(batch_input, current_epoch=current_epoch, is_training=True)
            if self.cfg.use_psd:
                self.PSD_add_logits(batch_logits, batch_input['indices'])
            total_loss += batch_loss.item()
            (batch_loss / self.cfg.update_freq).backward()

            if idx_batch % self.cfg.update_freq == 0 or idx_batch == num_batch - 1:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()
        return total_loss


    def train_old(self, num_epoches, batch_size, train_set=None):
        if train_set is not None:
            self.train_set = train_set

        self.best_f1_dev = 0
        for idx_epoch in range(num_epoches):
            self.cfg.logging(f'epoch {idx_epoch}/{num_epoches} ' + '=' * 100, is_printed=True)

            self.model.bilinear.reset_stats()
            epoch_loss = self.train_one_epoch(idx_epoch, batch_size)
            self.cfg.logging(f"Stats train: {self.model.bilinear.stats} ", is_printed=True)

            self.model.bilinear.reset_stats()
            d_tp, d_fp, d_fn, d_presicion, d_recall, d_f1 = self.tester.test(self.model, dataset='dev')
            self.cfg.logging(f"Stats dev: {self.model.bilinear.stats} ", is_printed=True)
            self.cfg.logging(f"epoch: {idx_epoch}, Dev result : loss={epoch_loss}, TP={d_tp}, FP={d_fp}, FN={d_fn}, P={d_presicion:.10f}, R={d_recall:.10f}, F1={d_f1:.10f}.", is_printed=True)

            if d_f1 > self.best_f1_dev:
                self.best_f1_dev = d_f1
                torch.save(self.model.state_dict(), self.cfg.save_path)

            self.cur_epoch += 1

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        self.model.bilinear.reset_stats()
        # self.model.bilinear.set_more_logging(True)
        t_tp, t_fp, t_fn, self.precision_test, self.recall_test, self.f1_test = self.tester.test(self.model, dataset='test')
        # self.model.bilinear.set_more_logging(False)
        self.cfg.logging(f"Stats test: {self.model.bilinear.stats} ", is_printed=True)
        self.cfg.logging(f"Test result: TP={t_tp}, FP={t_fp}, FN={t_fn}, P={self.precision_test:.10f}, R={self.recall_test:.10f}, F1={self.f1_test:.10f}", is_printed=True)

        return self.best_f1_dev
