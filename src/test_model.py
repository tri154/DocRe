import argparse
import torch
from config import Config
from model import Model
from tester import Tester
from preprocessing import Preprocessing

def setup():
    args = argparse.Namespace()

    args.dataset = "cdr"
    args.save_path = "best.pt"
    args.log_path = "log.txt"
    args.seed = 1920

    args.num_epoch = 15
    args.train_batch_size = 4
    args.test_batch_size = 8
    args.update_freq = 1
    args.warmup_ratio = 0.06
    args.max_grad_norm = 1.0

    args.new_lr = 8e-5
    args.pretrained_lr = 1.472039003976042e-05
    args.gate_lr = 1e-5
    args.lower_gate_lr=0.0
    args.upper_gate_lr=0.8
    args.adam_epsilon = 1e-6

    args.device = "cuda:0"
    args.transformer = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    args.seq_process_type="sd"

    args.graph_type='rgcn'
    args.num_bases = 2
    args.type_dim = 20
    args.low_layers=1
    args.high_layers=2

    args.use_moe=True
    args.moe_type="sparse"
    args.sparse_topk=1
    args.num_experts=4
    args.noise_scale=0.5
    args.importance_weight=0.5
    args.gatel_weight=0.1
    args.warmup_phase=2
    args.rerouting_epochs=[5, 10]

    args.use_psd = True
    args.lower_temp = 2.0
    args.upper_temp = 20.0
    args.loss_tradeoff = 30.0

    args.use_sc = False
    args.sc_temp = 0.16096448806072833
    args.sc_weight = 0.1509642367395748

    args.re_loss = "CE"
    args.focal_gamma=2.5
    args.penalty_weight=0.01

    args.noise_type=None

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", type=str)
    args = parser.parse_args()

    cfg = Config(setup())

    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
    model = Model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(args.model_pt, map_location=cfg.device))
    tester = Tester(cfg, dev_set=dev_set, test_set=test_set)
    model.bilinear.reset_stats()
    TP, FP, FN, P, R, F1 = tester.test(model, dataset='dev')
    print(f"Stats: {model.bilinear.stats} ")
    print(f"Stats: {model.bilinear.class_count} ")
    print(f"Dev set: P={P}, R={R}, F1={F1}")

    model.bilinear.reset_stats()
    TP, FP, FN, P, R, F1 = tester.test(model, dataset='test')
    print(f"Test set: P={P}, R={R}, F1={F1}")
    print(f"Stats: {model.bilinear.stats} ")
    print(f"Stats: {model.bilinear.class_count} ")
