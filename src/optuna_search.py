import optuna
import argparse
from main import run_training
from config import Config

def parse_args_from_trial(trial):
    suggested_sc_temp = trial.suggest_float('sc_temp', 0.01, 1.0, log=True)
    suggested_sc_weight = trial.suggest_float('sc_weight', 0.01, 1.0, log=True)
    suggested_trade_off = trial.suggest_float('trade_off', 0.1, 5.0, log=True)

    args = argparse.Namespace()
    args.deter_algo = True

    args.dataset = "cdr"
    args.save_path = "best.pt"
    args.log_path = "log.txt"
    args.seed = 2004

    args.num_epoch = 30
    args.train_batch_size = 4
    args.test_batch_size = 4
    args.update_freq = 1
    args.warmup_ratio = 0.06
    args.max_grad_norm = 1.0

    # Suggest hyperparameters with optuna
    args.new_lr = 1e-4
    args.pretrained_lr = 2e-5
    args.adam_epsilon = 1e-6

    args.device = "cuda:0"
    # args.device = "cpu"
    args.transformer = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    args.seq_process_type = "sd"

    args.graph_type = 'rgcn'
    args.num_bases = 2
    args.type_dim = 20
    args.graph_layers = 3

    args.use_psd = True
    args.lower_temp = 2.0
    args.upper_temp = 20.0
    args.loss_tradeoff = suggested_trade_off

    args.use_sc = True
    args.sc_temp = suggested_sc_temp
    args.sc_weight = suggested_sc_weight

    return args


def objective(trial):
    args = parse_args_from_trial(trial)
    cfg = Config(args)
    val_score = run_training(cfg, no_tqdm=True)
    return val_score

if __name__ == '__main__':
    study = optuna.create_study(
        direction="maximize",
        study_name="tuning",
    )
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial)

