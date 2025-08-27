export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATASET="cdr"

SAVE_PATH_BASE="outputs"
LOG_PATH_BASE="logs"

NUM_EPOCH=15
TRAIN_BATCH_SIZE=4
TEST_BATCH_SIZE=4
UPDATE_FREQ=1
WARMUP_RATIO=0.06
MAX_GRAD_NORM=1.0

NEW_LR=8e-5
PRETRAINED_LR=1.472039003976042e-05
ADAM_EPSILON=1e-6
PATIENCE=-1

DEVICE="cuda:0"
TRANSFORMER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEQ_PROCESS_TYPE="sd"

GRAPH_TYPE='rgcn'
NUM_BASES=2
TYPE_DIM=20
LOW_LAYERS=1
HIGH_LAYERS=2

USE_PSD=True
LOWER_TEMP=2.0
UPPER_TEMP=20.0
LOSS_TRADEOFF=4.999979907145212

USE_SC=False
SC_TEMP=0.5824951879607512
SC_WEIGHT=0.3108995767362708

RE_LOSS="CE"
FOCAL_GAMMA=2.5
BETA=1.0
ETA=0.0
T=1.0
PENALTY_WEIGHT=0.01

# SEEDS=(2004 1945 30 4 1975)
SEEDS=(1920 1942 1975 1969 2004)

for SEED in "${SEEDS[@]}"; do
  echo "==============================="
  echo "Running seed $SEED"
  echo "==============================="

  SAVE_PATH="${SAVE_PATH_BASE}/model_seed_${SEED}.pt"
  LOG_PATH="${LOG_PATH_BASE}/log_seed_${SEED}.txt"

  mkdir -p "$SAVE_PATH_BASE"
  mkdir -p "$LOG_PATH_BASE"

  python main.py \
    --dataset $DATASET \
    --save_path $SAVE_PATH \
    --log_path $LOG_PATH \
    --seed $SEED \
    --num_epoch $NUM_EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --update_freq $UPDATE_FREQ \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --new_lr $NEW_LR \
    --pretrained_lr $PRETRAINED_LR \
    --adam_epsilon $ADAM_EPSILON \
  	--patience $PATIENCE \
    --device $DEVICE \
    --transformer $TRANSFORMER \
    --seq_process_type $SEQ_PROCESS_TYPE\
    --graph_type $GRAPH_TYPE\
    --num_bases $NUM_BASES\
    --type_dim $TYPE_DIM \
    --low_layers $LOW_LAYERS \
    --high_layers $HIGH_LAYERS \
    --use_psd $USE_PSD \
    --lower_temp $LOWER_TEMP \
    --upper_temp $UPPER_TEMP \
    --loss_tradeoff $LOSS_TRADEOFF \
    --use_sc $USE_SC \
    --sc_temp $SC_TEMP \
    --sc_weight $SC_WEIGHT \
    --focal_gamma $FOCAL_GAMMA \
    --re_loss $RE_LOSS \
    --β $BETA \
    --η $ETA \
    --T $T \
    --penalty_weight $PENALTY_WEIGHT
done
