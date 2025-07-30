export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET="$1"

SAVE_PATH="checkpoint"
LOAD_PATH=None
LOG_PATH="log.txt"
SEED=2004

NUM_EPOCH=30
TRAIN_BATCH_SIZE=4
TEST_BATCH_SIZE=4
UPDATE_FREQ=1
WARMUP_RATIO=0.06
MAX_GRAD_NORM=1.0

NEW_LR=1e-4
PRETRAINED_LR=2e-5
ADAM_EPSILON=1e-6

DEVICE="cuda:0"
# DEVICE="cpu"
TRANSFORMER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEQ_PROCESS_TYPE="sd"

GRAPH_TYPE='rgat'
NUM_BASES=2
TYPE_DIM=20
GRAPH_LAYERS=5 #NOTE min=2

USE_PSD=True
LOWER_TEMP=2.0
UPPER_TEMP=20.0
LOSS_TRADEOFF=1.5505155156932497


USE_SC=True
SC_TEMP=0.8799535385474245
SC_WEIGHT=0.13922932736312713

python main.py \
  --dataset $DATASET \
  --save_path $SAVE_PATH \
  # --load_path $LOAD_PATH \
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
  --device $DEVICE \
  --transformer $TRANSFORMER \
  --seq_process_type $SEQ_PROCESS_TYPE\
  --graph_type $GRAPH_TYPE\
  --num_bases $NUM_BASES\
  --type_dim $TYPE_DIM \
  --graph_layers $GRAPH_LAYERS \
  --use_psd $USE_PSD \
  --lower_temp $LOWER_TEMP \
  --upper_temp $UPPER_TEMP \
  --loss_tradeoff $LOSS_TRADEOFF \
  --use_sc $USE_SC \
  --sc_temp $SC_TEMP \
  --sc_weight $SC_WEIGHT
