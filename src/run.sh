export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET="$1"

SAVE_PATH="best.pt"
LOG_PATH="log.txt"
SEED=2004

NUM_EPOCH=30
TRAIN_BATCH_SIZE=4
TEST_BATCH_SIZE=4
UPDATE_FREQ=1
WARMUP_RATIO=0.06
MAX_GRAD_NORM=1.0

NEW_LR=1e-4
PRETRAINED_LR=1.472039003976042e-05
ADAM_EPSILON=1e-6

# DEVICE="cuda:0"
DEVICE="cpu"
TRANSFORMER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEQ_PROCESS_TYPE="sd"

GRAPH_TYPE="rgcn"
NUM_BASES=2
TYPE_DIM=20
LOW_LAYERS=1
HIGH_LAYERS=2

USE_MOE=True
MOE_TYPE="sparse"
NOISE_TYPE="uniform"
NOISE_LIMIT=1
SPARSE_TOPK=1
NUM_EXPERTS=4
NOISE_SCALE=0.5
USE_IMPORTANCE_LOSS=True
IMPORTANCE_WEIGHT=0.1
USE_GATE_LOSS=True
GATEL_WEIGHT=0.1

USE_PSD=True
LOWER_TEMP=2.0
UPPER_TEMP=20.0
LOSS_TRADEOFF=4.999979907145212


USE_SC=False
SC_TEMP=0.16096448806072833
SC_WEIGHT=0.1509642367395748

RE_LOSS="CE" # AT, CE, sigmoidF1, softmaxF1
FOCAL_GAMMA=2.5
PENALTY_WEIGHT=0.01
# BETA=1.0
# ETA=0.0
# T=1.0

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
  --device $DEVICE \
  --transformer $TRANSFORMER \
  --seq_process_type $SEQ_PROCESS_TYPE\
  --graph_type $GRAPH_TYPE\
  --num_bases $NUM_BASES\
  --type_dim $TYPE_DIM \
  --low_layers $LOW_LAYERS \
  --high_layers $HIGH_LAYERS \
  --use_moe $USE_MOE \
  --sparse_topk $SPARSE_TOPK \
  --moe_type $MOE_TYPE \
  --noise_type $NOISE_TYPE \
  --noise_limit $NOISE_LIMIT \
  --num_experts $NUM_EXPERTS \
  --noise_scale $NOISE_SCALE \
  --use_importance_loss $USE_IMPORTANCE_LOSS \
  --importance_weight $IMPORTANCE_WEIGHT \
  --use_gate_loss $USE_GATE_LOSS \
  --gatel_weight $GATEL_WEIGHT \
  --use_psd $USE_PSD \
  --lower_temp $LOWER_TEMP \
  --upper_temp $UPPER_TEMP \
  --loss_tradeoff $LOSS_TRADEOFF \
  --use_sc $USE_SC \
  --sc_temp $SC_TEMP \
  --sc_weight $SC_WEIGHT \
  --focal_gamma $FOCAL_GAMMA \
  --re_loss $RE_LOSS \
  --penalty_weight $PENALTY_WEIGHT


  # --β $BETA \
  # --η $ETA \
  # --T $T \
