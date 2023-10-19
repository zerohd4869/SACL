#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cat ./run_main_t12_tg.sh

WORK_DIR="/SACL/SACL-XLMR"
MODEL_BASE="/SACL/ptms/afro-xlmr-large" # Davlan/afro-xlmr-large3
#CACHE_DIR="/SACL/ptms/afro-xlmr-large"


EXP_NO="sacl_xlmr-tg"

LOAD_MODEL_PATH="${WORK_DIR}/sacl_xlmr_best_models/sacl_xlmr_tg"

HIDDEN_SIZE=1024
POOL_HEAD="CLS_POOLING_F"
LA="tg"
EVAL_STEP=1
GRAD_ACC_STEP=1
BATCH_SIZE=64

EPOCH_NUM=10
EVAL_BATCH_SIZE=128
PATIENCE_NUM=15
MAX_LENGTH=512
DP_NUM=1
B_LOSS="CrossEntropy"

OUT_DIR="${WORK_DIR}/outputs/${MODEL_BASE##*/}/${EXP_NO}"
LOG_PATH="${WORK_DIR}/logs/${MODEL_BASE##*/}/${EXP_NO}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi
if [[ ! -d ${OUT_DIR} ]];then
    mkdir -p  ${OUT_DIR}
fi

echo "${EXP_NO}"
echo "${OUT_DIR}"
echo "${LOG_PATH}/${EXP_NO}.out"


python -u main_t12.py \
--la ${LA} \
--hidden_size ${HIDDEN_SIZE} \
--train_batch_size  ${BATCH_SIZE} \
--eval_batch_size   ${EVAL_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC_STEP} \
--num_epochs        ${EPOCH_NUM} \
--batch_interval    ${EVAL_STEP} \
--patience          ${PATIENCE_NUM} \
--max_sequence_length ${MAX_LENGTH} \
--pretrain_model_path   ${MODEL_BASE} \
--cache_dir         ${MODEL_BASE} \
--model_head        ${POOL_HEAD} \
--log_path          ${LOG_PATH}/${EXP_NO}.log \
--train_result_path ${OUT_DIR}/train_predict.csv \
--test_result_path  ${OUT_DIR}/test_logits.csv \
--result_dic_path   ${OUT_DIR}/ \
--model_save_path   ${LOAD_MODEL_PATH}/ \
--submission_path   ${OUT_DIR}/task1.txt \
--dropout_num       ${DP_NUM} \
--loss_fct_name  ${B_LOSS} \
> ${LOG_PATH}/${EXP_NO}.out 2>&1


python -u submit_task121_single.py \
--la ${LA} >> ${LOG_PATH}/${EXP_NO}.out 2>&1

