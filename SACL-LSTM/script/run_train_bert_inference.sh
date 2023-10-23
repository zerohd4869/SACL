#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/SACL/SACL-LSTM"
MODEL_DIR="${WORK_DIR}/sacl_lstm_best_models"
SEED="4 3 2 1 0"

EXP_NO="sacl_lstm_inference"


DATASET="iemocap"
DATA_DIR="${WORK_DIR}/data/iemocap/iemocap_features_roberta.pkl"
for seed in ${SEED[@]}
do
    echo "EXP_NO: ${EXP_NO}, DATASET: ${DATASET}, seed: ${seed}"
    python -u ${WORK_DIR}/code/run_train_bert_ie.py --status test --data_dir ${DATA_DIR} --load_model_state_dir "${MODEL_DIR}/${DATASET}/${seed}/loss_sacl-lstm.pkl" >> ${EXP_NO}.out 2>&1
done


DATASET="meld"
DATA_DIR="${WORK_DIR}/data/meld/meld_features_roberta.pkl"
for seed in ${SEED[@]}
do
    echo "EXP_NO: ${EXP_NO}, DATASET: ${DATASET}, seed: ${seed}"
    python -u ${WORK_DIR}/code/run_train_bert_me.py --status test --data_dir ${DATA_DIR} --load_model_state_dir "${MODEL_DIR}/${DATASET}/${seed}/loss_sacl-lstm.pkl" >> ${EXP_NO}.out 2>&1
done


DATASET="emorynlp"
DATA_DIR="${WORK_DIR}/data/emorynlp/emorynlp_features_roberta.pkl"
for seed in ${SEED[@]}
do
    echo "EXP_NO: ${EXP_NO}, DATASET: ${DATASET}, seed: ${seed}"
    python -u ${WORK_DIR}/code/run_train_bert_emo.py --status test --data_dir ${DATA_DIR} --load_model_state_dir "${MODEL_DIR}/${DATASET}/${seed}/f1_sacl-lstm.pkl" >> ${EXP_NO}.out 2>&1
done


echo "done"