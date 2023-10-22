#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/SACL/SACL-LSTM"

VV="sacl_lstm-text-softmax7"


DATASET="meld"
DATA_DIR="${WORK_DIR}/data/meld/meld_features_roberta.pkl"

SEED="4 3 2 1 0"
G="1" # 0 [1]
AT_RATE="1.0" # 0.1 [1.0]
SITU_RATE="0 0.1" # 0 [0.1]
SPE_RATE="0" # [0] 0.1
AT_EPISION="5" # 1 [5]
SCL_T="0.1 1.0" # [0.1] 0.5 [1.0]
SCL_W="0.1" # 0.05 [0.1] 0.5
SCL_W2="0.05" # [0.05] 0.1 0.5

ss=0
sp=0

for g in ${G[@]}
do

for at_rate in ${AT_RATE[@]}
do
for situ_rate in ${SITU_RATE[@]}
do
for speaker_rate in ${SPE_RATE[@]}
do
for at_epsilon in ${AT_EPISION[@]}
do
for scl_t in ${SCL_T[@]}
do
for scl_w in ${SCL_W[@]}
do
for scl_w2 in ${SCL_W2[@]}
do

    for seed in ${SEED[@]}
    do
        EXP_NO="${VV}_${DATASET}_bert-lastlayer_fgm-rate${at_rate}-si${situ_rate}-sp${speaker_rate}-e${at_epsilon}_scl-t${scl_t}-w${scl_w}-${scl_w2}_b16-gradacc2-seedall_pl-min2_lossSUM"
        echo "EXP_NO: ${EXP_NO}"
        echo "dataset: ${DATASET}, g: ${g}, at_rate: ${at_rate}, situ_rate: ${situ_rate}, speaker_rate: ${speaker_rate}, at_epsilon: ${at_epsilon}, scl_t: ${scl_t}, scl_w: ${scl_w}, scl_w2: ${scl_w2}, seed: ${seed}"

        LOG_PATH="${WORK_DIR}/logs/${DATASET}"
        OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${VV}/${EXP_NO}"
        if [[ ! -d ${LOG_PATH} ]];then
            mkdir -p  ${LOG_PATH}
        fi
        if [[ ! -d ${OUT_DIR} ]];then
            mkdir -p  ${OUT_DIR}
        fi

        # --scl_hidden_flag \
        # --load_model_state_dir ${MODEL_DIR}
        python -u ${WORK_DIR}/code/run_train_bert_me.py   \
            --adversary_flag \
            --at_rate ${at_rate} --situ_rate ${situ_rate} --speaker_rate ${speaker_rate} --at_epsilon ${at_epsilon}  \
            --scl_t ${scl_t} --scl_w ${scl_w} --scl_w2 ${scl_w2}  \
            --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR}   \
            --base_layer 1 --step_s ${ss}  --step_p ${sp} \
            --lr 0.0001 --l2 0.0002  --dropout 0.2  --gamma $g  \
            --batch_size 16 --gradient_accumulation_steps 2 --use_valid_flag --seed  ${seed} \
        >> ${LOG_PATH}/${EXP_NO}.out 2>&1
    done

done
done
done
done
done
done
done

done
