#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/SACL/SACL-LSTM"

VV="sacl_lstm-text-softmax6"


DATASET="iemocap"
DATA_DIR="${WORK_DIR}/data/${DATASET}/iemocap_features_roberta.pkl"

SEED="4 3 2 1 0"
G="0" # [0] 1
AT_RATE="1.0" # 0.1 [1.0]
AT_EPISION="5" # 1 [5]
SCL_T="0.1 1.0" # [0.1] 0.5 [1.0]
SCL_W="0.05" # [0.05] 0.1 0.5
SCL_W2="0.5" # 0.05 0.1 [0.5]

ss=0
sp=0

for g in ${G[@]}
do

for at_rate in ${AT_RATE[@]}
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
        EXP_NO="${VV}_${DATASET}_bert-lastlayer_fgm-rate${at_rate}-e${at_epsilon}_scl-t${scl_t}-w${scl_w}-${scl_w2}_b2-gradacc16-seedall_pl-min2_lossSUM"
        echo "EXP_NO: ${EXP_NO}"
        echo "dataset: ${DATASET}, g: ${g}, at_rate: ${at_rate}, at_epsilon: ${at_epsilon}, scl_t: ${scl_t}, scl_w: ${scl_w}, scl_w2: ${scl_w2}, seed: ${seed}"

        LOG_PATH="${WORK_DIR}/logs/${DATASET}"
        OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${VV}/${EXP_NO}"
        if [[ ! -d ${LOG_PATH} ]];then
            mkdir -p  ${LOG_PATH}
        fi
        if [[ ! -d ${OUT_DIR} ]];then
            mkdir -p  ${OUT_DIR}
        fi

        # --scl_hidden_flag \
        # --load_model_state_dir ${MODEL_DIR} \
        python -u ${WORK_DIR}/code/run_train_bert_ie.py \
            --scl_hidden_flag \
            --adversary_flag \
            --class_weight  \
            --at_rate ${at_rate} --at_epsilon ${at_epsilon} \
            --scl_t ${scl_t} --scl_w ${scl_w} --scl_w2 ${scl_w2}   \
            --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR} \
            --base_layer 2  --step_s ${ss}  --step_p ${sp}  \
            --lr 0.0001  --l2 0.0002  --dropout 0.2  --gamma $g   \
            --batch_size 2 --gradient_accumulation_steps 16  --epoch 100 --valid_rate 0.1 --seed  ${seed} \
        >> ${LOG_PATH}/${EXP_NO}.out 2>&1
    done

done
done
done
done
done

done



