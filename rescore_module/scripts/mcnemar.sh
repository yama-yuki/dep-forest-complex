#!/bin/bash
RESCORE_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/rescore_module
MODEL_DIR=${RESCORE_DIR}/models
DATA_DIR=${RESCORE_DIR}/data
PRED_DIR=${RESCORE_DIR}/predictions

## data type (ewt/gum/partut)
type=ewt

## model
pretrained_model=bert-base-uncased
## finetuned model
#model_1_name=bert-base-uncased_${order}_2_3e-5_24

cd ${RESCORE_DIR}

for sconj_num in 1 2 3
do
    for order in 1 2 3
    do
    model_2_name=${pretrained_model}_${order}_2_3e-5_32
    test_data_name=ud${sconj_num}_${order}

    pred_1=${PRED_DIR}/stanza/${type}_${sconj_num}.csv
    pred_2=${PRED_DIR}/${model_2_name}/${type}/${test_data_name}.csv

    python ${RESCORE_DIR}/mcnemar.py \
        --pred_1_path=${pred_1} \
        --pred_2_path=${pred_2}
    done
done
