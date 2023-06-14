#!/bin/bash

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex

## 1best parser
DATA_DIR=${PJ_DIR}/biaffine_forest/saves/ptb_cophead
test_path=${DATA_DIR}/test.conllu_1best.txt

## 1best from forest
PRED_DIR=${PJ_DIR}/outputs/k4
p1=vanilla_4.conllu
p2=rescore_4-01-01.conllu
p3=rescore_4-03-01.conllu
p4=rescore_4-05-03.conllu
p5=rescore_4-07-05.conllu

LIB_DIR=${PJ_DIR}/lib
cd ${LIB_DIR}
python eval.py \
    --test_path=${test_path} \
    --plist  ${PRED_DIR}/${p1} ${PRED_DIR}/${p2} ${PRED_DIR}/${p3} ${PRED_DIR}/${p4} ${PRED_DIR}/${p5}

