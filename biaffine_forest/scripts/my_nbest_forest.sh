#!/bin/bash

#module load cudnn/8.0-6.0

PARSER_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest

save_name=saves/ptb_cophead
NBEST=8

##dev/test/mytree
test_data=mytree
PKL_DIR=${PARSER_DIR}/pkl/${test_data}/k${NBEST}

##sbatch --gres=gpu:1 --job-name=forest_parse --partition=gpu_long --time=12:00:00 my_nbest_forest.sh

. ~/.bashrc
conda activate for
conda info -e

cd ${PARSER_DIR}
python network.py \
    --save_dir ${save_name} \
    --model Parser \
    --parse_forest \
    --n ${NBEST} \
    --pkl_dir=${PKL_DIR} \

