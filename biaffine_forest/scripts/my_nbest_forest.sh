#!/bin/bash

#module load cudnn/8.0-6.0

PARSER_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest

save_name=saves/ptb_cophead
NBEST=16

. ~/.bashrc
conda activate for
conda info -e

cd ${PARSER_DIR}
python network.py \
    --save_dir ${save_name} \
    --model Parser \
    --parse_forest \
    --n ${NBEST} \

