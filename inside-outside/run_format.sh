#!/bin/bash

. ~/.bashrc
conda activate for
conda info -e

data=test
k=16
name='1forests'

FOREST_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl/dep/${data}/k${k}
forest_pkl=${name}.pkl

OUT_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/inside-outside/out/${data}/k${k}
out_name=${name}.forest

##sbatch --job-name=in-out-format --account=lang --partition=lang_long --time=90:00:00 run_format.sh
python formatter.py \
    --forest_dir=${FOREST_DIR} \
    --forest_pkl=${forest_pkl} \
    --out_dir=${OUT_DIR} \
    --out_name=${out_name} \
    > log.out

