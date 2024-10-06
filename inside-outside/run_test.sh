#!/bin/bash

. ~/.bashrc
conda activate py27
conda info -e

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 run_test.sh

data=test
k=16
threshold='0'
log=log2.out

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
n=1
out_name=${n}forests.forest
OUT_DIR=${PJ_DIR}/inside-outside/out/unlabel2/${data}/k${k}/${n}
forest_path=${OUT_DIR}/${n}forests.forest

mkdir -p ${OUT_DIR}

echo 'PRUNING: '${n}'forests'
python pruner.py \
    --prob=${threshold} \
    --forest_path=${forest_path} \
    --out_dir=${OUT_DIR} \
    --suffix=pruned \
    > ${log}

