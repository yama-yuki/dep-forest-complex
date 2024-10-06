#!/bin/bash

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 test.sh

data=test
k=8
threshold=0

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
FOREST_DIR=${PJ_DIR}/biaffine_forest/pkl/unlabel2/${data}/k${k}

log=log.out

. ~/.bashrc
conda activate for
conda info -e

n=2
name=${n}forests
forest_pkl=${name}.pkl
OUT_DIR=${PJ_DIR}/inside-outside/test
out_name=${name}.forest

echo 'BINARIZING: '${n}'forests'
mkdir -p ${OUT_DIR}

python formatter_unlabel2.py \
    --n=${n} \
    --forest_dir=${FOREST_DIR} \
    --forest_pkl=${forest_pkl} \
    --out_dir=${OUT_DIR} \
    --out_name=${out_name} \
    > ${log}


echo 'DONE'
