#!/bin/bash

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 run.sh

data=test
k=32
threshold='1e-4'

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
FOREST_DIR=${PJ_DIR}/biaffine_forest/pkl/dep/${data}/k${k}

log=log.out

. ~/.bashrc
conda activate for
conda info -e

for n in {1..10}; do
    name=${n}forests
    forest_pkl=${name}.pkl
    OUT_DIR=${PJ_DIR}/inside-outside/out/${data}/k${k}/${n}
    out_name=${name}.forest

    echo 'BINARIZING: '${n}'forests'
    mkdir -p ${OUT_DIR}

    python formatter.py \
        --n=${n} \
        --forest_dir=${FOREST_DIR} \
        --forest_pkl=${forest_pkl} \
        --out_dir=${OUT_DIR} \
        --out_name=${out_name} \
        > ${log}
done

. ~/.bashrc
conda activate py27
conda info -e

for n in {1..10}; do
    OUT_DIR=${PJ_DIR}/inside-outside/out/${data}/k${k}/${n}
    forest_path=${OUT_DIR}/${n}forests.forest

    echo 'PRUNING: '${n}'forests'
    python pruner.py \
        --prob=${threshold} \
        --forest_path=${forest_path} \
        --out_dir=${OUT_DIR} \
        --suffix=pruned \
        > ${log}
done

echo 'DONE'

