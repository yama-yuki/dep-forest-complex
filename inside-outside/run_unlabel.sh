#!/bin/bash

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 run_unlabel.sh

data=test
k=2
threshold=0

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
FOREST_DIR=${PJ_DIR}/biaffine_forest/pkl/unlabel2/${data}/k${k}

log=log.out
<< COMMENTOUT
COMMENTOUT
. ~/.bashrc
conda activate for
conda info -e

for n in {1..10}; do
    name=${n}forests
    forest_pkl=${name}.pkl
    OUT_DIR=${PJ_DIR}/inside-outside/out/unlabel2/${data}/k${k}/${n}
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
done

echo 'DONE'


. ~/.bashrc
conda activate py27
conda info -e

for n in {1..10}; do
    OUT_DIR=${PJ_DIR}/inside-outside/out/unlabel2/${data}/k${k}/${n}
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


