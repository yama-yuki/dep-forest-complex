#!/bin/bash

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 run_pruner.sh

data=test
k=16
threshold='1e-4'

PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex
FOREST_DIR=${PJ_DIR}/biaffine_forest/pkl/dep/${data}/k${k}

log=log.out

. ~/.bashrc
conda activate for
conda info -e

n=1
forest_pkl=${name}.pkl
OUT_DIR=${PJ_DIR}/inside-outside/out/${data}/k${k}/${n}
out_name=${n}forests.forest

echo 'BINARIZING: '${n}'forests'
mkdir -p ${OUT_DIR}

python formatter.py \
    --forest_dir=${FOREST_DIR} \
    --forest_pkl=${forest_pkl} \
    --out_dir=${OUT_DIR} \
    --out_name=${out_name} \
    > ${log}


. ~/.bashrc
conda activate py27
conda info -e


OUT_DIR=${PJ_DIR}/inside-outside/out/${data}/k${k}/${n}
forest_path=${OUT_DIR}/${n}forests.forest

echo 'PRUNING: '${n}'forests'
python pruner.py \
    --prob=${threshold} \
    --forest_path=${forest_path} \
    --out_dir=${OUT_DIR} \
    --suffix=pruned \
    > ${log}


echo 'DONE'

