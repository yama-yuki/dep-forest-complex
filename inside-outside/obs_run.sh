#!/bin/bash

. ~/.bashrc
conda activate py27
conda info -e

forest_path=/home/is/yuki-yama/work/d3/dep-forest-complex/inside-outside/dep.forest

##sbatch --job-name=in-out-prune --account=lang --partition=lang_long --time=90:00:00 run.sh
python pruner.py \
    --prob=10 \
    --forest_path=${forest_path} \
    --out_dir=out \
    --suffix=pruned \
    > log.out


