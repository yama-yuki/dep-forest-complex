#!/bin/bash
#SBATCH -J ptb_1k --time=5-00:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=30GB
#SBATCH -c 5

#module load cudnn/8.0-6.0

save_name=saves/ptb_cophead
NBEST=16
ALPHA=1

python network.py \
    --save_dir ${save_name} \
    --model Parser \
    --nbest \
    --n ${NBEST} \
    --alpha ${ALPHA} \
    #--rescore \

