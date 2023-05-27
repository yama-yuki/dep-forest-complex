#!/bin/bash
#SBATCH -J ptb_10k -C K80 --partition=gpu --gres=gpu:1 --time=2-00:00:00 --output=train.out_ptb_10k --error=train.err_ptb_10k
#SBATCH --mem=20GB
#SBATCH -c 5

module load cudnn/8.0-6.0

python network.py --config_file config/ptb_cophead.cfg --model Parser

