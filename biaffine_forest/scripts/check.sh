#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -t 10:00:00

. ~/.bashrc
conda activate for
conda info -e
