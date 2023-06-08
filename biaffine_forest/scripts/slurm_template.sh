#SBATCH --job-name=forest_parse
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1

. ~/.bashrc
conda activate for
conda info -e

