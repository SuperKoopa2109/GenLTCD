#!/usr/bin/env bash
#SBATCH --partition training
#SBATCH --gres gpu:1
#SBATCH -o 
#SBATCH --mail-type ALL
#SBATCH --mail-user kaspareit@uni-postdam.de
export PYTHONUNBUFFERED="x"
srun python main.py

./contrastive_train.sh
./estimate_k.sh
./extract_features.sh
./k_means.sh