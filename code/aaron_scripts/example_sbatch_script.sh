#!/bin/bash
#SBATCH --mem=10g
#SBATCH --time=1-1:30:0
#SBATCH --gres=gpu:rtx2080:2
#SBATCH -c4



source /cs/labs/adiyoss/aarnotaub/venvs/my_venv.../bin/activate
cd ../thesis
dora ...


