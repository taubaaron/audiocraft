#!/bin/bash
#SBATCH --mem=10g
#SBATCH --time=1-1:30:0
#SBATCH --gres=gpu:rtx2080:2
#SBATCH -c4



source /cs/labs/adiyoss/aarnotaub/thesis/audiocraft/code/env_elbit/bin/activate.csh
#cd ../thesis/audiocraft/code
dora run solver=compression/encodec_base_8khz


