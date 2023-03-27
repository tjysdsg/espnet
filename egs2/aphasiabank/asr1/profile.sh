#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=00:20:00

set -e
set -u
set -o pipefail

. ./path.sh

set -x

second_all="10 15 20"                               # audio length in seconds
fs=16000                                            # sampling rate
# asr_model_file=/ocean/projects/cis210027p/jtang1/espnet/egs2/aphasiabank/asr1/exp/asr_conformer/valid.acc.ave.pth  # assuming config.yaml exists in same directory
asr_model_file=/ocean/projects/cis210027p/jtang1/espnet/egs2/aphasiabank/asr1/exp/asr_ebranchformer/valid.acc.ave.pth  # assuming config.yaml exists in same directory

for second in ${second_all}; do
    python local/profile_encoder.py \
        --second ${second} \
        --fs ${fs} \
        --model_file "${asr_model_file}"
done
