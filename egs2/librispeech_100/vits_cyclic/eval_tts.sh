#!/usr/bin/env bash
#SBATCH --job-name=see_me_rolling
#SBATCH --partition=RM-shared
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8

. ./path.sh

decode_dir=SET_DECODE_DIR
test_sets="dev_clean test_clean test_other"
dumpdir=dump

. utils/parse_options.sh

for dset in ${test_sets}; do
  # Evaluate MCD
  ./pyscripts/utils/evaluate_mcd.py \
    ${decode_dir}/${dset}/wav/wav.scp \
    ${dumpdir}/raw/${dset}/wav.scp

  # Evaluate log-F0 RMSE
  ./pyscripts/utils/evaluate_f0.py \
    ${decode_dir}/${dset}/wav/wav.scp \
    ${dumpdir}/raw/${dset}/wav.scp
done
