#!/usr/bin/env bash
#SBATCH --job-name=see_me_rolling
#SBATCH --partition=RM-shared
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh

decode_dir=
test_sets="dev_clean test_clean"
dumpdir=dump
stage=1
stop_stage=100

. utils/parse_options.sh

if [ -z "${decode_dir}" ]; then
    log "Require '--decode_dir'"
    exit 1
fi

# Evaluate MCD
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for dset in ${test_sets}; do
    ./pyscripts/utils/evaluate_mcd.py \
      ${decode_dir}/${dset}/wav/wav.scp \
      ${dumpdir}/raw/${dset}/wav.scp
  done
fi

# Evaluate log-F0 RMSE
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for dset in ${test_sets}; do
    ./pyscripts/utils/evaluate_f0.py \
      ${decode_dir}/${dset}/wav/wav.scp \
      ${dumpdir}/raw/${dset}/wav.scp
  done
fi

# Evaluate CER
# DON'T use sbatch for this stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for dset in ${test_sets}; do
    ./scripts/utils/evaluate_asr.sh \
      --model_tag "pyf98/librispeech_conformer_hop_length160" \
      --nj 32 \
      --inference_args "--beam_size 10 --ctc_weight 0.3 --lm_weight 0.0" \
      --gt_text ${dumpdir}/raw/${dset}/text \
      --stop_stage 3 \
      ${decode_dir}/${dset}/wav/wav.scp \
      ${decode_dir}/${dset}/asr_results
  done
fi
