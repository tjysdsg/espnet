#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1          # Processes starts from the specified stage.
stop_stage=10000 # Processes is stopped at the specified stage.
train_set="train"
val_set="test"
test_sets="test"
nj=1

exp_dir=exp/asr_train_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp
out_dir=exp/asr_align_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp
asr_config=${exp_dir}/config.yaml

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  log "Error: No positional arguments are required."
  exit 2
fi

. ./path.sh
. ./cmd.sh

mkdir -p ${out_dir}

${train_cmd} JOB=1:"${nj}" "${out_dir}"/asr_align.JOB.log \
  python3 -m espnet2.bin.asr_align \
  --asr_train_config ${asr_config} \
  --asr_model_file ${exp_dir}/valid.acc.best.pth \
  --fs 16000 \
  --audio exp/SSB00050353.wav \
  --text exp/text \
  --output "${out_dir}/aligned.txt" || exit 1
