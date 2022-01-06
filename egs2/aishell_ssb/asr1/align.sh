#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

wav_scp=data/autism/wav.scp
text=data/autism/text
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
  python3 local/batch_align.py \
  --asr-train-config ${asr_config} \
  --asr-model-file ${exp_dir}/valid.acc.best.pth \
  --wavscp "${wav_scp}" \
  --text "${text}" \
  --out-dir ${out_dir} || exit 1
