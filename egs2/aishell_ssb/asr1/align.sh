#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=1000000000

data_dir=data/autism_clean
wav_scp=${data_dir}/wav.scp

# text=${data_dir}/text
text=exp/asr_train_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp/decode_asr_asr_model_valid.acc.best/autism_clean/token

nj=1

exp_dir=exp/asr_train_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp
asr_config=${exp_dir}/config.yaml

out_dir=exp/asr_align_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp
align_out_dir=exp/align_clean

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  log "Error: No positional arguments are required."
  exit 2
fi

. ./path.sh
. ./cmd.sh

mkdir -p ${out_dir}
mkdir -p ${align_out_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ${train_cmd} JOB=1:"${nj}" "${out_dir}"/asr_align.JOB.log \
    python3 local/batch_align.py \
    --asr-train-config ${asr_config} \
    --asr-model-file ${exp_dir}/valid.acc.best.pth \
    --wavscp "${wav_scp}" \
    --text "${text}" \
    --out-dir ${out_dir} || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python local/postprocess_align.py --data-dir=${data_dir} --align-dir=${out_dir} --out-dir=${align_out_dir}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  python local/merge_segment_text.py --text=${text} --out-file=${align_out_dir}/hyp.txt
  cp data/autism/text ${align_out_dir}/ref.txt
fi
