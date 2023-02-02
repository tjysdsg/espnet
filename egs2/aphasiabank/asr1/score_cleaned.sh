#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
decode_dir=

log "$0 $*"
. utils/parse_options.sh

help_message=$(
  cat <<EOF
Usage: $0

Options:
    --decode_dir (string): Path to decoded results.
      For example, --decode_dir=exp/asr_train_asr_conformer_raw_en_char_sp/decode_asr_model_valid.acc.ave/test"
EOF
)

if [ -z "${decode_dir}" ]; then
  log "${help_message}"
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ./run.sh --stage 13 --stop_stage 13
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for sub in score_cer score_wer; do
    mkdir -p "${decode_dir}/${sub}_clean"
  done

  python local/clean_hyp_annotations.py \
    --token-type=char "${decode_dir}/score_cer/hyp.trn" "${decode_dir}/score_cer_clean/hyp.trn"
  python local/clean_hyp_annotations.py \
    --token-type=char "${decode_dir}/score_cer/ref.trn" "${decode_dir}/score_cer_clean/ref.trn"

  python local/clean_hyp_annotations.py \
    --token-type=word "${decode_dir}/score_wer/hyp.trn" "${decode_dir}/score_wer_clean/hyp.trn"
  python local/clean_hyp_annotations.py \
    --token-type=word "${decode_dir}/score_wer/ref.trn" "${decode_dir}/score_wer_clean/ref.trn"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for sub in score_cer_clean score_wer_clean; do
    sclite -r "${decode_dir}/${sub}/ref.trn" -h "${decode_dir}/${sub}/hyp.trn" \
      -i rm -o all stdout >"${decode_dir}/${sub}/result.txt"

    grep -e Avg -e SPKR -m 2 "${decode_dir}/${sub}/result.txt"
  done
fi
