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
  cat "${decode_dir}"/logdir/output.*/1best_recog/interctc_layer6.txt >"${decode_dir}"/interctc.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python local/score_interctc_aux.py "${decode_dir}"/interctc.txt
fi
