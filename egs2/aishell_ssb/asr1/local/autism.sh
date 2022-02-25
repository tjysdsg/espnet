#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
out_dir=data/autism

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
  log "Error: No positional arguments are required."
  exit 2
fi

mkdir -p ${out_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python local/autism.py \
    --filter=local/autism_list.txt \
    --override-utt2align=local/utt2align_correct.json \
    --data-dir=${AUTISM} \
    --out-dir=${out_dir} || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  cp local/utt2align_correct.json ${out_dir}/utt2align.json
  utils/fix_data_dir.sh ${out_dir} || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  mkdir -p ${out_dir}_clean
  python local/cut_long_wavs.py \
    --data-dir=${out_dir} \
    --out-dir=${out_dir}_clean || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  utils/fix_data_dir.sh ${out_dir}_clean || exit 1
fi

log "Data preparation completed"
