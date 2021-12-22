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
out_dir=data

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
  python local/aishell_ssb.py --wav-dir=${AISHELL_SSB} --out-dir=${out_dir}/aishell_ssb || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python local/split_train_test_val.py --in-dir=${out_dir}/aishell_ssb --out-dir=${out_dir} || exit 1

  utils/fix_data_dir.sh ${out_dir}/train || exit 1
  utils/fix_data_dir.sh ${out_dir}/test || exit 1
  utils/fix_data_dir.sh ${out_dir}/val || exit 1
fi


log "Data preparation completed"
