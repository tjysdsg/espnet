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
out_dir=data/aishell_ssb

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
  python local/aishell_ssb.py --wav-dir=${AISHELL_SSB} --out-dir=${out_dir}
fi

utils/fix_data_dir.sh ${out_dir}

log "Data preparation completed"
