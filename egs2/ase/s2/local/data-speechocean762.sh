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

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

log "$0 $*"

if [ $# -ne 0 ]; then
  log "Error: No positional arguments are required."
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for part in train test; do
    local/prep-speechocean762.sh ${SPEECHOCEAN762}/$part data/so762_$part || exit 1
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # create text and utt2scores
  tmp=data/so762_tmp
  python3 ase/fix_so762_format.py \
    --scores=local/speechocean762/scores.json \
    --output-dir=${tmp} || exit 1
  for part in train test; do
    cp ${tmp}/cpl.txt data/so762_$part/cpl.txt || exit 1
    cp ${tmp}/text data/so762_$part/text || exit 1
    cp ${tmp}/utt2scores data/so762_$part/utt2scores || exit 1
    utils/fix_data_dir.sh data/so762_$part || exit 1
  done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
