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
  mkdir -p data/local/l2arctic
  mkdir -p data/l2arctic

  python3 local/l2_arctic.py --l2-path=${L2ARCTIC} --output-dir=data/local/l2arctic

  cp data/local/l2arctic/ppl data/l2arctic/text
  cp data/local/l2arctic/{wav.scp,utt2spk} data/l2arctic/
fi

utils/fix_data_dir.sh data/l2arctic || exit 1;

log "Successfully finished. [elapsed=${SECONDS}s]"
