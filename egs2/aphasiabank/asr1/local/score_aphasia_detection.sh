#!/usr/bin/env bash
# Evaluate Aphasia detection performance and the ASR performance without the "[xxx]" tags.

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
subset=test

. utils/parse_options.sh

[ -z "${decode_dir}" ] && { log "Error: --decode_dir is required"; exit 2; };

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # TODO: Evaluate Aphasia detection
  echo "TODO"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Evaluate ASR
  echo "Please make sure the reference text (dump/raw/test/text) doesn't contain annotations ([xxx])"
  python local/clean_hyp_annotations.py "${decode_dir}/${subset}"
  ./run.sh --stage 13 --stop_stage 13
fi

