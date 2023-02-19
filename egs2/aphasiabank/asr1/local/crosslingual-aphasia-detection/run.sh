#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
DATA_DIR=/ocean/projects/cis210027p/shared/corpora/AphasiaBank/English

. ../../utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python3 transcript_processing.py \
    --create-data-aphasiabank \
    --transcripts-root-path ${DATA_DIR}/transcripts \
    --wavs-root-path ${DATA_DIR} \
    --language en \
    --save-dict-name transcript.json
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python3 dependency_parser.py \
    --language en \
    --data-dict-path transcript.json \
    --save-dict-name dep.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  python3 feature_extraction.py \
    --language en \
    --calculate-features-from-dict \
    --data-dict-path dep.json \
    --save-dict-name feats.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  python3 experiments.py \
    --data-dict-path-source feats.json
fi
