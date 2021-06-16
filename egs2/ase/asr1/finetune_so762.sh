#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="so762_train"
val_set="so762_test"
test_sets="so762_test"

asr_config=conf/tuning/train_asr_transformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 8 \
    --max_wav_duration 30 \
    --inference_nj 100 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm false \
    --token_type "word" \
    --train_set "${train_set}" \
    --valid_set "${val_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --local_data_opts "--stage 2" \
    "$@"
