#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
val_set="val"
test_sets="test"

# asr_config=conf/tuning/train_asr_transformer.yaml
asr_config=conf/tuning/train_asr_conformer_s3prlfrontend_wav2vec2.yaml
inference_config=conf/decode_asr.yaml

# --gpu_inference true \
./asr.sh \
  --ngpu 8 \
  --max_wav_duration 30 \
  --inference_nj 300 \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --inference_asr_model valid.acc.best.pth \
  --use_lm false \
  --token_type "word" \
  --feats_normalize uttmvn \
  --train_set "${train_set}" \
  --valid_set "${val_set}" \
  --test_sets "${test_sets}" \
  --lm_train_text "data/${train_set}/text" \
  "$@"
