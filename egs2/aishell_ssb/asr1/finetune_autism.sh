#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="autism_clean"
val_set="autism_clean"
test_sets="autism_clean"

asr_config=conf/tuning/train_asr_conformer_s3prlfrontend_wav2vec2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
  --ngpu 8 \
  --max_wav_duration 30 \
  --inference_nj 30 \
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
  --lm_train_text "data/train/text" \
  --skip_data_prep true \
  --asr_exp "exp/asr_train_asr_conformer_s3prlfrontend_wav2vec2_raw_word_sp" \
  "$@"
