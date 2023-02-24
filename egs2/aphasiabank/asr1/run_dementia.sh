#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_tag="dementia_conformer"

train_set="train"
valid_set="val"
test_sets="test"
tag_insertion=none

asr_config=conf/tuning/train_asr_conformer_dementia.yaml

feats_normalize=global_mvn
if [[ ${asr_config} == *"hubert"* ]] || [[ ${asr_config} == *"wavlm"* ]]; then
  feats_normalize=utt_mvn # https://github.com/espnet/espnet/issues/4006#issuecomment-1047898558
fi

inference_config=conf/decode.yaml

./asr.sh \
  --asr_tag "${asr_tag}" \
  --lang en \
  --inference_nj 2 \
  --gpu_inference true \
  --ngpu 1 \
  --max_wav_duration 33 \
  --audio_format wav \
  --feats_type raw \
  --token_type char \
  --use_lm false \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --feats_normalize ${feats_normalize} \
  --local_data_opts "--dataset DementiaBank --tag_insertion ${tag_insertion} " \
  --lm_train_text "data/${train_set}/text" "$@"
