#!/usr/bin/env bash

# Aphasia English recognition + detection experiment
#   - E-Branchformer
#   - WavLM
#   - InterCTC-6

set -e
set -u
set -o pipefail

asr_tag="ebranchformer_wavlm_interctc6"
train_set="train"
valid_set="val"
test_sets="test"
include_control=true
include_lang_id=false

asr_config=conf/tuning/train_asr_ebranchformer_small_wavlm_large1_interctc6.yaml
inference_config=conf/decode.yaml

feats_normalize=global_mvn
if [[ ${asr_config} == *"hubert"* ]] || [[ ${asr_config} == *"wavlm"* ]]; then
  feats_normalize=utt_mvn # https://github.com/espnet/espnet/issues/4006#issuecomment-1047898558
fi

./asr.sh \
  --asr_tag "${asr_tag}" \
  --lang en \
  --inference_nj 2 \
  --gpu_inference true \
  --ngpu 2 \
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
  --local_data_opts "--include_control ${include_control} --include_lang_id ${include_lang_id} --languages English" \
  --auxiliary_data_tags "utt2aph" \
  --post_process_local_data_opts "--stage 8" \
  --nlsyms_txt "local/nlsyms.txt" \
  --lm_train_text "data/${train_set}/text" "$@"
