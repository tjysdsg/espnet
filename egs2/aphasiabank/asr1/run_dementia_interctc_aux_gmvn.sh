#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_tag="dementia_ebranchformer_wavlm_interctc6_inv_gmvn"

train_set="train_sp_ssl"
valid_set="val_ssl"
test_sets="test_ssl"

asr_config=conf/tuning/train_asr_dementia_ebranchformer_wavlm_interctc6_extracted.yaml
inference_config=conf/decode.yaml

# Before running this script:
# - Run ./run_dementia.sh to prepare a regular dump
# - Run local/dementia/prepare_ssl.sh to extract SSL features

./asr.sh \
  --stage 4 \
  --feats_normalize global_mvn \
  --feats_type "extracted" \
  --asr_tag "${asr_tag}" \
  --lang en \
  --inference_nj 1 \
  --gpu_inference true \
  --ngpu 2 \
  --max_wav_duration 33 \
  --audio_format wav \
  --token_type char \
  --use_lm false \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --local_data_opts "--hey_stop_stage_1" \
  --auxiliary_data_tags "utt2aph" \
  --post_process_local_data_opts "--dataset DementiaBank --stage 6" \
  --nlsyms_txt "local/nlsyms.txt" \
  --lm_train_text "data/${train_set}/text" "$@"
