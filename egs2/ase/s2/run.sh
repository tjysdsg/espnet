#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="l2arctic_train"
val_set="l2arctic_test"
test_sets="l2arctic_test"

asr_config=conf/tuning/train_asr_transformer.yaml
inference_config=conf/decode_asr.yaml
# asr_args="--init_param data/trained.pth:::ctc,decoder.output_layer,decoder.embed,normalize.mean,normalize.std --freeze_param decoder.decoders encoder specaug frontend"
asr_args="--init_param data/trained.pth:::ctc,decoder.output_layer,decoder.embed,normalize.mean,normalize.std"

./asr.sh \
  --lang en \
  --ngpu 8 \
  --max_wav_duration 30 \
  --inference_nj 200 \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --asr_config "${asr_config}" \
  --asr_args "${asr_args}" \
  --inference_config "${inference_config}" \
  --asr_tag "finetune" \
  --use_lm false \
  --token_type "word" \
  --train_set "${train_set}" \
  --valid_set "${val_set}" \
  --test_sets "${test_sets}" \
  --lm_train_text "data/${train_set}/text" \
  "$@"
