#!/usr/bin/env bash
# Intermediate supervision using ASR output of pretrained model, decoded using CTC+Attention

set -e
set -u
set -o pipefail


fs=16000
n_fft=1024
n_shift=256
win_length=null

tag="unpaired_sudo_new"

train_set="train_clean_360"
valid_set="dev_clean"
test_sets="dev_clean test_clean test_other"

train_config=conf/tuning/train_unpaired_new.yaml
inference_config=conf/decode.yaml
inference_asr_config=conf/decode_asr.yaml


./tts.sh \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 6 \
    --inference_model valid.loss.ave.pth \
    --gpu_inference true \
    --inference_nj 2 \
    --use_multidecoder true \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector true \
    --token_type char \
    --cleaner none \
    --tag "${tag}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_config "${inference_asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --sudo_text "decode_train_clean_360/text" \
    --audio_format "wav" "$@"
