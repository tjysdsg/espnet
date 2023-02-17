#!/usr/bin/env bash
# Intermediate supervision using ASR output of pretrained model, decoded using CTC+Attention

set -e
set -u
set -o pipefail


fs=16000 # original 24000
n_fft=2048
n_shift=300
win_length=1200

tag="unpaired_360_sudo"

train_set="train_clean_360"
valid_set="dev_clean"
test_sets="dev_clean test_clean"

train_config=conf/tuning/train_transformer_xvector_md_unpaired.yaml
inference_config=conf/decode.yaml
inference_asr_config=conf/decode_asr.yaml


./tts.sh \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 6 \
    --inference_model valid.loss.ave.pth \
    --inference_nj 32 \
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
