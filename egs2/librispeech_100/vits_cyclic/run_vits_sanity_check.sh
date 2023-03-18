#!/usr/bin/env bash
set -e
set -u
set -o pipefail

fs=16000 # original 24000
n_fft=1024
n_shift=256
win_length=null

tag="vits_sanity_check_freeze"

train_set="train_clean_360"
valid_set="dev_clean"
test_sets="dev_clean test_clean dev_other test_other"

train_config=conf/tuning/train_vits_xvector_md_sanity_freeze.yaml
inference_config=conf/decode.yaml
inference_asr_config=conf/decode_asr.yaml
inference_tts_config=conf/tuning/decode_vits.yaml


./tts.sh \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 6 \
    --inference_model valid.loss.best.pth \
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
    --dumpdir dump \
    --tts_task gan_tts \
    --train_config "${train_config}" \
    --g2p none \
    --feats_extract linear_spectrogram \
    --inference_config "${inference_config}" \
    --inference_asr_config "${inference_asr_config}" \
    --inference_tts_config "${inference_tts_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --audio_format "wav" "$@"
