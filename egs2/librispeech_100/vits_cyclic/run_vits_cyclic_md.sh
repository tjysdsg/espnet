#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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
test_sets="test_clean dev_clean"

train_config=conf/tuning/train_vits_xvector_md_sanity_freeze.yaml
inference_config=conf/decode.yaml


./tts.sh \
    --ngpu 1 \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --stage 6 \
    --stop_stage 6 \
    --dumpdir dump \
    --expdir exp/vits_cyclic \
    --tts_task gan_tts \
    --use_multidecoder true \
    --use_xvector true \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --token_type phn \
    --tag "${tag}" \
    --srctexts "data/${train_set}/text" \
    --inference_config "${inference_config}" \
    --audio_format "wav" "$@"
