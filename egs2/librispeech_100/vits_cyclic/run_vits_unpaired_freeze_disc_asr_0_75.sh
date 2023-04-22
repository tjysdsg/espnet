#!/usr/bin/env bash
set -e
set -u
set -o pipefail

fs=16000 # original 24000
n_fft=1024
n_shift=256
win_length=null

tag="vits_unpaired_freeze_tts_asr_0_75_lr_1e_4_kl_full_wav"

train_set="train_clean_460"
valid_set="dev_clean"
test_sets="dev_clean test_clean"

train_config=conf/tuning/train_vits_char_unpaired_freeze_disc_460h_asr0_75.yaml
inference_tts_config=conf/decode.yaml
inference_asr_config=conf/decode_asr.yaml


./tts.sh \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 6 \
    --inference_model latest.pth \
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
    --expdir exp/vits_unpaired_460 \
    --tts_task gan_tts \
    --train_config "${train_config}" \
    --g2p none \
    --feats_extract linear_spectrogram \
    --inference_config "${inference_tts_config}" \
    --inference_asr_config "${inference_asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --sudo_text  "/ocean/projects/cis210027p/jtang1/C/dump/raw/train_clean_460/sudo_text" \
    --audio_format "wav" "$@"
