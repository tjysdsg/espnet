#!/usr/bin/env bash
python local/mix_sudo_text.py \
    dump/raw/train_clean_460/text \
    decode_train_clean_360/text \
    dump/raw/train_clean_460/sudo_text
