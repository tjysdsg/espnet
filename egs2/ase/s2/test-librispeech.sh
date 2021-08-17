#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_all"
val_set="dev"
test_sets="test"

asr_config=conf/tuning/train_asr_transformer.yaml
inference_config=conf/decode_asr.yaml

# --gpu_inference true \
./asr.sh \
  --stage 11 \
  --lang en \
  --ngpu 8 \
  --max_wav_duration 30 \
  --inference_nj 300 \
  --speed_perturb_factors "0.9 1.0 1.1" \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --asr_tag "finetune" \
  --use_lm false \
  --token_type "word" \
  --train_set "${train_set}" \
  --valid_set "${val_set}" \
  --test_sets "${test_sets}" \
  --lm_train_text "data/${train_set}/text" \
  "$@"

decode_dir=exp/asr_finetune/decode_asr_asr_model_valid.acc.best/test
ref_path=data/test/text

python3 ase/sp2phone.py --input=${decode_dir}/token --output=${decode_dir}/token.phones
python3 ase/wer.py ${decode_dir}/token.phones ${ref_path} --output-dir=${decode_dir}
