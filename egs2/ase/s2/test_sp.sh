decode_dir=exp/asr_finetune_nofreeze/decode_asr_asr_model_valid.acc.best/so762_test
test_dir=data/so762_test
python3 ase/ase_score.py \
  --hyp=${decode_dir}/token \
  --cpl=${test_dir}/cpl.txt \
  --utt2scores=${test_dir}/utt2scores \
  --output-dir=${decode_dir}
