#!/bin/bash

set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=10000
scoring_opts=
token_list=data/en_token_list/word/tokens.txt
python=python3
asr_exp=exp/asr_train_asr_transformer_raw_en_word_sp
inference_tag=decode_asr_asr_model_valid.acc.best
train_sets="so762_train libri_scoring_train"
test_sets="so762_test"
model_dir=exp/scoring_train # model save path
aug_test_data=false
aug_train_data=true

log "$0 $*"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

decode_dir="${asr_exp}/${inference_tag}"

combine_data() {
  sets=$1       # input data subsets
  output_dir=$2 # output dir

  rm -rf ${output_dir}
  mkdir -p ${output_dir}
  for x in ${sets}; do
    cat data/${x}/text >>${output_dir}/ref.txt
    cat data/${x}/utt2scores >>${output_dir}/utt2scores

    # combine onehots and probs
    ${python} ase/combine_prob_onehot.py --token=${decode_dir}/${x}/token \
      --probs=${decode_dir}/${x}/probs \
      --phone-table=${token_list} \
      --output-path=${decode_dir}/${x}/prob_combined.txt

    cat ${decode_dir}/${x}/prob_combined.txt >>${output_dir}/hyp.txt
  done
}

train_model() {
  action=$1    # train/test
  dir=$2       # data output dir
  data_sets=$3 # datasets
  aug=$4       # true/false
  echo "train_model $1 $2 $3 $4"

  # combine train sets
  combine_data "${data_sets}" ${dir}

  # perform data aug
  ref_file=${dir}/ref.txt
  utt2scores=${dir}/utt2scores
  if [ "${aug}" = "true" ]; then
    ${python} ase/aug_scoring_data.py \
      --text=${dir}/ref.txt \
      --scores=${dir}/utt2scores \
      --output-dir=${dir}

    ref_file=${dir}/ref_aug.txt
    utt2scores=${dir}/utt2scores_aug
  fi

  export PYTHONPATH=ase/
  ${python} ase/scoring_mlp.py "${action}" \
    ${dir}/hyp.txt \
    ${ref_file} \
    ${scoring_opts} \
    --phone-table=${token_list} \
    --scores=${utt2scores} \
    --model-dir=${model_dir} \
    --output-dir=${dir}
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 12: Training scoring model"
  train_model train data/scoring_train "${train_sets}" ${aug_train_data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 13: Testing scoring model"
  train_model test data/scoring_test "${test_sets}" ${aug_test_data}
fi
