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
# train_sets="so762_train libri_scoring_train"
# test_sets="so762_test libri_scoring_test"
train_sets="libri_scoring_train"
test_sets="libri_scoring_test"
model_dir=exp/scoring_train # model save path
aug_test_data=false
aug_train_data=true
aug_rule_file=conf/aug0.yaml

log "$0 $*"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

decode_dir="${asr_exp}/${inference_tag}"

combine_data() {
  sets=$1       # input data subsets
  output_dir=$2 # output dir
  hyp_file=$3   # token or probs

  rm -rf ${output_dir}
  mkdir -p ${output_dir}
  for x in ${sets}; do
    cat data/${x}/text >>${output_dir}/ref.txt
    cat data/${x}/utt2scores >>${output_dir}/utt2scores
    cat ${decode_dir}/${x}/${hyp_file} >>${output_dir}/hyp.txt
  done
}

train_model() {
  action=$1    # train/test
  dir=$2       # data output dir
  data_sets=$3 # datasets
  aug=$4       # true/false
  use_probs=$5 # true/false

  echo "train_model $1 $2 $3 $4 $5"

  hyp_file=token
  if [ "${use_probs}" = "true" ]; then
    hyp_file=probs
  fi

  # combine train sets
  combine_data "${data_sets}" ${dir} ${hyp_file}

  # perform data aug
  ref_file=${dir}/ref.txt
  utt2scores=${dir}/utt2scores
  if [ "${aug}" = "true" ]; then
    ${python} ase/aug_scoring_data.py \
      --text=${dir}/ref.txt \
      --rule-path=${aug_rule_file} \
      --scores=${dir}/utt2scores \
      --output-dir=${dir}

    ref_file=${dir}/ref_aug.txt
    utt2scores=${dir}/utt2scores_aug
  fi

  export PYTHONPATH=ase/
  ${python} ase/scoring_model.py "${action}" \
    ${dir}/hyp.txt \
    ${ref_file} \
    ${scoring_opts} \
    --phone-table=${token_list} \
    --scores=${utt2scores} \
    --model-dir=${model_dir} \
    --output-dir=${dir}
}

use_probs=false
if [[ "${scoring_opts}" == *"--use-probs"* ]]; then
  use_probs=true
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 12: Training scoring model"
  train_model train data/scoring_train "${train_sets}" ${aug_train_data} ${use_probs}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 13: Testing scoring model"
  train_model test data/scoring_test "${test_sets}" ${aug_test_data} ${use_probs}
fi

# if [ ${stage} -ge 20 ]; then
#   for dset in ${test_sets}; do
#     _data="${data_feats}/${dset}"
#     _dir="${asr_exp}/${inference_tag}/${dset}"
#
#     if [ "${dset}" = "test" ]; then # librispeech test
#       ${python} ase/wer.py "${_dir}/token" "${_data}/text" --output-dir=${_dir}
#     else # speechocean test
#       # TODO: use data/so762/utt2scores and data/so762/text
#       ${python} ase/ase_score.py "${_dir}/token" "local/speechocean762/text-phone" \
#         --scores=local/speechocean762/scores.json --output-dir=${_dir}
#       echo "Alignment results saved to ${_dir}/alignment.txt"
#     fi
#   done
# fi
