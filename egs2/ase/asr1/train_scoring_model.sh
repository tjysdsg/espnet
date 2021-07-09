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
test_sets="so762_test libri_scoring_test"
model_path=exp/scoring_train/model.pkl # model save path
aug_test_data=false

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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 12: Training scoring model"

  # use probs file if --use-probs
  hyp_file=token
  if [[ "${scoring_opts}" == *"--use-probs"* ]]; then
    hyp_file=probs
  fi

  # combine train sets
  _dir=data/scoring_train
  combine_data "${train_sets}" ${_dir} ${hyp_file}

  # perform data aug
  ${python} ase/aug_scoring_data.py \
    --text=${_dir}/ref.txt \
    --scores=${_dir}/utt2scores \
    --output-dir=${_dir}

  export PYTHONPATH=ase/
  ${python} ase/scoring_model.py train \
    ${_dir}/hyp.txt \
    ${_dir}/ref_aug.txt \
    ${scoring_opts} \
    --phone-table=${token_list} \
    --scores=${_dir}/utt2scores_aug \
    --model-path=${model_path} \
    --output-dir=${_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 13: Testing scoring model"

  # use probs file if --use-probs
  hyp_file=token
  if [[ "${scoring_opts}" == *"--use-probs"* ]]; then
    hyp_file=probs
  fi

  # combine test sets
  _dir=data/scoring_test
  combine_data "${test_sets}" ${_dir} ${hyp_file}

  if [ "${aug_test_data}" = "true" ]; then
    ${python} ase/aug_scoring_data.py \
      --text=${_dir}/ref.txt \
      --scores=${_dir}/utt2scores \
      --output-dir=${_dir}
  fi

  # Calculate ASE metrics
  export PYTHONPATH=ase/
  ${python} ase/scoring_model.py test \
    ${_dir}/hyp.txt \
    ${_dir}/ref.txt \
    ${scoring_opts} \
    --phone-table=${token_list} \
    --scores=${_dir}/utt2scores \
    --model-path=${model_path} \
    --output-dir=${_dir}
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
