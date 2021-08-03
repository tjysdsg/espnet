#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
    if [ "${b}" -le "${a}" ]; then
      a="${b}"
    fi
  done
  echo "${a}"
}

inference_nj=300    # The number of parallel jobs in decoding.
gpu_inference=false # Whether to perform gpu decoding.
python=python3      # Specify python to execute espnet commands.
dumpdir=dump        # Directory to dump features.
data_feats=${dumpdir}/raw
datasets=train_all_sp
audio_format=flac

expdir=exp # Directory to save experiments.
asr_tag=train_asr_transformer_raw_en_word_sp
asr_exp="${expdir}/asr_${asr_tag}"

inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
inference_config=conf/decode_asr.yaml
inference_tag=decode_asr_asr_model_valid.acc.best

run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

if ${gpu_inference}; then
  _cmd="${cuda_cmd}"
  _ngpu=1
else
  _cmd="${decode_cmd}"
  _ngpu=0
fi

_opts=
if [ -n "${inference_config}" ]; then
  _opts+="--config ${inference_config} "
fi

# 2. Generate run.sh
log "Generate '${asr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 11 using this script"
mkdir -p "${asr_exp}/${inference_tag}"
echo "${run_args} --stage 11 \"\$@\"; exit \$?" >"${asr_exp}/${inference_tag}/run.sh"
chmod +x "${asr_exp}/${inference_tag}/run.sh"

for dset in ${datasets}; do
  _data="${data_feats}/${dset}"
  _dir="${asr_exp}/${inference_tag}/${dset}"
  _logdir="${_dir}/logdir"
  mkdir -p "${_logdir}"

  _feats_type="$(<${_data}/feats_type)"
  if [ "${_feats_type}" = raw ]; then
    _scp=wav.scp
    if [[ "${audio_format}" == *ark* ]]; then
      _type=kaldi_ark
    else
      _type=sound
    fi
  else
    _scp=feats.scp
    _type=kaldi_ark
  fi

  # 1. Split the key file
  key_file=${_data}/${_scp}
  split_scps=""
  _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")
  for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
  done
  # shellcheck disable=SC2086
  utils/split_scp.pl "${key_file}" ${split_scps}

  # 2. Submit decoding jobs
  log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
  # shellcheck disable=SC2086
  ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
    ${python} -m espnet2.bin.asr_inference \
    --ngpu "${_ngpu}" \
    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
    --key_file "${_logdir}"/keys.JOB.scp \
    --asr_train_config "${asr_exp}"/config.yaml \
    --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
    --output_dir "${_logdir}"/output.JOB \
    ${_opts}

  # 3. Concatenates the output files from each jobs
  for f in token token_int score text probs; do
    for i in $(seq "${_nj}"); do
      cat "${_logdir}/output.${i}/1best_recog/${f}"
    done | LC_ALL=C sort -k1 >"${_dir}/${f}"
  done
done
