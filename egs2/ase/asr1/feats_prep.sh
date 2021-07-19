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

stage=1
stop_stage=1000000
python=python3
nj=300
dumpdir=dump
data_feats=${dumpdir}/raw
datasets=train_all_sp

feats_type=raw
audio_format=flac
fs=16k

token_type=word
lm_train_text="data/train_all/text"
nlsyms_txt=none
token_list=data/en_token_list/word/tokens.txt
cleaner=none
g2p=none
oov="<unk>"
blank="<blank>"
sos_eos="<sos/eos>"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [ "${feats_type}" = raw ]; then
    log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

    for dset in ${datasets}; do
      _suf=""
      utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
      rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
      _opts=
      if [ -e data/"${dset}"/segments ]; then
        # "segments" is used for splitting wav files which are written in "wav".scp
        # into utterances. The file format of segments:
        #   <segment_id> <record_id> <start_time> <end_time>
        #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
        # Where the time is written in seconds.
        _opts+="--segments data/${dset}/segments "
      fi
      # shellcheck disable=SC2086
      scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
        "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

      echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
    done

  elif [ "${feats_type}" = fbank_pitch ]; then
    log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}"

    for dset in ${datasets}; do
      _suf=""
      # 1. Copy datadir
      utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

      # 2. Feature extract
      _nj=$(min "${nj}" "$(wc <"${data_feats}${_suf}/${dset}/utt2spk" -l)")
      steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
      utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

      # 3. Derive the the frame length and feature dimension
      scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
        "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

      # 4. Write feats_dim
      head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' |
        cut -d, -f2 >${data_feats}${_suf}/${dset}/feats_dim

      # 5. Write feats_type
      echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
    done

  elif [ "${feats_type}" = fbank ]; then
    log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
    log "${feats_type} is not supported yet."
    exit 1

  elif [ "${feats_type}" = extracted ]; then
    log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
    # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

    for dset in ${datasets}; do
      _suf=""
      # Generate dummy wav.scp to avoid error by copy_data_dir.sh
      awk <data/"${dset}"/cmvn.scp ' { print($1,"<DUMMY>") }' >data/"${dset}"/wav.scp
      utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

      pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - |
        awk '{ print $2 }' | cut -d, -f2 >"${data_feats}${_suf}/${dset}/feats_dim"

      echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
    done

  else
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
  fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
    log "Stage 5: Generate character level token_list from ${lm_train_text}"

    cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/lm_train.txt"

    _opts="--non_linguistic_symbols ${nlsyms_txt}"

    # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
    # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
    ${python} -m espnet2.bin.tokenize_text \
      --token_type "${token_type}" \
      --input "${data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
      --field 2- \
      --cleaner "${cleaner}" \
      --g2p "${g2p}" \
      --write_vocabulary true \
      --add_symbol "${blank}:0" \
      --add_symbol "${oov}:1" \
      --add_symbol "${sos_eos}:-1"
  else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
  fi
fi
