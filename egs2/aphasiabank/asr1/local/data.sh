#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(
  cat <<EOF
Usage: $0

Options:
    --include_control (bool): Whether to include control group data
    --include_aphasia_type (bool): Whether to include aphasia type in the beginning of each sentence ("[wernicke]")
EOF
)
SECONDS=0

stage=1
stop_stage=7  # stage 8 is for interctc labels
include_control=false
include_aphasia_type=false
include_lang_id=false
languages="English French"
asr_data_dir=  # see asr.sh stage 4
tag_insertion=append

log "$0 $*"
. utils/parse_options.sh

if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${APHASIABANK}" ]; then
  log "Error: \$APHASIABANK is not set in db.sh."
  exit 2
fi

tmp=data/local
mkdir -p $tmp

# Things to manually prepare:
# - Download AphasiaBank data from https://aphasia.talkbank.org/
# - Set ${APHASIABANK} to the path to data root (which contains "English", "French", "Greek" etc.) in db.sh
# - Download transcripts from https://aphasia.talkbank.org/data/
# - Unzip and copy all *.cha files into ${APHASIABANK}/<lang>/transcripts

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Converting *.mp4 and *.mp3 files into .wav"
  log "Will skip converting if the wav file already exists"

  for lang in ${languages}; do
    for ext in mp3 mp4; do
      for subdir in Aphasia Control; do
        files=$(find "${APHASIABANK}/${lang}/${subdir}" -type f -name "*.${ext}")
        for f in $files; do
          filename=$(basename -- "$f")
          dir=$(dirname "$f")
          filename="${filename%.*}"

          if [ ! -f "$dir/${filename}.wav" ]; then
            echo "Converting $f to $dir/${filename}.wav"
            ffmpeg -y -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 "${dir}/${filename}.wav" &>/dev/null
          # else
          #   echo "Skip converting $f to $dir/${filename}.wav as it already exists"
          fi
        done
      done
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Extracting speaker information"

  # install pylangacq
  pip install --upgrade pylangacq

  # generate data/spk_info.txt
  for lang in ${languages}; do
    python local/extract_speaker_info.py --transcript-dir="${APHASIABANK}/${lang}/transcripts" --out-dir=data/${lang}
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Extracting sentence information"

  # generate data/local/<lang>/text
  log "Tag insertion method: ${tag_insertion}"
  for lang in ${languages}; do
    _opts="--transcript-dir=${APHASIABANK}/${lang}/transcripts --out-dir=$tmp/${lang} --tag-insertion=${tag_insertion} "

    if "${include_aphasia_type}"; then
      log "**Including the aphasia type**"
      _opts+="--spk2aphasia-type=data/${lang}/spk2aphasia_type "
    fi

    if "${include_lang_id}"; then
      log "**Including language id**"
      _opts+="--lang=$lang "
    fi

    python local/extract_sentence_info.py ${_opts}
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Split data into train, test, and val"

  # combine multiple languages
  for file in text utt2spk; do
    rm -f $tmp/$file
    touch $tmp/$file
    for lang in ${languages}; do
      cat $tmp/$lang/$file >>$tmp/$file
    done
  done

  # split data, generate text and utt2spk
  python local/split_train_test_val.py --text=$tmp/text --out-dir=$tmp
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  log "Generating data files of the entire database"

  # generate wav.scp of all data
  find ${APHASIABANK}/*/Aphasia -type f -name "*.wav" >$tmp/all_wav.list

  # add control group data if needed
  if "${include_control}"; then
    log "**Including the control group**"
    find ${APHASIABANK}/*/Control -type f -name "*.wav" >>$tmp/all_wav.list
  fi

  awk -F'/' '{printf("%s\t%s\n",$NF,$0)}' $tmp/all_wav.list >$tmp/all_wav.scp
  sed -i 's/\.wav//' $tmp/all_wav.scp
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  log "Generating data files of subsets"

  # generate 'wav.scp' and 'segments' files for subsets
  for x in train val test; do
    python local/generate_wavscp_and_segments.py \
      --wavscp=$tmp/all_wav.scp \
      --reco-list=$tmp/$x/utt.list \
      --out-dir="$tmp/$x/"
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  log "Finalizing data"

  # finalize
  for x in train val test; do
    cp -r $tmp/$x data/
    utils/fix_data_dir.sh data/$x
  done
fi

if [ ${stage} -eq 8 ]; then
  log "Creating utt2aph for interctc aux task"

  python local/create_aph_tags.py "${asr_data_dir}"
fi
