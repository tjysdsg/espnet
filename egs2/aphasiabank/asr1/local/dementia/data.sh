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
    --include_detection_tag(bool): Whether to include detection tags in the beginning of each sentence ("[DEM]")
EOF
)
SECONDS=0

stage=1
stop_stage=5  # stage 6 is for interctc labels
asr_data_dir= # see asr.sh stage 4
tag_insertion=append
include_investigators=false

log "$0 $*"
. utils/parse_options.sh

if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${DEMENTIABANK}" ]; then
  log "Error: \$DEMENTIABANK is not set in db.sh."
  exit 2
fi

tmp=data/local
mkdir -p $tmp

# Things to manually prepare:
# - Download DementiaBank data from https://dementia.talkbank.org/
# - Set ${DEMENTIABANK} to the path to data root (which contains "English", "French", "Greek" etc.) in db.sh
# - Download transcripts from https://dementia.talkbank.org/data/
# - Unzip and copy all *.cha files into ${DEMENTIABANK}/<lang>/<subset>/transcripts

PITT_DIR="${DEMENTIABANK}/English/Pitt/"
# ADRESS2020_DIR="${DEMENTIABANK}/English/ADReSS2020/"

# install pylangacq
pip install --upgrade pylangacq

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Converting *.mp4 and *.mp3 files into .wav"
  log "Will skip converting if the wav file already exists"

  for ext in mp3 mp4; do
    files=$(find "${PITT_DIR}" -type f -name "*.${ext}")
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
fi

# One big difference between AphasiaBank and DementiaBank is that, different segments of a speaker's conversation
# are store in different files. So under Dementia/ or Control/ sub-folders, there are many files with the same name.
# Although there are no overlapping between Dementia/ and Control/ since they contain different sets of speakers.

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Extracting sentence information"
  log "Tag insertion method: ${tag_insertion}"

  # generate data/local/pitt/<story>/text and data/local/pitt/text
  for group in Control Dementia; do
    for story in cookie fluency recall sentence; do
      _opts="--transcript-dir=${PITT_DIR}/transcripts/${group}/${story}/ --out-dir=$tmp/pitt/${group}_${story}/ --tag-insertion=${tag_insertion} --story=${story} "

      if [ "${include_investigators}" = true ]; then
        log "**Including investigators' utterances**"
        _opts+="--include-investigators "
      fi

      python local/dementia/extract_sentence_info.py ${_opts}
    done
  done

  cat $tmp/pitt/*/text >$tmp/pitt/text
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Split data into train, test, and val"

  _opts="--text=$tmp/pitt/text --out-dir=$tmp "
  if [ "${include_investigators}" = true ]; then
    log "**Including investigators' utterances**"
    _opts+="--include-investigators "
  fi

  # split data, generate text and utt2spk
  python local/dementia/split_train_test_val.py ${_opts}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Generating data files of subsets"

  # generate 'wav.scp' and 'segments' files for subsets
  for x in train val test; do
    python local/dementia/generate_wavscp_and_segments.py \
      --data-root="${PITT_DIR}" \
      --reco-list=$tmp/$x/utt.list \
      --out-dir="$tmp/$x/"
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  log "Finalizing data"

  # finalize
  for x in train val test; do
    cp -r $tmp/$x data/
    utils/fix_data_dir.sh data/$x
  done
fi

if [ ${stage} -eq 6 ]; then
  log "Creating utt2aph for interctc aux task"

  python local/dementia/create_aph_tags.py "${asr_data_dir}"
fi
