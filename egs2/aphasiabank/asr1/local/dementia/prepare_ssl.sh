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

stage=1
stop_stage=100
nj=1

log "$0 $*"
. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # 'dump_hubert_feature.sh' reads wave files from a common dir, so we symbolically link dump/raw/test in dump/raw/org
  if [ ! -d dump/raw/org/test ]; then
    ln -s ../test dump/raw/org/test
  fi

  # S3PRL HuBERT large example
  local/dementia/dump_ssl_feature.sh \
    --feat_dir ssl_feats \
    --datadir dump/raw/org \
    --train_set train_sp \
    --dev_set val \
    --test_sets "test" \
    --use_gpu true \
    --nj ${nj} \
    --feature_type s3prl \
    --s3prl_upstream_name wavlm_large \
    --layer 24
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for dset in train_sp val test; do
    utils/copy_data_dir.sh dump/raw/org/${dset} data/${dset}_ssl
    cp ssl_feats/s3prl/${dset}/feats.scp data/${dset}_ssl
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  ./asr.sh --stage 3 --stop_stage 3 \
    --train_set train_sp_ssl --valid_set val_ssl --test_sets "test_ssl" \
    --feats_type "extracted" --nj 4
fi
