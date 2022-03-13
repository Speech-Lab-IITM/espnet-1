#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export ftp_proxy=http://172.50.0.50:9090
export https_proxy=http://172.50.0.50:9090
export http_proxy=http://172.50.0.50:9090

train_set="train_clean_360_HuBERT6_km500"
valid_set="dev_HuBERT6_km500"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_scratch.yaml
dictdir=data/HuBERT6_km500_token_list_iter1/word

./joint_pretraining.sh \
    --lang en \
    --ngpu 8 \
    --max_wav_duration 30 \
    --dictdir "${dictdir}" \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
