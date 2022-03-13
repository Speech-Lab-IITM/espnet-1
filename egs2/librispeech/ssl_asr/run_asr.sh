#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960_HuBERT6_km500"
valid_set="dev_HuBERT6_km500"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_scratch.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
