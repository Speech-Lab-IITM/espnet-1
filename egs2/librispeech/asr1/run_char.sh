#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
#test_sets="test_clean test_other dev_clean dev_other"
#test_sets="test_other dev_clean dev_other"
test_sets="test_clean"

asr_config=conf/tuning/train_hubert_char.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr_char.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --token_type char \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
