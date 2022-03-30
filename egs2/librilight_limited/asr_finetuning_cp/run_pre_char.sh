#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

source path.sh

train_set="train_10h"
valid_set="dev"
test_sets="test_clean test_other"
#test_sets="test_clean test_other dev_clean dev_other"

asr_config=
#inference_config=conf/decode_asr_char.yaml
inference_asr_model=valid.loss.best.pth

./asr.sh \
    --stage 11 \
    --stop_stage 13 \
    --lang en \
    --ngpu 2 \
    --token_type char \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
