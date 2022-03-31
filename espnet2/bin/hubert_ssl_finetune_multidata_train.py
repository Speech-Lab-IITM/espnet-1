#!/usr/bin/env python3
from espnet2.tasks.hubert_ssl_multidata_finetune import MultiDataHubertTask


def get_parser():
    parser = HubertTask.get_parser()
    return parser


def main(cmd=None):
    """Hubert pretraining.

    Example:
        % python hubert_train.py asr --print_config --optim adadelta \
                > conf/hubert_asr.yaml
        % python hubert_train.py --config conf/train_asr.yaml
    """
    MultiDataHubertTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
