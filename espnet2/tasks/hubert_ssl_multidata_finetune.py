#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert
import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,  ####Change
)
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.multidata_hubert.espnet_model import HubertPretrainModel #### For multidata
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq.models.hubert.hubert import HubertConfig

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend, sliding_window=SlidingWindow),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
#### Change 
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        hubert=FairseqHubertEncoder,
    ),
    type_check=AbsEncoder,
    default="hubert",
)


class MultiDataHubertTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        # Added for multidata configuration
        group.add_argument(
            "--data_specific_blocks",
            type=int_or_none,
            default=3,
            help="The number of data specific encoder blocks",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(HubertPretrainModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        parser.add_argument(
            "--pred_masked_weight",
            type=float,
            default=1.0,
            help="weight for predictive loss for masked frames",
        )
        parser.add_argument(
            "--pred_nomask_weight",
            type=float,
            default=0.0,
            help="weight for predictive loss for unmasked frames",
        )
        parser.add_argument(
            "--loss_weights",
            type=float,
            default=0.0,
            help="weights for additional loss terms (not first one)",
        )
        parser.add_argument(
            "--hubert_dict",
            type=str,
            default="./dict.txt",
            help="word-based target dictionary for Hubert pretraining stage",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text",)
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("dataset_type", ) # changed for multidata
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> HubertPretrainModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder - Common
        # For multidata
        args.encoder_conf['num_blocks'] = (args.encoder_conf['num_blocks'] if args.encoder_conf['num_blocks'] is not None else 12) - args.data_specific_blocks
        encoder_class = encoder_choices.get_class(args.encoder)
        ###Change
        encoder = encoder_class(
            input_size=input_size,
            **args.encoder_conf,
        )

        # For multidata
        args.encoder_conf['num_blocks'] = args.data_specific_blocks
        """
        cfg_overides = {
            "encoder_embed_dim": args.encoder_conf['output_size'],
            "encoder_ffn_embed_dim": args.encoder_conf['linear_units'],
            "encoder_attention_heads": args.encoder_conf['attention_heads'],
            "encoder_layers": args.encoder_conf['num_blocks'],
            "final_dim": args.encoder_conf['output_size'],
            "dropout": args.encoder_conf['dropout_rate'],
            "attention_dropout": args.encoder_conf['attention_dropout_rate'],
            "label_rate": args.encoder_conf['label_rate'],
        }
              
        cfg_overides = {**cfg_overides, **args.encoder_conf}
        cfg = HubertConfig()
        for key, value in cfg_overides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        """
        data_specific_encoders = []
        """
        data_specific_encoders.append(TransformerEncoder(cfg))
        data_specific_encoders.append(TransformerEncoder(cfg))
        data_specific_encoders = torch.nn.ModuleList(data_specific_encoders)
        """
        Transformer1=[]
        Transformer2=[]
        for i in range(args.data_specific_blocks):
            """
            Transformer.append(TransformerSentenceEncoderLayer("embedding_dim": args.encoder_conf['output_size'],"ffn_embedding_dim": args.encoder_conf['linear_units'],"num_attention_heads": args.encoder_conf['attention_heads'],"dropout": args.encoder_conf['dropout_rate'],"attention_dropout": args.encoder_conf['attention_dropout_rate'],"activation_dropout": 0.1,"activation_fn": "relu","layer_norm_first": False))
            
            Transformer1.append(TransformerSentenceEncoderLayer(args.transformer_conf['output_size'],args.transformer_conf['linear_units'],args.transformer_conf['attention_heads'],args.transformer_conf['dropout_rate'],args.transformer_conf['attention_dropout_rate'],0.1,"relu",False))
            Transformer2.append(TransformerSentenceEncoderLayer(args.transformer_conf['output_size'],args.transformer_conf['linear_units'],args.transformer_conf['attention_heads'],args.transformer_conf['dropout_rate'],args.transformer_conf['attention_dropout_rate'],0.1,"relu",False))
            """
            Transformer1.append(TransformerSentenceEncoderLayer(args.encoder_conf['output_size'],args.encoder_conf['linear_units'],args.encoder_conf['attention_heads'],args.encoder_conf['dropout_rate'],args.encoder_conf['attention_dropout_rate'],0.1,"relu",False))

            Transformer2.append(TransformerSentenceEncoderLayer(args.encoder_conf['output_size'],args.encoder_conf['linear_units'],args.encoder_conf['attention_heads'],args.encoder_conf['dropout_rate'],args.encoder_conf['attention_dropout_rate'],0.1,"relu",False))
        from fairseq.modules import LayerNorm
        import torch.nn as nn

        Transformer1.append(LayerNorm(args.encoder_conf['output_size']))
        Transformer1.append(nn.Linear(args.encoder_conf['output_size'], args.encoder_conf['final_dim']))
        Transformer2.append(LayerNorm(args.encoder_conf['output_size']))
        Transformer2.append(nn.Linear(args.encoder_conf['output_size'], args.encoder_conf['final_dim']))
        Transformer1=torch.nn.ModuleList(Transformer1)
        Transformer2=torch.nn.ModuleList(Transformer2)
        data_specific_encoders.append(Transformer1)
        data_specific_encoders.append(Transformer2)
        data_specific_encoders = torch.nn.ModuleList(data_specific_encoders)        
        label_embs_concat_0 = torch.nn.Parameter(torch.FloatTensor(sum(encoder.encoder.num_classes), args.encoder_conf['final_dim']))
        label_embs_concat_1 = torch.nn.Parameter(torch.FloatTensor(sum(encoder.encoder.num_classes), args.encoder_conf['final_dim']))

        # 8. Build model
        model = HubertPretrainModel(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            data_specific_encoders=data_specific_encoders,
            label_embs_concat_0=label_embs_concat_0,
            label_embs_concat_1=label_embs_concat_1,
            token_list=token_list,
            **args.model_conf,
        )

        # 9. Initialize  #####set it to None
        if args.init is not None:   
            initialize(model, args.init)
        ####Change Extra loading of data_specific_layers happen here, hubert encoder happens while model building
        data_specific_encoders.load_state_dict(torch.load('/nlsasfs/home/nltm-pilot/metilda/multitask_gen_nptel_100_100_ssl/multitask/b.pt'))
       
        assert check_return_type(model)
        return model
