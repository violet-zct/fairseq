# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import torch
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    MonolingualDataset,
    DocBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
    RawLabelDataset,
    ReferenceDataset,
    ContextLanguagePairDataset
)
from fairseq import models
from . import FairseqTask, register_task
from fairseq import checkpoint_utils
from torch.serialization import default_restore_location
from fairseq.checkpoint_utils import _upgrade_state_dict


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    state = torch.load(
        path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
    )
    args = state['args']
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    state = _upgrade_state_dict(state)
    return state


def load_model(args, path):
    from fairseq import tasks

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    print('| loading model(s) from {}'.format(path))
    state = load_checkpoint_to_cpu(path, None)

    model_args = state['args']
    task = tasks.setup_task(model_args)

    # build model for ensemble
    model = task.build_model(model_args)
    model.load_state_dict(state['model'], strict=True)
    # if args.distributed_world_size > 1:
    #     model = models.DistributedFairseqModel(model_args, model)

    for param in model.parameters():
        param.requires_grad = False

    # Move models to GPU
    # if use_fp16:
    #     model.half()
    if use_cuda:
        model.cuda()

    return model, task, model_args


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        # use bos as the special separator to concat context and the source sentence
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        # tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
    )


@register_task('doc_translation')
class DocumentTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language given a context sequence.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        parser.add_argument('-c', '--context-suffix', type=str,
                            help='path to context: split.context-suffix')
        parser.add_argument('--input-form', type=str, choices=['cat', 'sep'], default='cat')
        parser.add_argument('--context-form', type=str, choices=['doc', 'codes', 'sent'])
        parser.add_argument('--context-compress', type=str, default=None,
                            help='[format: 2,2,2], strides for conv layers in compression')
        parser.add_argument('--context-model-path', type=str, default=None,
                            help='if not None, use external vqvae to compress the document or apply quantization on codes')
        parser.add_argument('--encode-code', type=int, default=0,
                            help='if 1, use the code book of the pretrained vqvae to encode the codes')
        parser.add_argument('--fix-code-book', type=int, default=0)
        parser.add_argument('--window_size', type=int, default=3)
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, ctx_dict, ctx_model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.ctx_dict = ctx_dict
        self.ctx_model = ctx_model

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        if args.context_model_path is not None:
            context_model, context_task, model_args = load_model(args, args.context_model_path)
            context_dict = context_task.dictionary
            args.codebook_size = model_args.bottom_latent_k
        else:
            context_dict = src_dict
            context_model = None

        return cls(args, src_dict, tgt_dict, context_dict, context_model)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        context_compress = None
        if self.args.context_form != 'code' and self.args.context_compress is not None:
            context_compress = list(map(int, self.args.context_compress.strip().split(',')))

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        langpair_dataset = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=(self.args.input_form == 'cat'),
        )

        ctx_path = os.path.join(data_path, split + '.' + self.args.context_suffix)
        if self.args.context_form == 'codes':
            # ctx_dataset = RawLabelDataset([torch.IntTensor(map(int, line.strip().split())) for line in open(ctx_path).readlines()])
            # ctx_dataset = ReferenceDataset(ctx_dataset, index_list, sizes=ctx_dataset.sizes)
            raise NotImplementedError
        elif self.args.context_form == 'sent':
            ctx_dataset = langpair_dataset.src
        elif self.args.context_form == 'doc' or self.args.context_form == 'window':
            ctx_dataset = data_utils.load_indexed_dataset(
                ctx_path, self.ctx_dict, self.args.dataset_impl, combine=False)  # in fact, the binary datasets doesn't need the dict
            if ctx_dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {}".format(os.path.join(data_path, ctx_path))
                )

            dataset = DocBlockDataset(
                ctx_dataset,
                ctx_dataset.sizes,
                self.args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode='complete_doc',
                include_targets=False,
                context_mode=self.args.context_form,
                window_size=self.args.window_size,
            )
            print("| Loaded {} documents/context!".format(len(dataset)))
            assert len(dataset) == len(ctx_dataset)
            # return {'id': index, 'source': source, 'target': target}: target = None
            ctx_dataset = MonolingualDataset(
                dataset,
                dataset.sizes,
                self.ctx_dict,
                self.ctx_dict,
                add_eos_for_other_targets=False,
                shuffle=False,
                targets=None,
                add_bos_token=False,
            )
        else:
            raise ValueError

        self.datasets[split] = ContextLanguagePairDataset(ctx_dataset, langpair_dataset, input_form=self.args.input_form,
                                                          context_form=self.args.context_form,
                                                          context_compress=context_compress,
                                                          context_dict=self.ctx_dict)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # this function is called in interactive.py and hub_utils.py
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
