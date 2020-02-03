# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    MonolingualDataset,
    DocBlockDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask

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


@register_task("soft_language_modeling")
class SoftLanguageModelingTask(LanguageModelingTask):
    # load a pretrained vqvae model and generate soft code distributions, learn a LM to fit
    def __init__(self, args, vqvae_model, vqvae_task, vqvae_args):
        super().__init__(args, vqvae_task.dictionary, vqvae_task.output_dictionary, vqvae_task.targets)
        self.vqvae_model = vqvae_model
        self.vqvae_task = vqvae_task
        self.vqvae_args = vqvae_args

        self.padding_idx = vqvae_task.dictionary.pad_index

    def add_args(parser):
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--context-model-path', type=str, default=None,
                            help='if not None, use external vqvae to compress the document or apply quantization on codes')
        parser.add_argument('--code-extract-strategy', type=str, default=None,
                            help=['soft', 'argmax', 'topp', 'topk', 'full'])

    @classmethod
    def setup_task(cls, args, **kwargs):
        context_model, context_task, model_args = load_model(args, args.context_model_path)
        # context_dict = context_task.dictionary
        model_args.data = args.data
        args.codebook_size = model_args.bottom_latent_k
        return cls(args, context_model, context_task, model_args)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        use_ctx_dataset = getattr(self.vqvae_args, 'use_context_dataset', 0)
        paths = self.vqvae_args.data.split(":")
        assert len(paths) > 0

        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.vqvae_args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        if use_ctx_dataset:
            dataset = DocBlockDataset(
                dataset,
                dataset.sizes,
                self.vqvae_args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.vqvae_args.sample_break_mode,
                include_targets=True,
                context_mode=self.vqvae_args.context_mode,
                window_size=self.vqvae_args.window_size,
            )
        else:
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.vqvae_args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.vqvae_args.sample_break_mode,
                include_targets=True,
            )

        add_eos_for_other_targets = (
                self.vqvae_args.sample_break_mode is not None
                and self.vqvae_args.sample_break_mode != "none"
        )

        self.datasets[split] = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.vqvae_args.add_bos_token,
        )

    def sample_and_generate(self):
        pass

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)