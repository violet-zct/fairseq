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
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask


@register_task("VQVAE_language_modeling")
class VQVAELanguageModelingTask(LanguageModelingTask):

    # @staticmethod
    # def add_args(parser):
    #     """Add task-specific arguments to the parser."""
    #     # fmt: off

    def extract_codes(self, sample, model):
        model.eval()
        tokens, lengths = sample['target'], sample['net_input']['src_lengths']
        emb_inds = model.extract_codes(tokens, lengths) # B x T
        return emb_inds

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)