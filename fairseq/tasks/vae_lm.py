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

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        self.prefill = getattr(args, 'prefill', 1)
        self.strides = list(map(int, args.bottom_conv_stride.split(',')))

    def extract_codes(self, sample, model):
        model.eval()
        with torch.no_grad():
            tokens, lengths = sample['target'], sample['net_input']['src_lengths']
            _, _, _, _, codes = model.forward_encoder(tokens, lengths) # B x T
            emb_inds = codes['bottom_codes']
        return emb_inds

    def reconstruct(self, sample, model, generator, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate([model], sample, prefix_tokens=prefix_tokens)

    def sampling(self, dummy_sample, codes, code_masks, model, generator, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate([model], dummy_sample, codes=codes,
                                      code_masks=code_masks, prefix_tokens=prefix_tokens)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.vae_sequence_generator import VAESequenceGenerator
            return VAESequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 1000),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0

        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            prefill=self.prefill,
            strides=self.strides
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode="eos",
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
                add_bos_token=self.args.add_bos_token,
                prefill=self.prefill,
                strides=self.strides,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample["net_input"]["src_tokens"]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)