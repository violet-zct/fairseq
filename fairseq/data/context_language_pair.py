# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, ctx_pad_idx, left_pad_source=False, left_pad_target=False,
    input_feeding=True, input_form='cat', context_form='doc', context_compress=None,
    quantitize=False,
):
    # quantitize means encode the codes with the original code book embeddings
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_ctx(pad):
        return data_utils.collate_tokens(
            [s['context'] for s in samples], pad, left_pad=False,
        )

    def compute_compressed_lengths(lengths, strides):
        for s in strides:
            lengths = (lengths - 1) / s + 1
        return lengths

    id = torch.LongTensor([s['id'] for s in samples])

    def cat_merge_source():
        ctx_key = 'context'
        sent_key = 'source'
        context = None
        context_lengths = None
        src_tokens = None
        src_mask = None
        code_lengths = None
        process_context = None

        if input_form == 'cat' and context_form == 'doc' and context_compress is None:
            context_lengths = torch.LongTensor([s['context'].numel() for s in samples])
            # source = [original doc; <bos>; sent], no context, because they share vocab
            src_tokens = data_utils.collate_tokens([torch.cat([s[ctx_key], s[sent_key]]) for s in samples], pad_idx, left_pad=False,)
            src_mask = torch.arange(src_tokens.size(1), device=src_tokens.device).type_as(src_tokens).expand(
                len(src_tokens), src_tokens.size(1))
            src_mask = src_mask < context_lengths.unsqueeze(1)
        elif input_form == 'cat' and context_form == 'doc' and context_compress is not None:
            # source = [pesudo compressed doc; <bos>; sent], context = doc
            context = merge_ctx(ctx_pad_idx)
            code_lengths = compute_compressed_lengths(context_lengths, context_compress)
            process_context = 'compress_doc'
            if quantitize:
                process_context = 'quantitize_doc'
        elif input_form == 'cat' and context_form == 'codes':
            # source = [pesudo codes; <bos>; sent], context = codes
            context = merge_ctx(-1)
            code_lengths = context.neq(-1).sum(1)
            context[context.eq(-1)] = 0
            if quantitize:
                process_context = 'quantitize_code'
        elif input_form == 'sep':
            # source = sent, context = doc / codes
            src_tokens = merge('source', left_pad=left_pad_source)
            context = merge_ctx(ctx_pad_idx)
            context_lengths = torch.LongTensor([s.numel() for s in samples])
        else:
            raise ValueError

        if code_lengths is not None:
            context_lengths = code_lengths
            src_tokens = data_utils.collate_tokens(
                [torch.cat([s[sent_key][0].new(m.item()).fill_(pad_idx), s[sent_key]])
                 for s, m in zip(samples, code_lengths)], pad_idx, left_pad=False, )
            src_mask = torch.arange(src_tokens.size(1), device=src_tokens.device).type_as(src_tokens).expand(
                len(src_tokens), src_tokens.size(1))
            src_mask = src_mask < code_lengths.unsqueeze(1)
        return src_tokens, src_mask, context, context_lengths, process_context

    src_tokens, src_mask, context, context_lengths, process_context = cat_merge_source()
    # sort by descending source length
    src_lengths = torch.LongTensor([s.numel() for s in src_tokens])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_mask': src_mask,
            'context': context,
            'context_lengths': context_lengths
        },
        'target': target,
        'process_context': process_context
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class ContextLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets with a reference data set for context.
    """

    def __init__(
        self, ctx_dataset, langpair_dataset,
        input_form='cat', context_form='doc', context_compress=None, context_dict=None,
        encode_code=False, shuffle=True
    ):
        self.ctx_dataset = ctx_dataset
        self.langpair_dataset = langpair_dataset

        assert len(self.ctx_dataset) == len(langpair_dataset)

        self.input_form = input_form
        self.context_form = context_form
        if context_compress is not None:
            context_compress = map(int, context_compress.strip().split(','))
        self.context_compress = context_compress

        self.context_dict = context_dict
        self.encode_code = encode_code

        self.shuffle = shuffle

    def __getitem__(self, index):
        tgt_item = self.langpair_dataset.tgt[index]
        src_item = self.langpair_dataset.src[index]

        ctx_item = self.ctx_dataset[index]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'context': ctx_item['source'] if isinstance(ctx_item, dict) else ctx_item,
        }

        return example

    def __len__(self):
        return len(self.ctx_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """

        return collate(
            samples, pad_idx=self.langpair_dataset.src_dict.pad(), eos_idx=self.langpair_dataset.src_dict.eos(),
            ctx_pad_idx=self.context_dict.pad(),
            left_pad_source=False, left_pad_target=False,
            input_feeding=True,
            input_form=self.input_form, context_form=self.context_form, context_compress=self.context_compress,
            quantitize=self.encode_code
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.size(index))

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        src_size, tgt_size = self.langpair_dataset.size(index)
        src_size = max(src_size, self.ctx_dataset.size(index)) if self.input_form == 'sep' else (src_size + self.ctx_dataset.size(index))
        return (src_size, tgt_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.langpair_dataset.tgt_sizes is not None:
            indices = indices[np.argsort(self.langpair_dataset.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.langpair_dataset.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.langpair_dataset.src, 'supports_prefetch', False)
            and (getattr(self.langpair_dataset.tgt, 'supports_prefetch', False) or self.langpair_dataset.tgt is None)
        )

    def prefetch(self, indices):
        self.langpair_dataset.src.prefetch(indices)
        if self.langpair_dataset.tgt is not None:
            self.langpair_dataset.tgt.prefetch(indices)
