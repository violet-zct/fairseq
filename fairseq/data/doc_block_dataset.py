# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data import FairseqDataset, plasma_utils


class DocBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.
        This class reads in the document (optional: limited by the tokens_per_sample).
    Return the context of a sentence instance, the context can be document block.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
        context_mode (str, optional): Mode used for extracting context for a given sentence index.
            - 'doc': the context is document block, the block length is determined by the block_size
            - 'window': the context are surrounding sentences of window_size, in this case, the block_siz is a very large number.
        window_size (int, optional): use window_size * 2 + 1 sentences as context
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode='complete_doc',
        include_targets=False,
        document_sep_len=1,
        context_mode='doc',
        window_size=3,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                'Please build Cython components with: `pip install --editable .` '
                'or `python setup.py build_ext --inplace`'
            )

        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else 'none'

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        slice_indices = _get_slice_indices_fast(sizes, break_mode, block_size, document_sep_len)

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(
                        len(sizes), dtype=np.long
                    ),  # starting offset within starting index
                    np.arange(len(sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes,
                slice_indices,
            )
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

        self.context_mode = context_mode
        self.window_size = window_size

        context_index, _sizes = self.rebuild_index()
        self._context_index = plasma_utils.PlasmaArray(np.array(context_index))
        self._sizes = plasma_utils.PlasmaArray(np.array(_sizes))

    def rebuild_index(self):
        new_index = []
        sizes = []
        cur_new_idx = 0
        # there are edge cases, that there are extremely long sentences, that makes the block extremely long
        if self.context_mode == 'doc':
            for doc_idx in range(len(self.slice_indices)):
                start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[doc_idx]
                if start_offset > 0:
                    true_start_idx = min(start_ds_idx+1, end_ds_idx)
                else:
                    true_start_idx = start_ds_idx
                for sidx in range(start_ds_idx, end_ds_idx + 1):
                    if sidx == start_ds_idx and start_offset > 0:
                        continue
                    new_index.append((doc_idx, true_start_idx, end_ds_idx))
                    sizes.append(sum([len(self.dataset[ii]) for ii in range(true_start_idx, end_ds_idx+1)]))
                    cur_new_idx += 1
        elif self.context_mode == 'window':
            # in this, we read each document is a complete one without breaking in the middle, set tokens-per-samples to be a large number
            for doc_idx in range(len(self.slice_indices)):
                start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[doc_idx]
                for sidx in range(start_ds_idx, end_ds_idx + 1):
                    new_index.append((doc_idx, max(start_ds_idx, sidx-self.window_size), min(end_ds_idx, sidx+self.window_size)))
                    sizes.append(sum([len(self.dataset[ii]) for ii in range(new_index[-1][1], new_index[-1][2] + 1)]))
                    cur_new_idx += 1
                if end_ds_idx < len(self.dataset)-1:
                    assert len(self.dataset[end_ds_idx+1]) == 1
        else:
            raise ValueError
        return new_index, sizes

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    @property
    def context_index(self):
        return self._context_index.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        doc_idx, start_ds_idx, end_ds_idx = self.context_index[index]
        item = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            source = torch.cat([item.new([self.eos]), item[0: len(item)-1]])
            past_target = torch.cat(
                [item.new([self.pad, self.eos]), item[0: len(item) - 2]]
            )
            return source, item, past_target

        return item

    def __len__(self):
        return len(self.context_index)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for _, start_ds_idx, end_ds_idx in [self.context_mode[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )