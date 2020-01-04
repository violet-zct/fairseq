# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from . import FairseqDataset


class RawLabelDataset(FairseqDataset):

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.sizes = np.array([len(l) for l in labels])

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)

    def size(self, index):
        return self.sizes[index]
