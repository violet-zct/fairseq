from fairseq.data import FairseqDataset
import numpy as np


class ReferenceDataset(FairseqDataset):
    """ Retrieve sample based on the index
    """

    def __init__(self,
                 dataset,
                 index_dataset,
                 sizes,
                 strides=None):
        # the index_data can be a list
        self.dataset = dataset
        self.index_dataset = index_dataset

        sizes = np.array(sizes)
        if strides is not None:
            for s in strides:
                sizes = (sizes - 1) / s + 1
        self.sizes = sizes

    def __getitem__(self, index):
        return self.dataset[self.index_dataset[index]]

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.sizes[self.index_dataset[index]]