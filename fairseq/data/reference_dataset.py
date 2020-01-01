from fairseq.data import FairseqDataset


class ReferenceDataset(FairseqDataset):
    """ Retrieve sample based on the index
    """

    def __init__(self,
                 dataset,
                 index_dataset):
        # the index_data can be a list
        self.dataset = dataset
        self.index_dataset = index_dataset

    def __getitem__(self, index):
        return self.dataset[self.index_dataset[index]]

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.dataset.size(index)