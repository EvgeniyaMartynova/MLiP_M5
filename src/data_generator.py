from torch.utils.data import DataLoader, Dataset, IterableDataset
from itertools import islice
import numpy as np
import torch

class SlidingWindow(Dataset):
    def __init__(self, timeseries, train_seq_len=28):
        super(SlidingWindow).__init__()
        self.timeseries = timeseries
        self.train_seq_len = train_seq_len
        self.input_size = timeseries.shape[1]
        self.num_classes = timeseries.shape[1]

    def __len__(self):
        return len(self.timeseries) - self.train_seq_len

    def __getitem__(self, idx):
        x = self.timeseries[idx:idx + self.train_seq_len, :]
        y = self.timeseries[idx + self.train_seq_len, :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def main():
    # Demonstration of data generation
    # define input sequence
    in_seq1 = np.array([x for x in range(0, 100, 10)])
    in_seq2 = np.array([x for x in range(5, 105, 10)])
    in_seq3 = np.array([x for x in range(10, 110, 10)])
    out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = np.hstack((in_seq1, in_seq2, in_seq3, out_seq))
    print(dataset.shape[1])

    print(dataset)
    print("start interating")
    sliding_window_dataset = SlidingWindow(dataset, train_seq_len=3)
    data_loader = DataLoader(sliding_window_dataset)
    for x, y in data_loader:
        print("Sequence:")
        print(x)
        print("Label:")
        print(y)


if __name__ == '__main__':
    main()