from math import ceil
from typing import List
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, data_feat, data_labels, task_labels, num_classes, scale=True):

        self.num_classes = num_classes
        self.scale = scale

        self.data_feat, self.data_labels, self.tasks_labels = data_feat, data_labels, task_labels
        print('Total number videos ' + str(len(self.data_labels)))

    def to_categorical(self, y):
        """ 1-hot encodes a tensor """
        return np.eye(self.num_classes, dtype='uint8')[y]

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        feat = self.data_feat[idx].float()
        if self.scale:
            feat = feat - torch.min(feat) / (torch.max(feat) - torch.min(feat))
        label = torch.tensor(self.data_labels[idx], dtype=torch.int64)
        return feat, label, len(feat)


class ChunckedVideoDataset(Dataset):
    def __init__(self,
                 data_feat,
                 data_labels,
                 task_labels,
                 num_classes,
                 chunk_size: int = 128,
                 scale: bool = False):

        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.scale = scale

        self.data_feat, self.data_labels, self.tasks_labels = data_feat, data_labels, task_labels

        self.chunk_per_vid = [ceil(len(f)/self.chunk_size) for f in data_feat]
        self.cumulative_chunk_per_vid = np.cumsum(self.chunk_per_vid)
        self.total_chunks = np.sum(self.chunk_per_vid)
        print(f"Total number chunks {self.total_chunks}")

    def to_categorical(self, y):
        """ 1-hot encodes a tensor """
        return np.eye(self.num_classes, dtype='uint8')[y]

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        vid_idx = max(np.searchsorted(self.cumulative_chunk_per_vid, idx, side="right") - 1, 0)
        # print(f"Getting index {idx} Vid index {vid_idx} Length {len(self.data_feat[vid_idx])}")
        
        start_idx = idx - self.cumulative_chunk_per_vid[vid_idx] if vid_idx > 0 else idx
        feat = self.data_feat[vid_idx][start_idx:start_idx+self.chunk_size].float()
        # print(f"Start Index {start_idx} and feat length {len(feat)}")
        if self.scale:
            feat = feat - torch.min(feat) / (torch.max(feat) - torch.min(feat))
        label = torch.tensor(self.data_labels[vid_idx][start_idx:start_idx+self.chunk_size], dtype=torch.int64)
        return feat, label, len(feat)


def collate_videos(data: List):
    """
    Pad sequences that are shorter than the example with largest length 
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.zeros((len(data), max_len))
    # labels = torch.stack(list(labels))
    lengths = torch.tensor(list(lengths))

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat(
            [data[i][0], torch.full(size=(max_len - j, k), fill_value=48)])
        labels[i] = torch.cat(
            [data[i][1], torch.full(size=(max_len - j, ), fill_value=48)])

    return features.float(), labels, lengths.long()


if __name__ == "__main__":
    from Breakfast.read_datasetBreakfast import load_data, read_mapping_dict

    NUM_CLASSES = 48
    COMP_PATH = 'Breakfast/'

    train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder = os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder = os.path.join(COMP_PATH, 'data/')
    mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

    actions_dict = read_mapping_dict(mapping_loc)

    test_data_feat, test_data_labels, test_tasks_labels = load_data(
        test_split, actions_dict, GT_folder, DATA_folder)

    vid_test_dataset = VideoDataset(
        test_data_feat, test_data_labels, test_tasks_labels, NUM_CLASSES)
    vid_test_dataloader = DataLoader(
        vid_test_dataset, batch_size=2, shuffle=True, collate_fn=collate_videos)

    _train_x, _train_y, _train_x_lengths = next(iter(vid_test_dataloader))
    print(_train_x.shape, _train_y.shape, _train_x_lengths)
