import json
import os
import random
from pathlib import Path

import model_train
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

TRAIN_RATIO = model_train.TRAIN_RATIO


class CustomDataset(Dataset):
    def __init__(self, data_root: str, segmet_length=128):
        self.data_root = data_root
        self.segmet_length = segmet_length

        mapping_path = Path(data_root) / "mapping.json"
        mapping = json.load(mapping_path.open())
        # id和标签的对照表
        self.speaker2id = mapping["speaker2id"]

        metadata_path = Path(data_root) / "metadata.json"
        # id和其对应数据路径/特征对照表
        metadata = json.load(open(metadata_path))["speakers"]

        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                # data路径和data标签
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]

        mel = torch.load(os.path.join(self.data_root, feat_path))

        if len(mel) > self.segmet_length:
            start = random.randint(0, len(mel) - self.segmet_length)
            mel = torch.FloatTensor(mel[start : start + self.segmet_length])
        else:
            mel = torch.FloatTensor(mel)
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


# dataloader
def collate_batch(batch):
    mel, speaker = zip(*batch)
    # 将元组列表“解包成”多个列表
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)


def create_dataset(data_root: str):
    """Generate dataset"""
    my_dataset = CustomDataset(data_root)
    # 拆分数据集为training集和valid集
    trainlen = int(TRAIN_RATIO * len(my_dataset))
    lengths = [trainlen, len(my_dataset) - trainlen]
    speaker_num = my_dataset.get_speaker_number()

    train_dataset, valid_dataset = random_split(my_dataset, lengths)
    return train_dataset, valid_dataset, speaker_num


def create_dataloader(
    data_root: str, batch_size: int, num_workers: int, shuffle: bool = True
):
    """Generate dataloader"""
    trainset, validset, speaker_num = create_dataset(data_root)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num
