import csv
import numpy
import torch
from torch.utils.data import DataLoader, Dataset, random_split

TRAIN_RATIO = 0.9


def csv_fileReader(path: str) -> numpy.ndarray:
    with open(path) as file:
        csv_list = list(csv.reader(file))
        data = numpy.array(csv_list)
        data = data[1:, 1:]
        data = data.astype(float)
        return data


class MyDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        my_data = csv_fileReader(path)
        self.data = torch.tensor(my_data[:, :-1], dtype=torch.float32)
        self.dim = self.data.shape[1]
        self.targets = torch.tensor(my_data[:, -1], dtype=torch.float32)
        # 检验数据集是否有误
        if self.data.shape[0] == 0:
            raise ValueError("数据集为空，请检查路径或数据格式")
        if self.targets.shape[0] != self.data.shape[0]:
            raise ValueError("数据集的标签数量与特征数量不匹配")
        if len(self.targets.shape) >= 2:
            raise ValueError("数据集的标签维度不正确，应该为单列")

        # 从第41列开始数据需要进行标准化处理,是对每一列（特征:dim=1）计算均值和标准差（根据数据集特点填写）
        self.data[:, 40:] = (
            self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)
        ) / self.data[:, 40:].std(dim=0, keepdim=True)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


class MyTestDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        my_data = csv_fileReader(path)
        self.data = torch.tensor(my_data, dtype=torch.float32)
        # 检验数据集是否有误
        if self.data.shape[0] == 0:
            raise ValueError("数据集为空，请检查路径或数据格式")
        if self.data.shape[1] < 92:
            raise ValueError("数据集的特征维度不足，请检查数据格式")
        self.data[:, 40:] = (
            self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)
        ) / self.data[:, 40:].std(dim=0, keepdim=True)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_dataset(path: str, dataset_type: str = "train") -> Dataset:
    train_size = int(len(MyDataset(path)) * TRAIN_RATIO)
    val_size = len(MyDataset(path)) - train_size

    train_dataset, val_dataset = random_split(
        MyDataset(path),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42069),
    )

    if dataset_type == "train":
        return train_dataset
    elif dataset_type == "dev":
        return val_dataset
    elif dataset_type == "test":
        return MyTestDataset(path)
    else:
        raise ValueError("dataset_type must be 'train', 'dev', or 'test'")


def create_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = True
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloader
