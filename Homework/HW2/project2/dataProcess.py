import numpy
import torch
from torch.utils.data import DataLoader, Dataset


"""
数据集的划分
1. 训练集: train_11.npy
2. 测试集: test_11.npy
3. 验证集: train_11.npy
4. 训练集和验证集的划分: 训练集的前90%作为训练集，后10%作为验证集
"""


class MyDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X.copy()).float()
        if y is not None:
            y = y.astype(numpy.int32)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def create_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle=False
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # 加速数据加载
        drop_last=False,
    )
    return dataloader


def create_dev_DataLoader(dataset: Dataset, batch_size: int, num_workers: int):
    """
    创建验证集的dataloader
    :param dataset: 验证集
    :param batch_size: 批大小
    :param num_workers: 工作线程数
    :return: 验证集的dataloader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


"""
本项目的数据集构成:
"""
# 检查数据集
if __name__ == "__main__":
    # 查看数据加载情况
    print("Loading data...")
    data_root = "Homework/resources/HW2/timit_11/timit_11/"  # 此处为项目根目录（即DL2021而非project2）
    train_dataset = numpy.load(data_root + "train_11.npy")
    test_dataset = numpy.load(data_root + "test_11.npy")
    train_label_dataset = numpy.load(data_root + "train_label_11.npy")
    # 查看数据集的形状
    print("train_dataset shape: ", train_dataset.shape)
    print("test_dataset shape: ", test_dataset.shape)
    print("train_label_dataset shape: ", train_label_dataset.shape)
    # 查看dataloader中的内容
    T_dataset = MyDataset(
        data_root + "train_11.npy", "train", data_root + "train_label_11.npy"
    )
    T_dataloader = create_dataloader(T_dataset, 8, 0)
    for data, label in T_dataloader:
        print(f"data: \n{data},\n label: \n{label}")
        break
