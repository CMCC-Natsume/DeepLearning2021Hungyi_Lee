from torch.utils.data import Dataset, DataLoader
import numpy
import torch
# import csv

ROUND = 0

"""
本项目的数据集构成:
"""
# 检查数据集
if __name__ == "__main__":
    print("Loading data...")
    data_root = 'Homework/resources/HW2/timit_11/timit_11/' # 此处为项目根目录（即DL2021而非project2）
    train_dataset = numpy.load(data_root + "train_11.npy")
    test_dataset = numpy.load(data_root + "test_11.npy")
    train_label_dataset = numpy.load(data_root + "train_label_11.npy")
    print("train_dataset shape: ", train_dataset.shape)
    print("test_dataset shape: ", test_dataset.shape)
    print("train_label_dataset shape: ", train_label_dataset.shape)




"""
数据集的划分
1. 训练集: train_11.npy
2. 测试集: test_11.npy
3. 验证集: train_11.npy
4. 训练集和验证集的划分: 训练集的前90%作为训练集，后10%作为验证集
"""
class MyDataset(Dataset):
    def __init__(self, path: str, mode: str, inputLabel: str = None):
        super().__init__()
        self.mode = mode
        # 判断是否为测试集:
        if mode == 'test':
            self.data = numpy.load(path)
            self.targets = None
        else:
            # 非test数据集
            target = numpy.load(inputLabel, mmap_mode='r')  # 本两行根据数据特点进行切片
            data = numpy.load(path, mmap_mode='r')
            # target = numpy.load(inputLabel)  # 不使用 mmap_mode则会将整个数据集加载到内存中，大概率会导致内存不足
            # data = numpy.load(path)

            # 划分训练集和验证集:
            train_index = []
            dev_index = []
            num_of_data = data.shape[0]
            for i in range(num_of_data):
                if i % 10 == ROUND:
                    train_index.append(i)
                else:
                    dev_index.append(i)

            # train数据集(放入属性前最后处理):
            if mode == 'train':
                self.targets = target[train_index]
                self.data = data[train_index]
            # dev数据集:
            elif mode == 'dev':
                self.targets = target[dev_index]
                self.data = data[dev_index]
            else:
                print("Error: mode is not train or dev")
                raise ValueError("mode is not train or dev")

        self.dim = self.data.shape[1]


    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'dev':
            return self.data[item], self.targets[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    dataloader = \
    DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dataloader


# def csv_fileReader(path: str) -> numpy.ndarray:
#     with open(path) as file:
#         csv_list = list(csv.reader(file))
#         data = numpy.array(csv_list)
#         data = data[1:, 1:]
#         data = data.astype(float)
#         return data
