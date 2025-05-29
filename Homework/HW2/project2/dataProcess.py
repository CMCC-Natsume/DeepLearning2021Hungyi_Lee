from torch.utils.data import Dataset, DataLoader
import numpy
import torch
# import csv

ROUND = 0
VALIDATION_RATIO = 0.1  # 验证集比例


"""
数据集的划分
1. 训练集: train_11.npy
2. 测试集: test_11.npy
3. 验证集: train_11.npy
4. 训练集和验证集的划分: 训练集的前90%作为训练集，后10%作为验证集
"""
class MyDataset(Dataset):
    def __init__(self, path: str, mode: str, inputLabel: str = ''):
        super().__init__()
        self.mode = mode
        # 判断是否为测试集:
        if mode == 'test':
            self.data = torch.from_numpy(numpy.load(path, mmap_mode='r').copy()).float()
            self.targets = None
        else:
            # 非test数据集
            target = numpy.load(inputLabel, mmap_mode='r')  # 本两行根据数据特点进行切片
            data = numpy.load(path, mmap_mode='r')
            if target.dtype.kind in {'U', 'S'}:
                print("\tWarning: target is string, converting to int64")
                target = target.astype(numpy.int64)
            # 不使用 mmap_mode则会将整个数据集加载到内存中，大概率会导致内存不足
            
            # 划分训练集和验证集:
            num_of_data = data.shape[0]
            split_index = int(num_of_data * (1 - VALIDATION_RATIO))  # 划分点
            train_index = []
            dev_index = []
            train_index = list(range(0, split_index))
            dev_index = list(range(split_index, num_of_data))
            
            # train数据集(放入属性前最后处理):
            if mode == 'train':
                self.data = torch.from_numpy(data[train_index]).float()
                self.targets = torch.tensor(target[train_index], dtype=torch.long)
            # dev数据集:
            elif mode == 'dev':
                self.data = torch.from_numpy(data[dev_index]).float()
                self.targets = torch.tensor(target[dev_index], dtype=torch.long)
            else:
                print("Error: mode is not train or dev")
                raise ValueError("mode is not train or dev")
        if mode == 'train' or mode == 'dev':
            print("标签范围:", target[train_index].min(), target[train_index].max())
        self.dim = self.data.shape[1]


    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'dev':
            return self.data[item], self.targets[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle=False):
    dataloader = \
    DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dataloader



def create_dev_DataLoader(dataset: Dataset, batch_size: int, num_workers: int):
    """
    创建验证集的dataloader
    :param dataset: 验证集
    :param batch_size: 批大小
    :param num_workers: 工作线程数
    :return: 验证集的dataloader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


# def csv_fileReader(path: str) -> numpy.ndarray:
#     with open(path) as file:
#         csv_list = list(csv.reader(file))
#         data = numpy.array(csv_list)
#         data = data[1:, 1:]
#         data = data.astype(float)
#         return data


"""
本项目的数据集构成:
"""
# 检查数据集
if __name__ == "__main__":
    # 查看数据加载情况
    print("Loading data...")
    data_root = 'Homework/resources/HW2/timit_11/timit_11/' # 此处为项目根目录（即DL2021而非project2）
    train_dataset = numpy.load(data_root + "train_11.npy")
    test_dataset = numpy.load(data_root + "test_11.npy")
    train_label_dataset = numpy.load(data_root + "train_label_11.npy")
    # 查看数据集的形状
    print("train_dataset shape: ", train_dataset.shape)
    print("test_dataset shape: ", test_dataset.shape)
    print("train_label_dataset shape: ", train_label_dataset.shape)
    # 查看dataloader中的内容
    T_dataset = MyDataset(data_root + "train_11.npy", 'train', data_root + "train_label_11.npy")
    T_dataloader = create_dataloader(T_dataset, 8, 0)
    for data, label in T_dataloader:
        print(f"data: \n{data},\n label: \n{label}")
        break
