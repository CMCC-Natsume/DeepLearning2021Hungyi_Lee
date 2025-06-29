from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms as transforms

VALIDATION_RATIO = 0.1  # 验证集比例


"""
数据集的划分
1. 训练集: 
2. 测试集: 
3. 验证集: 
4. 训练集和验证集的划分: 训练集的前90%作为训练集，后10%作为验证集
"""


def create_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle=False
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloader


def create_dataset(data_root: str):
    # 以下为部分数据预处理：
    train_transfrom = transforms.Compose(
        [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ]
    )
    test_transfrom = transforms.Compose(
        [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ]
    )

    train_dataset = DatasetFolder(
        root=data_root + "training/labeled",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=train_transfrom,
    )
    valid_dataset = DatasetFolder(
        root=data_root + "validation",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=test_transfrom,
    )
    unlabeled_dataset = DatasetFolder(
        root=data_root + "training/unlabeled",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=train_transfrom,
    )
    test_dataset = DatasetFolder(
        root=data_root + "testing",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=test_transfrom,
    )
    print("Finishing creating datasets!\n")

    return train_dataset, valid_dataset, unlabeled_dataset, test_dataset


# def create_dev_DataLoader(dataset: Dataset, batch_size: int, num_workers: int):
#     """
#     创建验证集的dataloader
#     :param dataset: 验证集
#     :param batch_size: 批大小
#     :param num_workers: 工作线程数
#     :return: 验证集的dataloader
#     """
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         drop_last=False,
#     )
