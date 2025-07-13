from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
import torch
import model


unlabeled_transfrom = transforms.Compose(
    # 对伪标签数据的特殊处理
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 不知道为什么不能使用
    ]
)
stable_transfrom = transforms.Compose(
    # 对验证数据的特殊处理
    [
        transforms.Resize([128, 128]),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
    ]
)


def create_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle=True, drop_last=False
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader


def create_dataset(data_root: str):
    # 以下为部分数据预处理：
    train_transfrom = transforms.Compose(  # online augmentation
        [
            transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 不知道为什么不能使用/imagenet的均值和方差
        ]
    )

    test_transfrom = transforms.Compose(
        [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ]
    )

    def target_tf(y):
        return torch.tensor(y, dtype=torch.int64)

    train_dataset = DatasetFolder(
        root=data_root + "training/labeled",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=train_transfrom,
        target_transform=target_tf,
    )
    if model.add_valid_data_into_training:
        # 确认完其余要素之后使用所有数据训练现有模型
        valid_dataset = DatasetFolder(
            root=data_root + "validation",
            loader=lambda x: Image.open(x),
            extensions=(".jpg",),
            transform=train_transfrom,
            target_transform=target_tf,
        )
    else:
        # 训练时不使用transforms
        valid_dataset = DatasetFolder(
            root=data_root + "validation",
            loader=lambda x: Image.open(x),
            extensions=(".jpg",),
            transform=test_transfrom,
            target_transform=target_tf,
        )
        another_valid_dataset = DatasetFolder(
            root=data_root + "validation",
            loader=lambda x: Image.open(x),
            extensions=(".jpg",),
            transform=stable_transfrom,  # 验证集的另一种数据增强方式
            target_transform=target_tf,
        )
        ConcatValidDataset = torch.utils.data.ConcatDataset(
            [valid_dataset, another_valid_dataset]
        )
        valid_dataset = ConcatValidDataset  # 将两种验证集数据增强方式合并

    unlabeled_dataset = DatasetFolder(
        root=data_root + "training/unlabeled",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=test_transfrom,  # 检验数据可靠性的时候不使用数据增强
        target_transform=target_tf,
    )
    test_dataset = DatasetFolder(
        root=data_root + "testing",
        loader=lambda x: Image.open(x),
        extensions=(".jpg",),
        transform=test_transfrom,
        target_transform=target_tf,
    )
    print("Finishing creating datasets!\n")

    return train_dataset, valid_dataset, unlabeled_dataset, test_dataset


class PseudoLabelDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # 为什么这里不放transforms呢？
        # --因为最早unlabeledDataset已经进行过一次数据增强了
        self.images = images
        self.labels = labels
        self.transform = transform
        # 但是发现先加transforms会让train_acc大幅提高而valid_acc基本不变

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(self.images[idx])
        label = self.labels[idx]
        return image, label
