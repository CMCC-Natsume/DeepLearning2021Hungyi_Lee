from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms as transforms


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
    train_transfrom = transforms.Compose(  # online augmentation
        [
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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


class PseudoLabelDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        if transform is None:
            self.transform = transforms.transforms.Compose(  # online augmentation
                [
                    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
