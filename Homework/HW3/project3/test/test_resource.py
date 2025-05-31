from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms as transforms

"""
本项目的数据集构成:
"""

# 检查数据集

# 查看数据加载情况
print("\nLoading data...")
data_root = (
    "Homework/resources/HW3/food-11/"  # 此处为项目根目录（即DL2021而非project2）
)
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
# 查看数据集的形状
print("train_dataset shape: ", train_dataset.samples[0][0])
print("test_dataset shape: ", test_dataset.samples[0][0])
# 查看dataloader中的内容
T_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# 查看数据集的标签
for data, label in T_dataloader:
    print(f"\ndata: \n{data},\nlabel:\n{label}")
    break
