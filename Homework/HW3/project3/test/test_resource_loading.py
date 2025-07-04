import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms as transforms

# 确认资源文件夹存在并列出内容
print()
print(os.listdir("Homework/resources/HW3/food-11/"))


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


# 查看数据集的存储内容
print(f"train_dataset type:\t{type(train_dataset)}\n")
count = 0
for sample in train_dataset.samples:
    if count < 5:
        print(f"sample: {sample}")
        count += 1
print()
count = 0
for sample in unlabeled_dataset.samples:
    if count < 5:
        print(f"sample: {sample}")
        count += 1
print()


# 查看dataloader中的内容
T_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
V_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)
U_dataloader = DataLoader(unlabeled_dataset, batch_size=2, shuffle=True, num_workers=0)
print(f"train_dataset length:\t{len(train_dataset)}")
print(f"t_dataloader length:\t{len(T_dataloader)}\n")
print(f"valid_dataset length:\t{len(valid_dataset)}")
print(f"v_dataloader length:\t{len(V_dataloader)}\n")
print(f"unlabeled_dataset length:\t{len(unlabeled_dataset)}")
print(f"u_dataloader length:\t{len(U_dataloader)}\n")
# 查看数据集的标签
# for data, label in T_dataloader:
#     print(f"\ndata: \n{data},\n\nlabel:\n{label}")
#     break
