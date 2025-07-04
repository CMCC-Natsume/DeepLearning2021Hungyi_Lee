import dataProcess
import graphMaking
import model
import numpy as np
import torch
from torchvision.transforms import transforms as transforms


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 42069
BATCH_SIZE = model.BATCH_SIZE
NUM_WORKERS = 0
same_seeds(seed)  # 固定随机种子，确保结果可复现

# 检查是否有GPU可用
# （应用至损失函数、网络模型、数据上）
print("Checking device...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU :\t{torch.cuda.get_device_name(0)}\n")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用，使用CPU进行训练")


# 填入资源文件路径
print("Loading data...")
data_root = "Homework/resources/HW3/food-11/"  # 此处为项目根目录（即DL2021）
(train_dataset, valid_dataset, unlabeled_dataset, test_dataset) = (
    dataProcess.create_dataset(data_root=data_root)
)


# 训练集和验证集的划分
print("Start creating dataloader")
# train_dataloader = dataProcess.create_dataloader(
#     train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True
# )
test_dataloader = dataProcess.create_dataloader(test_dataset, BATCH_SIZE, NUM_WORKERS)
valid_dataloader = dataProcess.create_dataloader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
print("Finishing creating dataLoaders!\n")


# 训练开始
print("Start Training:")
my_model = model.MyModel()
my_model.to(device)

train_loss, dev_loss = model.model_training(
    train_dataset, unlabeled_dataset, valid_dataloader, my_model
)
graphMaking.plot_learning_curve(train_loss, dev_loss, "ModelOfFood11")
