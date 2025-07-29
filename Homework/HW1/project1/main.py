import time

import dataProcess
import graphMaking
import model
import numpy as np
import torch


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 0.固定随机种子，确保结果可复现
seed = 42069
same_seeds(seed)


# 1.检查是否有GPU可用（应用至损失函数、网络模型、数据上）
print("Checking device...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU :\t{torch.cuda.get_device_name(0)}\n")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用 使用CPU进行训练")


# 2.填入资源文件路径
print("Loading data...\n")
data_root = "Homework/resources/HW1/"
train_dataset = dataProcess.MyDataset(data_root + "covid.train.csv", "train")
test_dataset = dataProcess.MyDataset(data_root + "covid.test.csv", "test")
dev_dataset = dataProcess.MyDataset(data_root + "covid.train.csv", "dev")


# 3.训练集和验证集的划分
train_dataloader = dataProcess.make_dataloader(train_dataset, 32, 0)
test_dataloader = dataProcess.make_dataloader(test_dataset, 32, 0, shuffle=False)
dev_dataloader = dataProcess.make_dataloader(train_dataset, 32, 0)


# 4.训练开始
print("Start Training:")
print(f"Train dataset dim: {train_dataset.dim}")
my_model = model.MyModel(train_dataset.dim)
my_model.to(device)
start_time = time.time()
train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")


# 5.训练结束，结果绘制：
graphMaking.plot_learning_curve(train_loss, dev_loss, "MyModel")
print(f"Length of train_loss: {len(train_loss)}")
print(f"Length of dev_loss: {len(dev_loss)}")
# 打印前5个训练损失值和前5个验证损失值，观察数值大小和数量级
print(f"Last 5 train_loss values: {train_loss[-5:]}")
print(f"Last 5 dev_loss values: {dev_loss[-5:]}")

# 6.测试集结果
print("Start Testing:")
model.save_predictions(
    my_model, test_dataloader, "Homework/HW1/submission/submission.csv"
)
print("Testing completed. Predictions saved to 'submission.csv'")
