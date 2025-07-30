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
FOCUS_PORTION = 0.65
BATCH_SIZE = model.BATCH_SIZE
NUM_WORKERS = model.NUM_WORKERS
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
print("Loading data...")
data_root = "Homework/resources/HW1/"
train_dataset = dataProcess.create_dataset(data_root + "covid.train.csv", "train")
dev_dataset = dataProcess.create_dataset(data_root + "covid.train.csv", "dev")
test_dataset = dataProcess.create_dataset(data_root + "covid.test.csv", "test")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Dev dataset size: {len(dev_dataset)}")
print(f"Test dataset size: {len(test_dataset)}\n")


# 3.训练集和验证集的划分
train_dataloader = dataProcess.create_dataloader(
    train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True
)
dev_dataloader = dataProcess.create_dataloader(
    dev_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=False
)
test_dataloader = dataProcess.create_dataloader(
    test_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=False
)


# 4.训练开始
print("Start Training:")
my_model = model.MyModel(93)
my_model.to(device)
start_time = time.time()
train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")


# 5.训练结束，结果绘制：
plot_manager = graphMaking.PlottingManager(
    save_dir="Homework/HW1/project1-new/saved_graphs"
)
plot_manager.plot_learning_curve(
    train_loss, dev_loss, "MyModel", focus_portion=FOCUS_PORTION
)
# 测试用：
print(f"Last 5 train_loss values: {train_loss[-5:]}")
print(f"Last 5 dev_loss values: {dev_loss[-5:]}\n")

# 6.测试集结果
print("Start Testing:")
# my_model.load_state_dict(torch.load(model.best_model_path, map_location=device))
# my_model.to(device)
model.save_predictions(
    my_model, test_dataloader, "Homework/HW1/submission/submission.csv"
)
print("Testing completed. Predictions saved to 'submission.csv'")
