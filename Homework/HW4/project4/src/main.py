import time
import data_process
import graph_making
import model
import model_train
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
BATCH_SIZE = model_train.BATCH_SIZE
NUM_WORKERS = model_train.NUM_WORKERS
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
data_root = "Homework/resources/HW4/Dataset/"  # 此处为项目根目录（即DL2021）


# 3.训练集和验证集的划分
print("Starting creating dataloader")
train_dataloader, valid_dataloader, speaker_num = data_process.create_dataloader(
    data_root, BATCH_SIZE, NUM_WORKERS, shuffle=True
)
# test_dataloader = data_process.create_dataloader(
#     data_root, BATCH_SIZE, NUM_WORKERS, shuffle=False
# )
print("Finishing creating dataLoaders!\n")


# 4.训练开始
print("Start Training:")
my_model = model.MyModel()
my_model.to(device)
start_time = time.time()  # 计时器
train_loss, dev_loss, train_acc, dev_acc = model_train.model_training(
    train_dataloader, valid_dataloader, my_model
)
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")


# 5.训练结束，结果绘制：
graph_making.plot_loss_curve(train_loss, dev_loss, "ModelOfVoxceleb")
graph_making.plot_accuracy_curve(train_acc, dev_acc, "ModelOfVoxceleb")


# 6.测试集结果
# print("Start Testing:")
# predictions = model_train.test(model=my_model, test_data=test_dataloader)
# model_train.save_predictions_to_csv(
#     predictions=predictions, filepath="Homework/HW4/submission/submission.csv"
# )
