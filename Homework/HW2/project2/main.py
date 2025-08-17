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


# 固定随机种子，确保结果可复现
seed = 42069
BATCH_SIZE = 32
NUM_WORKERS = 0
VAL_RATIO = 0.1
same_seeds(seed)


# 01 检查是否有GPU可用
# （应用至损失函数、网络模型、数据上）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU :\t{torch.cuda.get_device_name(0)}\n")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用，使用CPU进行训练")


# 02 填入资源文件路径
print("Loading data...")
data_root = "Homework/resources/HW2/timit_11/timit_11/"  # 此处为项目根目录（即DL2021而非project2）
train = np.load(data_root + "train_11.npy", mmap_mode="r")
train_label = np.load(data_root + "train_label_11.npy", mmap_mode="r")
print("Size of training data: {}".format(train.shape))

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = (
    train[:percent],
    train_label[:percent],
    train[percent:],
    train_label[percent:],
)
train_dataset = dataProcess.MyDataset(train_x, train_y)
dev_dataset = dataProcess.MyDataset(val_x, val_y)
print("Finishing creating datasets!\n")


# 03 训练集和验证集的划分
print("Start creating dataloader")
train_dataloader = dataProcess.create_dataloader(
    train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True
)
dev_dataloader = dataProcess.create_dataloader(dev_dataset, BATCH_SIZE, NUM_WORKERS)
print("Finishing creating dataLoaders!\n")


# 04 训练开始
print("Start Training:")
my_model = model.MyModel()
my_model.to(device)
start_time = time.time()  # 计时器
train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")


# 5.训练结束，结果绘制：
graphMaking.plot_learning_curve(train_loss, dev_loss, "ModelOfTIMIT11")
# test在验证集上获取预测和真实标签
print("Generating confusion matrix...")
my_model.eval()
true_labels = []
pred_labels = []
with torch.no_grad():
    for data, label in dev_dataloader:
        data, label = data.to(device), label.to(device)
        output = my_model(data)
        _, predicted = torch.max(output, 1)
        true_labels.extend(label.cpu().numpy().tolist())
        pred_labels.extend(predicted.cpu().numpy().tolist())
# 绘制并保存混淆矩阵
graphMaking.plot_confusion_matrix(
    true_labels,
    pred_labels,
    save_path="Homework/HW2/project2/savedGraph/confusion_matrix.png",
)
print("Confusion matrix saved!")


# 06 测试集结果
print("Start Testing:")
test = np.load(data_root + "test_11.npy", mmap_mode="r")
print("Size of testing data: {}".format(test.shape))
test_dataset = dataProcess.MyDataset(test)
test_dataloader = dataProcess.create_dataloader(test_dataset, BATCH_SIZE, NUM_WORKERS)
predictions = model.test(my_model, test_dataloader)
model.save_predictions_to_csv(
    predictions, filepath="Homework/HW2/submission/submission.csv"
)
