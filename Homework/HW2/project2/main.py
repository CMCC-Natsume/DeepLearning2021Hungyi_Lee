import torch
import dataProcess
import model
import graphMaking

# 使用自动求导引擎
torch.backends.cudnn.benchmark = True

seed = 42069
BATCH_SIZE = 32
NUM_WORKERS = 0
torch.manual_seed(seed)

# 检查是否有GPU可用
# （应用至损失函数、网络模型、数据上）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU :\t{torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用，使用CPU进行训练")


# 填入资源文件路径
print("Loading data...")
data_root = 'Homework/resources/HW2/timit_11/timit_11/' # 此处为项目根目录（即DL2021而非project2）
train_dataset = dataProcess.MyDataset(data_root + "train_11.npy", 'train', data_root + "train_label_11.npy")
test_dataset = dataProcess.MyDataset(data_root + "test_11.npy", 'test', None)
dev_dataset = dataProcess.MyDataset(data_root + "train_11.npy", 'dev', data_root + "train_label_11.npy")
print(f"Finishing creating datasets!")


# 训练集和验证集的划分
print(f"Start creating dataloader")
train_dataloader = dataProcess.create_dataloader(train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True)
test_dataloader = dataProcess.create_dataloader(test_dataset, BATCH_SIZE, NUM_WORKERS)
dev_dataloader = dataProcess.create_dataloader(dev_dataset, BATCH_SIZE, NUM_WORKERS)
print(f"Finishing creating dataLoaders!")
print(f"\tTraining Dataset.dim = {train_dataset.dim}\n")


# 训练开始
print(f"Start Training:")
my_model = model.MyModel(train_dataset.dim)
my_model.to(device)

train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
graphMaking.plot_learning_curve(train_loss, dev_loss, "ModelOfTIMIT11")


