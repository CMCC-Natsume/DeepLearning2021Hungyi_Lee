import torch
import dataProcess
import model
import graphMaking

seed = 42069
torch.manual_seed(seed)

# 检查是否有GPU可用
# （应用至损失函数、网络模型、数据上）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用，使用CPU进行训练")


# 填入资源文件路径
print("Loading data...")
data_root = 'Homework/resources/Hw2/timit_11/timit_11/' # 此处为项目根目录（即DL2021而非project2）
train_dataset = dataProcess.MyDataset(data_root + "train_11.npy", 'train')
test_dataset = dataProcess.MyDataset(data_root + "test_11.npy", 'test')
dev_dataset = dataProcess.MyDataset(data_root + "train_11.npy", 'dev')


# 训练集和验证集的划分
# train_dataloader = dataProcess.create_dataloader(train_dataset, 32, 0)
# test_dataloader = dataProcess.create_dataloader(test_dataset, 32, 0)
# dev_dataloader = dataProcess.create_dataloader(train_dataset, 32, 0)


# # 训练开始
# my_model = model.MyModel(train_dataset.dim)
# my_model.to(device)

# print(train_dataset.dim)
# train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
# graphMaking.plot_learning_curve(train_loss, dev_loss, "MyModel")










