import torch
import dataProcess
import model
import graphMaking

seed = 42069
torch.manual_seed(seed)

print(f"{torch.cuda.is_available()}\n\n")

# 检查是否有GPU可用
# （应用至损失函数、网络模型、数据上）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("现在没有GPU可用，使用CPU进行训练")


# 填入资源文件路径
train_dataset = dataProcess.MyDataset("resources/covid.train.csv", 'train')
test_dataset = dataProcess.MyDataset("resources/covid.test.csv", 'test')
dev_dataset = dataProcess.MyDataset("resources/covid.train.csv", 'dev')


# 训练集和验证集的划分
train_dataloader = dataProcess.make_dataloader(train_dataset, 32, 0)
test_dataloader = dataProcess.make_dataloader(test_dataset, 32, 0)
dev_dataloader = dataProcess.make_dataloader(train_dataset, 32, 0)


my_model = model.MyModel(train_dataset.dim)
my_model.to(device)

print(train_dataset.dim)
train_loss, dev_loss = model.model_training(train_dataloader, dev_dataloader, my_model)
graphMaking.plot_learning_curve(train_loss, dev_loss, "MyModel")










