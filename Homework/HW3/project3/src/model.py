import dataProcess
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MAX_EPOCH = 35
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
THRESHOLD = 0.65

seed = 42069
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MyModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11),  # 11个类别
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        # output: [batch_size, 11]
        return x

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_loss = []
    dev_loss = []
    epoch_loss = []
    train_accuracy = 0.0
    dev_accuracy = 0.0
    min_loss = 100
    epoch = 0
    my_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        my_optimizer, step_size=10, gamma=0.5
    )  # 学习率衰减
    while epoch < MAX_EPOCH:
        model.train()
        # 重置上一个epoch的损失和准确率
        train_accuracy = 0.0
        dev_accuracy = 0.0
        epoch_loss.clear()
        for data, label in tqdm(train_data):
            # 将数据和标签移动到设备上
            data, label = data.to(device), label.to(device)
            # 正向传播
            my_optimizer.zero_grad()
            outputs = model(data)
            # 计算损失（绘图）
            loss = model.calculate_loss(outputs, label)
            loss = loss.to(device)
            epoch_loss.append(loss.detach().item())
            # 反向传播
            loss.backward()
            _, predicted_label = torch.max(outputs, 1)
            my_optimizer.step()
            # Classification问题计算预测正确个数
            train_accuracy += (predicted_label == label).sum().item()

        # 学习率衰减
        scheduler.step()
        # 计算平均损失和准确率
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_accuracy /= len(train_data.dataset)
        # 验证集计算loss
        val_loss, dev_accuracy = model_validation(model, dev_data)
        dev_loss.append(val_loss)
        # 留作Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            print(
                f"\n---NOW! in epoch: {epoch + 1}, the lowest loss(validation) is {val_loss}"
            )

        # 每个epoch结束后，打印当前的损失和准确率
        print(
            f"Epoch: {epoch + 1}:\tTrain_loss: {train_loss[epoch]:3.6f}, Train Acc: {train_accuracy:3.6f}. | Dev_loss: {val_loss:3.6f}, Dev Acc: {dev_accuracy:3.6f}"
        )

        epoch += 1

    return train_loss, dev_loss


def model_validation(model: MyModel, dev_data: DataLoader):
    model.eval()
    loss = []
    dev_accuracy = 0.0
    # 计算验证集的损失
    for data, label in dev_data:
        # 将数据和标签移动到设备上
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(data)
            loss.append(model.calculate_loss(output, label).item())
            _, predicted_label = torch.max(output, 1)
            dev_accuracy += (predicted_label == label).sum().item()
    # 计算平均损失
    dev_accuracy /= len(dev_data.dataset)
    return sum(loss) / len(loss), dev_accuracy


def get_pseudo_labels(dataset: Dataset, model: MyModel):
    selected_images = []
    selected_labels = []

    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)  # cross entropy中的softmax是在creterion()时进行的

    for batch in tqdm(data_loader):
        data, _ = batch

        with torch.no_grad():
            logits = model(data.to(device))
        probabilities = softmax(logits)
        confidence, pseudo_labels = torch.max(probabilities, dim=1)
        mask = confidence > THRESHOLD  # 筛选置信度大于阈值的样本

        if mask.any():
            selected_images.append(data[mask])  # shape = [N_i, C, H, W]
            selected_labels.append(pseudo_labels[mask])

    model.train()
    if selected_images:
        # 每一组筛选出的图像的第0维不同（如第一组3张、第二组6张，在第0维将其合并为总的9张）
        selected_images = torch.cat(selected_images, dim=0)
        selected_labels = torch.cat(selected_labels, dim=0)
        return dataProcess.PseudoLabelDataset(
            images=selected_images,
            labels=selected_labels,
        )
    else:
        return None


# def test(model: MyModel, test_data: DataLoader):
#     """
#     标记
#     :param model:
#     :param test_data:
#     :return:
#     """
#     model.eval()
#     loss = []
#     for data, label in test_data:
#         output = model(data)
#         loss.append(model.calculate_loss(output, label))
#     return sum(loss) / len(loss)
