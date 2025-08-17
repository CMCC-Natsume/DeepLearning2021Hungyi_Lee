import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

MAX_EPOCH = 20
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 0.0005


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(429, 512),
            nn.BatchNorm1d(512),  # 批次归一化层
            nn.ReLU(),
            # nn.Dropout(0.15),  # Dropout层，防止过拟合
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 39),  # 39个音素类别
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input_data):
        return self.network(input_data)

    def calculate_loss(self, prediction, label):
        """
        标记
        :return:
        """
        return self.criterion(prediction, label)


def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_loss = []
    dev_loss = []
    epoch_loss = []
    train_accuracy = 0.0
    dev_accuracy = 0.0
    min_loss = 100
    epoch = 0
    my_optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        my_optimizer, step_size=11, gamma=0.5
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
                f"--NOW!! In epoch: {epoch}, the lowest loss(valid): {val_loss:3.6f} , accuracy:{dev_accuracy:3.6f}"
            )

        # 每个epoch结束后，打印当前的损失和准确率
        print(
            f"Epoch: {epoch + 1}:\tTrain_loss: {train_loss[epoch]:3.6f}, Train Acc: {train_accuracy:3.6f}. | Dev_loss: {val_loss:3.6f}, Dev Acc: {dev_accuracy:3.6f}"
        )

        epoch += 1
        if epoch != MAX_EPOCH:
            print(f"\n\nEpoch :\t{epoch + 1}")

    return train_loss, dev_loss


def model_validation(model: MyModel, dev_data: DataLoader):
    model.eval()  # 标记
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


def test(model: MyModel, test_data: DataLoader):
    model.eval()
    predictions = []
    print("Testing the model...")

    with torch.no_grad():
        for data in tqdm(test_data):
            data = data.to(device)
            output = model(data)
            _, predicted_label = torch.max(output, 1)
            predictions.extend(predicted_label.cpu().numpy().tolist())

    return predictions


def save_predictions_to_csv(predictions: list, filepath: str = "predict.csv"):
    """
    Save the predicted results to a CSV file at a specified path.
    :param predictions: A list of predicted class IDs.
    :param filepath: Full path (including filename) to save the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    print(f"Saving predictions to {filepath}...")
    with open(filepath, "w") as f:
        f.write("Id,Class\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    print(f"Predictions saved to {filepath}.")
