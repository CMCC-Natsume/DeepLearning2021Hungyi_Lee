import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


MAX_EPOCH = 4000
LEARN_RATE = 0.002


class MyModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=50),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(50, 10),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(10, 1),
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, input_data):
        return self.network(input_data).squeeze(1)

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_losses_per_epoch = []  # Changed variable name for clarity
    dev_loss = []
    min_loss = 100
    epoch = 1
    my_optimizer = optim.AdamW(model.parameters(), lr=LEARN_RATE, weight_decay=1e-5)
    model.to(device)
    while epoch < MAX_EPOCH:
        model.train()
        epoch_train_losses = []
        for data, label in train_data:
            data, label = data.to(device), label.to(device)
            my_optimizer.zero_grad()
            output_data = model(data)
            loss = model.calculate_loss(output_data, label)
            loss = loss.to(device)
            epoch_train_losses.append(
                loss.detach().cpu()
            )  # Append batch loss to temporary list
            loss.backward()
            my_optimizer.step()

        # Calculate the average training loss for the current epoch
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses_per_epoch.append(
            avg_train_loss
        )  # Append the average to the main list

        valid_loss = dev(model, dev_data)
        if valid_loss < min_loss:
            min_loss = valid_loss
            print(
                f"\n--NOW!! In epoch: {epoch}, the lowest loss(valid) is {valid_loss}"
            )
        dev_loss.append(valid_loss.detach().cpu())

        epoch += 1

    return train_losses_per_epoch, dev_loss


def dev(model: MyModel, dev_data: DataLoader):
    model.eval()  # 标记
    loss = []
    for data, label in dev_data:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss.append(model.calculate_loss(output, label))
    return sum(loss) / len(loss)


def save_predictions(model, test_dataloader, output_path="prediction.csv"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    # 保存为Kaggle要求的CSV格式
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "tested_positive"])
        for i, pred in enumerate(predictions):
            writer.writerow([i, pred.item()])
