from torch import nn, optim
from torch.utils.data import DataLoader
from termcolor import colored
import torch
import torch.nn.functional as F
import numpy as np
import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


MAX_EPOCH = 5000
PATIENCE = 200
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
NUM_WORKERS = 0
best_model_path = "Homework/HW1/project1-new/saved_models/best_model.pth"


# 试用模型1（效果较差）
class SelfAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        基于自注意力机制的表格数据回归模型。

        Args:
            input_dim (int): 输入特征维度（例如93）。
            num_heads (int): 多头注意力的头数。
            hidden_dim (int): 隐藏层维度。
            dropout (float): Dropout比例，用于正则化。
        """
        super().__init__()

        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 多头自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.criterion = nn.MSELoss(reduction="mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        """
        前向传播。

        Args:
            input_data (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)

        Returns:
            torch.Tensor: 回归输出，形状为 (batch_size,)
        """
        # 投影到隐藏维度
        x = self.input_projection(input_data)  # (batch_size, hidden_dim)

        # 添加一维以适配多头注意力（需要形状为 (seq_len, batch_size, hidden_dim)）
        x = x.unsqueeze(0)  # (1, batch_size, hidden_dim)

        # 自注意力层
        attn_output, _ = self.self_attention(x, x, x)  # (1, batch_size, hidden_dim)
        x = self.norm1(x + attn_output)  # 残差连接 + LayerNorm
        x = x.squeeze(0)  # (batch_size, hidden_dim)

        # 前馈网络
        ffn_output = self.feed_forward(x)  # (batch_size, hidden_dim)
        x = self.norm2(x + ffn_output)  # 残差连接 + LayerNorm

        # 输出层
        output = self.output_layer(x).squeeze(1)  # (batch_size,)
        return output

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


# 试用模型2（效果较差）
class MyResNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.Linear1 = nn.Linear(in_features=input_dim, out_features=64)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=64, out_features=64)
        self.Linear3 = nn.Linear(in_features=64, out_features=1)
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, input_data):
        x = self.Linear1(input_data)
        x = self.ReLU(x)
        residual = x
        x = self.Linear2(x)
        x = self.ReLU(x)
        return self.Linear3(x + residual).squeeze(1)

    def calculate_loss(self, prediction, label):
        return torch.sqrt(self.criterion(prediction, label))


class MyModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 1),
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, input_data):
        return self.network(input_data).squeeze(1)

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_losses_per_epoch = []
    dev_loss = []
    min_loss = 10000
    epoch = 1
    stable_count = 0
    my_optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
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
        dev_loss.append(valid_loss.detach().cpu())
        if valid_loss < min_loss:
            stable_count = 0
            min_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                colored(
                    f"\n--NOW!! In epoch: {epoch}, the lowest loss(valid) is {valid_loss}",
                    "green",
                )
            )
        else:
            stable_count += 1
            print(
                f"\n--In epoch: {epoch}, the Valid-Loss is {valid_loss}. Train-Loss is {avg_train_loss}"
            )
        if stable_count >= PATIENCE:
            print(
                colored(
                    f"\n--Early stopping at epoch {epoch}, no improvement for {PATIENCE} epochs.",
                    "red",
                )
            )
            print(f"Best validation loss: {min_loss}\tTotal epochs: {epoch}")
            return train_losses_per_epoch, dev_loss

        epoch += 1

    return train_losses_per_epoch, dev_loss


def dev(model: MyModel, dev_data: DataLoader):
    model.eval()
    loss = []
    for data, label in dev_data:
        data, label = data.to(device), label.to(device)
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
