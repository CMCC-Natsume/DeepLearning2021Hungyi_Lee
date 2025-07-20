import os
import dataProcess
import torch
from termcolor import colored
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 训练参数表：
MAX_EPOCH = 240
BATCH_SIZE = 32
SEMI_EPOCH = 109
LEARNING_RATE = 0.0006
SCHEDULER_START = 150
SCHEDULER_STEP = 50
NUM_WORKERS = 8
WEIGHT_DECAY = 2e-4
THRESHOLD = 0.92
PSEUDO_INTERVAL = 20
do_semi_supervised = True
add_valid_data_into_training = False  # 是否将验证集数据加入训练集(Last Run)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MyModel(nn.Module):
    """
    自定义模型
    该模型为一个简单的卷积神经网络
    包含3个卷积层和3个全连接层
    输入为128x128的RGB图像
    输出为11个类别的概率分布
    """

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            # ---*---
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            # ---*---
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # ---*---
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # ---*---
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # ---*---
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 11),
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


def model_training(
    train_dataset: Dataset,
    unlabeled_dataset: Dataset,
    dev_data: DataLoader,
    model: MyModel,
):
    """
    模型训练函数
    :param train_dataset: 训练集
    :param dev_data: 验证集
    :param model: 模型
    :return: 训练损失和验证损失
    训练集和验证集的损失和准确率
    """
    train_loss = []
    dev_loss = []
    epoch_loss = []
    train_accuracy = 0.0
    dev_accuracy = 0.0
    max_dev_accuracy = 0
    epoch = 0
    best_epoch = 0
    my_optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        my_optimizer, step_size=SCHEDULER_STEP, gamma=0.5
    )
    # 原始训练数据（不含伪标签）：
    train_data = dataProcess.create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    # 定义保存模型的路径
    save_dir = "Homework/HW3/project3/saved_models"
    os.makedirs(save_dir, exist_ok=True)  # 创建目录如果不存在
    best_model_path = os.path.join(save_dir, "best_model.pth")

    while epoch < MAX_EPOCH:
        # 半监督学习部分
        if do_semi_supervised and epoch > SEMI_EPOCH and epoch % PSEUDO_INTERVAL == 0:
            print("Generating pseudo-labels...")
            pseudo_label_dataset = get_pseudo_labels_dataset(
                dataset=unlabeled_dataset, model=model
            )
            # 生成伪标签dataloader
            train_data = generate_pseudo_labeled_data(
                pseudo_label_dataset=pseudo_label_dataset, train_dataset=train_dataset
            )

        if add_valid_data_into_training:
            train_data = add_valid_data(train_dataset, dev_data.dataset)

        model.train()
        # 重置上一个epoch的损失和准确率
        train_accuracy = 0.0
        dev_accuracy = 0.0
        epoch_loss.clear()
        print("Training model...")
        for data, label in tqdm(train_data, ncols=120):
            data, label = data.to(device), label.to(device)
            my_optimizer.zero_grad()
            outputs = model(data)
            loss = model.calculate_loss(outputs, label)
            loss = loss.to(device)
            epoch_loss.append(loss.detach().item())
            loss.backward()
            _, predicted_label = torch.max(outputs, 1)
            my_optimizer.step()
            # 预测正确个数
            train_accuracy += (predicted_label == label).sum().item()

        if epoch > SCHEDULER_START:
            scheduler.step()
        # 计算平均损失和准确率
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_accuracy /= len(train_data.dataset)
        # 验证集计算loss
        print("Validating model...")
        val_loss, dev_accuracy = model_validation(model, dev_data)
        dev_loss.append(val_loss)
        # 留作Early stopping
        if dev_accuracy > max_dev_accuracy:
            max_dev_accuracy = dev_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(
                colored(
                    f"--NOW!! In epoch: {epoch}, the lowest loss(valid): loss{val_loss:3.6f} , accuracy:{dev_accuracy:3.6f}",
                    "yellow",
                )
            )

        print(
            f"Train_loss: {train_loss[epoch]:3.6f}, Train Acc: {train_accuracy:3.6f}. | Dev_loss: {val_loss:3.6f}, Dev Acc: {dev_accuracy:3.6f}"
        )
        epoch += 1
        if epoch != MAX_EPOCH:
            print(f"\n\nEpoch :\t{epoch}")

    # 训练结束
    print(
        f"\n\nTraining finished! Best epoch: {best_epoch + 1}, with highest val_accuracy: {max_dev_accuracy:3.6f}"
    )
    return train_loss, dev_loss


def model_validation(model: MyModel, dev_data: DataLoader):
    model.eval()
    loss = []
    dev_accuracy = 0.0
    # 计算验证集的损失
    for data, label in tqdm(dev_data, ncols=120):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            output = model(data)
            loss.append(model.calculate_loss(output, label).item())
            _, predicted_label = torch.max(output, 1)
            dev_accuracy += (predicted_label == label).sum().item()
    # 计算平均损失
    dev_accuracy /= len(dev_data.dataset)
    return sum(loss) / len(loss), dev_accuracy


def get_pseudo_labels_dataset(dataset: Dataset, model: MyModel):
    """获取伪标签
    :param dataset: 数据集
    :param model: 模型
    :return: PseudoLabelDataset
    """
    all_imgs = []
    all_labels = []

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # 加速数据加载
    )

    model.eval()
    softmax = nn.Softmax(dim=-1)  # cross entropy中的softmax是在creterion()时进行的

    with torch.no_grad():
        for batch in tqdm(data_loader):
            data, _ = batch
            data = data.to(device)
            logits = model(data.to(device))
            probabilities = softmax(logits)
            confidence, pseudo_labels = torch.max(probabilities, dim=1)
            # 筛选置信度大于阈值的样本
            mask = confidence > THRESHOLD
            mask = mask.to(device)

            if mask.any():
                # --这里一定要将数据重新放回cpu:
                # model_training()中使用了num_workers>0的DataLoader
                # 这会导致数据在GPU上，无法直接使用
                # 需要将数据放回CPU
                imgs = data[mask].cpu()  # [n_i, C, H, W]
                labs = pseudo_labels[mask].cpu()  # [n_i]
                # imgs = data[mask]
                # labs = pseudo_labels[mask]
                all_imgs.append(imgs)
                all_labels.append(labs)

    if not all_imgs:
        return None
    imgs = torch.cat(all_imgs, dim=0)
    labs = torch.cat(all_labels, dim=0)
    model.train()

    return dataProcess.PseudoLabelDataset(
        images=imgs,
        labels=labs,
        transform=dataProcess.unlabeled_transfrom,  # 使用未标记数据的增强方式
    )


def generate_pseudo_labeled_data(pseudo_label_dataset: Dataset, train_dataset: Dataset):
    if pseudo_label_dataset is not None:
        print(
            f"Pseudo-labels generated. len(Psedo_Label_Dataset) = {len(pseudo_label_dataset)}"
        )
        # 将伪标签数据集添加到训练集中
        ConcatDataset = torch.utils.data.ConcatDataset(
            [train_dataset, pseudo_label_dataset]
        )
        train_data = dataProcess.create_dataloader(
            dataset=ConcatDataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            drop_last=True,
        )
    else:
        print("制造伪标签失败，继续使用原训练集")
        train_data = dataProcess.create_dataloader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )

    return train_data


def add_valid_data(train_dataset: Dataset, valid_dataset: Dataset):
    """
    将验证集数据添加到训练集中
    :param train_dataset: 训练集
    :param valid_dataset: 验证集
    :return: 合并后的训练集
    """
    print("Adding validation data to training set...")
    ConcatDataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
    train_data = dataProcess.create_dataloader(
        dataset=ConcatDataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )
    return train_data


def test(model: MyModel, test_data: DataLoader):
    """
    Predict results for the entire test set.
    :param model: The trained model.
    :param test_data: DataLoader for the test set.
    :return: A list of predicted class IDs.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    print("Making predictions on the test set...")
    with torch.no_grad():  # Disable gradient calculation
        for data, _ in tqdm(test_data):  # We don't need labels for prediction
            data = data.to(device)
            outputs = model(data)
            _, predicted_label = torch.max(outputs, 1)
            predictions.extend(predicted_label.cpu().numpy().tolist())
    print("Finished making predictions.")
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
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    print(f"Predictions saved to {filepath}.")
