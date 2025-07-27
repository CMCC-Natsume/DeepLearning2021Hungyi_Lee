import math
import os
import model
import torch
from termcolor import colored
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import data_process
import json
import csv

# 参数表
TRAIN_RATIO = 0.9
BATCH_SIZE = 32
NUM_WORKERS = 8
WARMUP_STEPS = 1000
VALID_STEPS = 2000
SAVED_STEPS = 10000
MAX_STEP = 80000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
add_valid_data_into_training = False
save_dir = "Homework/HW4/project4/saved_models"
output_path = "predict.csv"
NCOL = 90

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def model_training(
    train_data: DataLoader,
    dev_data: DataLoader,
    model: model.MyModel,
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
    batch_loss = []
    total_train_acc = []
    total_dev_acc = []
    train_accuracy = 0.0
    dev_accuracy = 0.0
    max_dev_accuracy = -1.0

    my_optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        my_optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEP
    )
    train_iterator = iter(train_data)
    model.to(device)

    # 定义保存模型的路径
    os.makedirs(save_dir, exist_ok=True)  # 创建目录如果不存在
    best_model_path = os.path.join(save_dir, "best_model.pth")
    pbar = tqdm(total=VALID_STEPS, ncols=NCOL, desc="Train", unit="  step")

    # 本次训练因为使用特别的调度器，因此以step为循环控制变量
    for step in range(MAX_STEP):
        model.train()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_data)
            batch = next(train_iterator)

        train_accuracy = 0.0
        dev_accuracy = 0.0
        batch_loss.clear()

        # 单个batch内的训练过程：
        data, label = batch
        data, label = data.to(device), label.to(device)
        my_optimizer.zero_grad()
        outputs = model(data)
        loss = model.calculate_loss(outputs, label)
        loss = loss.to(device)
        batch_loss.append(loss.detach().item())
        loss.backward()
        predicted_label = outputs.argmax(1)
        my_optimizer.step()
        scheduler.step()

        # 训练信息反馈
        train_accuracy += (predicted_label == label).sum().item()
        train_accuracy /= len(data)
        total_train_acc.append(train_accuracy)
        train_loss.append(sum(batch_loss) / len(batch_loss))
        # 验证集计算loss

        pbar.update()
        pbar.set_postfix(
            accuracy=f"{train_accuracy:.2f}",
            step=step + 1,
        )
        if (step + 1) % VALID_STEPS == 0:
            pbar.close()
            print(f"Step: {step + 1}. ")
            val_loss, dev_accuracy = model_validation(model, dev_data)
            dev_loss.append(val_loss)
            total_dev_acc.append(dev_accuracy)
            if dev_accuracy > max_dev_accuracy:
                max_dev_accuracy = dev_accuracy
                torch.save(model.state_dict(), best_model_path)
                pbar.write(
                    colored(
                        f"Step {step + 1}, best model saved. (accuracy={max_dev_accuracy:.4f})",
                        "yellow",
                    )
                )
            print("\n")
            pbar = tqdm(total=VALID_STEPS, ncols=NCOL, desc="Train", unit=" step")

        # 留作Early stopping

    pbar.close()
    print(colored(f"\nThe highest valid accuracy is {max_dev_accuracy}", "blue"))
    return train_loss, dev_loss, total_train_acc, total_dev_acc


def get_cosine_schedule_with_warmup(
    optimizer: optim,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    学习率调度器，余弦退火
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
          The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
          The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
          The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
          The number of waves in the cosine schedule (the defaults is to just   decrease from the max value to 0
          following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
          The index of the last epoch when resuming training.

    Return:
        obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_validation(model: model.MyModel, dev_data: DataLoader):
    model.eval()
    loss = []
    dev_accuracy = 0.0
    # 计算验证集的损失
    for data, label in tqdm(dev_data, ncols=NCOL, desc="Valid"):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            output = model(data)
            loss.append(model.calculate_loss(output, label).item())
            _, predicted_label = torch.max(output, 1)
            dev_accuracy += (predicted_label == label).sum().item()
    # 计算平均损失
    dev_accuracy /= len(dev_data.dataset)
    return sum(loss) / len(loss), dev_accuracy


def test(model: model.MyModel, data_dir: str):
    """
    Predict results for the entire test set.
    :param model: The trained model.
    :param test_data: DataLoader for the test set.
    :return: A list of predicted class IDs.
    """
    model.eval()  # Set the model to evaluation mode
    print("Making predictions on the test set...")
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = data_process.InferenceDataset(data_dir)
    test_data = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=data_process.inference_collate_batch,
    )

    speaker_num = len(mapping["id2speaker"])
    model.eval()
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(test_data, ncols=NCOL):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])
    with open("Homework/HW4/submission/submission.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
