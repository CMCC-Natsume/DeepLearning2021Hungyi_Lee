import matplotlib
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")  # 强制使用非交互式后端(WSL中使用时需要添加本行)

# 设置中文显示
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    true_labels,
    pred_labels,
    title="Confusion Matrix",
    save_path="Homework/HW2/project2/savedGraph/confusion_matrix.png",
):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 动态设置颜色范围（基于数据最大值）
    vmin = 0
    vmax = np.percentile(cm, 98)  # 避免 vmax 为 0

    # 设置绘图风格
    plt.figure(figsize=(12, 10), dpi=300)  # 增加分辨率
    font_size = max(4, 300 // cm.shape[0]) - 2
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": font_size},
    )

    # 设置英文标签
    plt.xlabel("Predict labels", fontsize=12)
    plt.ylabel("Real labels", fontsize=12)
    plt.title(title, fontsize=14)

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # 提高保存分辨率
    plt.close()


def plot_learning_curve(train_loss, dev_loss, title=""):
    train_loss = [
        t.cpu().item() if isinstance(t, torch.Tensor) else t for t in train_loss
    ]
    dev_loss = [t.cpu().item() if isinstance(t, torch.Tensor) else t for t in dev_loss]
    min_loss = min(min(train_loss), min(dev_loss)) - 0.1
    max_loss = max(max(train_loss), max(dev_loss)) + 0.1
    plt.figure(1, figsize=(6, 4))
    plt.plot(range(len(train_loss)), train_loss, c="tab:red", label="train")
    plt.plot(range(len(dev_loss)), dev_loss, c="tab:cyan", label="dev")
    plt.ylim(min_loss, max_loss)  # 设置y轴范围
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.savefig(
        "Homework/HW2/project2/savedGraph/learning_curve.png", bbox_inches="tight"
    )  # 保存学习曲线
    plt.close()  # 关闭图形，避免内存泄漏


def plot_pred(dv_set, model, device, lim=35.0, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    plt.figure(2, figsize=(5, 5))
    plt.scatter(targets, preds, c="r", alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c="b")
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel("ground truth value")
    plt.ylabel("predicted value")
    plt.title("Ground Truth v.s. Prediction")
    plt.savefig("prediction_scatter.png", bbox_inches="tight")  # 保存预测散点图
    plt.close()  # 关闭图形
