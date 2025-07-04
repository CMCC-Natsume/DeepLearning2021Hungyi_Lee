import matplotlib
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # 强制使用非交互式后端(WSL中使用时需要添加本行)


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
        "Homework/HW3/project3/savedGraph/learning_curve.png", bbox_inches="tight"
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
