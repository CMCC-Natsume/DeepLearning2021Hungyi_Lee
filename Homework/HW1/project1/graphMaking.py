import matplotlib
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # 强制使用非交互式后端(WSL中使用时需要添加本行)


def plot_learning_curve(
    train_loss,
    dev_loss,
    title="",
    root="Homework/HW1/project1/saved_graph/learning_curve.png",
    focus_portion=0.25,  # New parameter to define focus
):
    train_loss = [
        t.cpu().item() if isinstance(t, torch.Tensor) else t for t in train_loss
    ]
    dev_loss = [t.cpu().item() if isinstance(t, torch.Tensor) else t for t in dev_loss]

    # Calculate focus index
    num_epochs = len(train_loss)
    start_focus_idx = int(num_epochs * (1 - focus_portion))

    # Get losses for the focused portion
    focused_train_loss = train_loss[start_focus_idx:]
    focused_dev_loss = dev_loss[start_focus_idx:]

    # Calculate min/max for the focused portion, ensuring it's not empty
    if focused_train_loss and focused_dev_loss:
        min_loss = min(min(focused_train_loss), min(focused_dev_loss)) - 0.1
        max_loss = max(max(focused_train_loss), max(focused_dev_loss)) + 0.1
    else:  # Fallback to original calculation if focus portion is too small
        min_loss = min(min(train_loss), min(dev_loss)) - 0.1
        max_loss = max(max(train_loss), max(dev_loss)) + 0.1

    plt.figure(1, figsize=(6, 4))
    plt.plot(range(len(train_loss)), train_loss, c="tab:red", label="train")
    plt.plot(range(len(dev_loss)), dev_loss, c="tab:cyan", label="dev")
    plt.ylim(min_loss, max_loss)  # Apply focused limits
    plt.xlabel("Epoch")
    plt.ylabel("MSE-Loss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.grid(True)
    plt.savefig(root, bbox_inches="tight")
    plt.close()


def plot_pred(dv_set, model, device, lim=35.0, preds=None, targets=None):
    # 此函数与损失曲线无关，保持不变，但为了完整性一并列出
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
    plt.xlabel("Ground Truth Value")  # 标签更清晰
    plt.ylabel("Predicted Value")  # 标签更清晰
    plt.title("Ground Truth vs. Prediction")  # 标题更清晰
    plt.savefig("prediction_scatter.png", bbox_inches="tight")  # 保存预测散点图
    plt.close()  # 关闭图形
