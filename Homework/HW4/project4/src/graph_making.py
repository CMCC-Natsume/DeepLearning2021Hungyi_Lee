import matplotlib
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # 强制使用非交互式后端(WSL中使用时需要添加本行)


def plot_loss_curve(train_loss, dev_loss, title="", valid_steps=2000):
    """
    Plot training and validation loss curves against training steps.

    Args:
        train_loss (list): List of training losses recorded per step.
        dev_loss (list): List of validation losses recorded every valid_steps.
        title (str): Title of the plot.
        valid_steps (int): Number of steps between validation checks (default: 2000).
    """
    # Convert tensor values to float
    train_loss = [
        t.cpu().item() if isinstance(t, torch.Tensor) else t for t in train_loss
    ]
    dev_loss = [t.cpu().item() if isinstance(t, torch.Tensor) else t for t in dev_loss]

    # Define x-axis for training and validation losses
    train_steps = list(range(1, len(train_loss) + 1))  # Steps: 1, 2, ..., 70000
    valid_steps = [
        i * valid_steps for i in range(1, len(dev_loss) + 1)
    ]  # Steps: 2000, 4000, ...

    # Set y-axis limits
    min_loss = 0
    max_loss = max(max(train_loss), max(dev_loss)) + 0.1

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, c="tab:red", label="Train Loss", alpha=0.7)
    plt.plot(
        valid_steps,
        dev_loss,
        c="tab:cyan",
        label="Validation Loss",
        marker="o",
        linestyle="--",
        markersize=8,
    )

    # Customize plot
    plt.ylim(min_loss, max_loss)
    plt.xlabel("Training Steps")
    plt.ylabel("CrossEntropyLoss")
    plt.title(f"Learning Curve: {title} (Loss)")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(
        "Homework/HW4/project4/saved_graphs/learning_curve_loss.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_accuracy_curve(train_acc, dev_acc, title="", valid_steps=2000):
    """
    Plot training and validation accuracy curves against training steps.

    Args:
        train_acc (list): List of training accuracies recorded per step.
        dev_acc (list): List of validation accuracies recorded every valid_steps.
        title (str): Title of the plot.
        valid_steps (int): Number of steps between validation checks (default: 2000).
    """
    # Convert tensor values to float
    train_acc = [
        t.cpu().item() if isinstance(t, torch.Tensor) else t for t in train_acc
    ]
    dev_acc = [t.cpu().item() if isinstance(t, torch.Tensor) else t for t in dev_acc]

    # Define x-axis for training and validation accuracies
    train_steps = list(range(1, len(train_acc) + 1))  # Steps: 1, 2, ..., 70000
    valid_steps = [
        i * valid_steps for i in range(1, len(dev_acc) + 1)
    ]  # Steps: 2000, 4000, ...

    # Set y-axis limits
    min_acc = min(min(train_acc), min(dev_acc)) - 0.05
    max_acc = max(max(train_acc), max(dev_acc)) + 0.05

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_acc, c="tab:blue", label="Train Accuracy", alpha=0.7)
    plt.plot(
        valid_steps,
        dev_acc,
        c="tab:orange",
        label="Validation Accuracy",
        marker="o",
        linestyle="--",
        markersize=8,
    )

    # Customize plot
    plt.ylim(min_acc, max_acc)
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {title} (Accuracy)")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(
        "Homework/HW4/project4/saved_graphs/learning_curve_accuracy.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
