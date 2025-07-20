这个脚本可以在你每次训练达到新高度（如验证集准确率、损失、F1-score 等指标有提升）时自动执行以下任务：

1. 保存超参数（学习率、网络结构、动量、调度器等）。
2. 保存训练图（如 loss/accuracy 曲线）。
3. 将所有内容保存到一个以时间戳命名的新建文件夹中。

下面是一个功能完整的 Python 脚本示例，假设你使用的是 PyTorch（如果你使用其他框架，我可以帮你改）：

---

### ✅ `auto_save_checkpoint.py`

```python
import os
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def save_training_snapshot(model, optimizer, scheduler, metrics, plots=None, extra_info=None, save_dir='checkpoints'):
    """
    保存当前训练的快照（模型信息 + 配置 + 图表）。

    Args:
        model: PyTorch 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        metrics: 字典，包含如 {'val_accuracy': 0.92, 'val_loss': 0.3} 等关键指标
        plots: 字典，如 {'loss_curve': <matplotlib Figure 对象>, ...}
        extra_info: 额外信息字典，例如 {'batch_size': 64, 'momentum': 0.9}
        save_dir: 根目录，默认 'checkpoints'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_path = os.path.join(save_dir, f'run_{timestamp}')
    os.makedirs(folder_path, exist_ok=True)

    # 保存模型结构和参数
    torch.save(model.state_dict(), os.path.join(folder_path, 'model_weights.pth'))

    # 保存训练配置
    config = {
        'learning_rate': optimizer.param_groups[0]['lr'],
        'momentum': optimizer.param_groups[0].get('momentum', None),
        'scheduler': str(type(scheduler).__name__) if scheduler else None,
        'metrics': metrics
    }

    if extra_info:
        config.update(extra_info)

    with open(os.path.join(folder_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 保存图表
    if plots:
        for name, fig in plots.items():
            fig_path = os.path.join(folder_path, f'{name}.png')
            fig.savefig(fig_path)
            plt.close(fig)

    print(f"[✓] Training snapshot saved to: {folder_path}")
```

---

### 🧪 使用示例：

你可以在训练循环中这样用它（假设你在验证集上得到了一个新的最佳分数）：

```python
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    plots = {
        'loss_curve': plot_loss_curve(loss_history),
        'accuracy_curve': plot_accuracy_curve(acc_history),
    }

    save_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={'val_accuracy': val_accuracy, 'val_loss': val_loss},
        plots=plots,
        extra_info={
            'batch_size': 64,
            'network_architecture': str(model),
        }
    )
```

---

### 📈 示例绘图函数（可选）：

```python
def plot_loss_curve(losses):
    fig, ax = plt.subplots()
    ax.plot(losses, label='Loss')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig

def plot_accuracy_curve(accuracies):
    fig, ax = plt.subplots()
    ax.plot(accuracies, label='Accuracy')
    ax.set_title('Accuracy Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    return fig
```

---

是否希望我帮你将这个脚本与你的训练主循环自动集成？如果你提供你目前使用的训练代码框架（比如是否用 PyTorch、`Trainer` 类、`Keras` 等），我可以帮你进一步定制。
