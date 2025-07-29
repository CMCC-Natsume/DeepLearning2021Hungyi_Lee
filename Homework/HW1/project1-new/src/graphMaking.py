import matplotlib.pyplot as plt
import numpy as np
import os


class PlottingManager:
    """
    一个用于绘制和保存深度学习模型学习曲线的类。
    适用于在没有图形界面的环境（如WSL2）中运行。
    """

    def __init__(self, save_dir="plots"):
        """
        初始化绘图管理器。
        Args:
            save_dir (str): 保存图像的目录。
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # 设置matplotlib后端为'Agg'，这样就不会尝试显示图形窗口
        plt.switch_backend("Agg")
        print(f"绘图将保存到目录: {os.path.abspath(self.save_dir)}")

    def plot_learning_curve(
        self, train_losses, dev_losses, model_name="MyModel", focus_portion=1.0
    ):
        """
        绘制训练损失和验证损失的学习曲线，并将其保存为图片。

        Args:
            train_losses (list): 每个epoch的平均训练损失列表。
            dev_losses (list): 每个epoch的平均验证损失列表。
            model_name (str): 模型的名称，用于文件名和图表标题。
            focus_portion (float): 如果小于1.0，则只绘制曲线的最后一部分。
            例如，0.6 表示绘制最后60%的数据点。
        """
        if not train_losses or not dev_losses:
            print("警告：训练损失或验证损失数据为空，无法绘制学习曲线。")
            return

        num_epochs = len(train_losses)
        start_index = int(num_epochs * (1 - focus_portion))

        epochs = np.arange(start_index, num_epochs)
        train_data_to_plot = train_losses[start_index:]
        dev_data_to_plot = dev_losses[start_index:]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_data_to_plot, label="Training Loss")
        plt.plot(epochs, dev_data_to_plot, label="Validation Loss")

        plt.title(f"{model_name} Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 构建保存路径和文件名
        file_name = f"{model_name}_learning_curve.png"
        save_path = os.path.join(self.save_dir, file_name)

        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存，这在非交互式环境中很重要
        print(f"学习曲线已保存到: {save_path}")


# 在main.py中使用这个绘图类
# 假设你的main.py中已经导入了graphMaking
#
# 示例用法:
# # 5.训练结束，结果绘制：
# plot_manager = graphMaking.PlottingManager(save_dir="plots") # 可以指定保存目录
# plot_manager.plot_learning_curve(train_loss, dev_loss, "MyModel", focus_portion=0.6)
