import torch
from src import dataProcess
from src.model import MyModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_pseudo_labels(dataset: Dataset, model: MyModel):
    """获取伪标签
    :param dataset: 数据集
    :param model: 模型
    :return: PseudoLabelDataset
    """
    selected_images = []
    selected_labels = []

    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)  # cross entropy中的softmax是在creterion()时进行的

    for batch in tqdm(data_loader):
        data, _ = batch

        with torch.no_grad():
            logits = model(data.to(device))
        probabilities = softmax(logits)
        confidence, pseudo_labels = torch.max(probabilities, dim=1)
        mask = confidence > 0.65  # 筛选置信度大于阈值的样本

        if mask.any():
            selected_images.append(data[mask])  # shape = [N_i, C, H, W]
            selected_labels.append(pseudo_labels[mask])

    model.train()
    if selected_images:
        # 每一组筛选出的图像的第0维不同（如第一组3张、第二组6张，在第0维将其合并为总的9张）
        selected_images = torch.cat(selected_images, dim=0)
        selected_labels = torch.cat(selected_labels, dim=0)
        return selected_images, selected_labels
    else:
        return None, None


data_root = "Homework/resources/HW3/food-11/"
(train_dataset, valid_dataset, unlabeled_dataset, test_dataset) = (
    dataProcess.create_dataset(data_root=data_root)
)
my_model = MyModel()
my_model.to(device)
get_pseudo_labels(unlabeled_dataset, model=my_model)
