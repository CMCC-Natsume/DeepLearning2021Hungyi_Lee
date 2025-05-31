import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

MAX_EPOCH = 1000
LEARN_RATE = 0.001
MOMENTUM = 0.9

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 标记
seed = 42069
torch.manual_seed(seed)


class MyModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.network2 = nn.Sequential(
            nn.Linear(input_dim, 100), nn.ReLU(), nn.Linear(100, 1)
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, input_data):
        return self.network2(input_data).squeeze(1)  # 标记

    def calculate_loss(self, prediction, label):
        """
        标记
        :return:
        """
        return self.criterion(prediction, label)


def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_loss = []
    dev_loss = []
    min_loss = 100
    epoch = 1
    my_optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
    while epoch < MAX_EPOCH:
        model.train()
        for data, label in train_data:
            data = data.to(device)
            label = label.to(device)
            my_optimizer.zero_grad()
            output_data = model(data)
            loss = model.calculate_loss(output_data, label)
            loss = loss.to(device)
            # print(f"loss is {loss}")
            train_loss.append(loss.detach())
            loss.backward()
            my_optimizer.step()

        the_loss = dev(model, dev_data)
        if the_loss < min_loss:
            min_loss = the_loss
            print(f"epoch: {epoch}, the lowest loss is {the_loss}")
        dev_loss.append(the_loss.detach())

        epoch += 1

    return train_loss, dev_loss


def dev(model: MyModel, dev_data: DataLoader):
    model.eval()  # 标记
    loss = []
    for data, label in dev_data:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss.append(model.calculate_loss(output, label))
    return sum(loss) / len(loss)


# def test(model: MyModel, test_data: DataLoader):
#     """
#     标记
#     :param model:
#     :param test_data:
#     :return:
#     """
#     model.eval()
#     loss = []
#     for data, label in test_data:
#         output = model(data)
#         loss.append(model.calculate_loss(output, label))
#     return sum(loss) / len(loss)
