import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

MAX_EPOCH = 1000
LEARNING_RATE = 0.001
MOMENTUM = 0.9

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

seed = 42069
torch.manual_seed(seed)

class MyModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 39),
        )
        self.criterion = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, input_data):
        return self.network(input_data)



    def calculate_loss(self, prediction, label):
        """
        标记
        :return:
        """
        return self.criterion(prediction, label)




def model_training(train_data: DataLoader, dev_data: DataLoader, model: MyModel):
    train_loss = []
    dev_loss = []
    train_accuracy = 0.0
    dev_accuracy = 0.0
    min_loss = 100
    epoch = 1
    my_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    while epoch < MAX_EPOCH:
        model.train()
        for data, label in train_data:
            data = data.to(device)
            label = label.to(device)
            my_optimizer.zero_grad()
            outputs = model(data)
            loss = model.calculate_loss(outputs, label)
            _, predicted_label = torch.max(outputs, 1)
            loss = loss.to(device)
            train_loss.append(loss.detach())
            loss.backward()
            my_optimizer.step()
            train_accuracy += (predicted_label == label).sum().item()

        train_accuracy /= len(train_data.dataset)
        the_loss = dev(model, dev_data)
        if the_loss < min_loss:
            min_loss = the_loss
            print(f"  NOW! in epoch: {epoch}, the lowest loss is {the_loss}")
        dev_loss.append(the_loss.detach())
        print(f"epoch: {epoch}, train_loss: {train_loss[epoch]:3.6f}, dev_loss: {the_loss:3.6f}, train_accuracy: {train_accuracy:3.6f}")
        train_accuracy = 0.0
        dev_accuracy = 0.0
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



