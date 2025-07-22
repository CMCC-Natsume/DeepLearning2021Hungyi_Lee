from torch import nn


class MyModel(nn.Module):
    """
    自定义模型
    该模型为一个简单的
    包含
    输入为
    输出为600个类别的概率分布
    """

    def __init__(self, d_model=70, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=1
        )

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            # nn.Linear(d_model, d_model),
            # nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        """
        args:
          x: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(x)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch  size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)
