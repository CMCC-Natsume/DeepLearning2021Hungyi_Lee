from torch import nn
import torch


class MyModel(nn.Module):
    """
    自定义模型
    该模型为一个简单的
    包含
    输入为
    输出为600个类别的概率分布
    """

    def __init__(self, d_model=150, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=1
        )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            # nn.Linear(d_model, d_model),
            # nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )
        self.attention_pooling = SelfAttentionPooling(input_dim=d_model)

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
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        # stats = out.mean(dim=1)
        stats, _ = self.attention_pooling(out)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        # 定义一个线性层来计算注意力分数
        # 通常会有一个非线性激活函数 (tanh 或 relu)
        self.W = nn.Linear(
            input_dim, input_dim
        )  # 或者可以映射到一个更小的维度，然后再映射回来
        self.v = nn.Parameter(torch.rand(input_dim))  # 上下文向量，可学习参数

    def forward(self, x, mask=None):
        # x 预期形状: (batch_size, sequence_length, input_dim)

        # 1. 计算原始注意力分数
        # tanh(W * x)
        # 形状: (batch_size, sequence_length, input_dim)
        attn_scores = torch.tanh(self.W(x))

        # 将上下文向量 v 广播到与 attn_scores 相同的大小，并进行点积
        # (batch_size, sequence_length, input_dim) * (input_dim) -> (batch_size, sequence_length)
        attn_scores = torch.matmul(attn_scores, self.v)

        # 2. 归一化注意力权重
        # 形状: (batch_size, sequence_length)
        if mask is not None:
            # 如果有 mask (用于处理填充)，将填充位置的注意力分数设置为非常小的值
            attn_scores = attn_scores.masked_fill(
                mask == 0, -1e9
            )  # mask should be (batch_size, sequence_length)

        attn_weights = torch.softmax(attn_scores, dim=1)  # 对序列长度维度进行softmax

        # 确保 attn_weights 的形状为 (batch_size, sequence_length, 1) 以便进行元素乘法
        attn_weights = attn_weights.unsqueeze(2)

        # 3. 加权求和
        # (batch_size, sequence_length, input_dim) * (batch_size, sequence_length, 1)
        # 结果形状: (batch_size, input_dim)
        pooled_output = torch.sum(x * attn_weights, dim=1)

        return pooled_output, attn_weights  # 可以选择也返回注意力权重以便分析
