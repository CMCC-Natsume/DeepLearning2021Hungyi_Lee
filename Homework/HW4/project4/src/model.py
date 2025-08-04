from torch import nn
import torch
import torch.nn.functional as F


class MyModel(nn.Module):
    """
    自定义模型
    该模型为一个简单的
    包含
    输入为
    输出为600个类别的概率分布
    """

    def __init__(self, d_model=160, n_spks=600, dropout=0.1, nhead=8, kernel_size=3):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8
        )  # 留作对照，未放入模型中

        self.conformer = Conformer(
            input_dim=d_model,
            dim_feedforward=256,
            kernel_size=kernel_size,
            dropout=dropout,
            nhead=nhead,
        )

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Linear(d_model, n_spks)
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
        out = self.conformer(out)

        # stats = out.mean(dim=1)  # 原模型的平均池化
        stats, _ = self.attention_pooling(out)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

    def calculate_loss(self, prediction, label):
        return self.criterion(prediction, label)


class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_classes, s=30.0, m=0.35):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)  # Initialize weights

    def forward(self, x, labels):
        # Normalize the feature vectors and weights
        x = F.normalize(x, p=2, dim=1)  # L2 normalize input features
        weight = F.normalize(self.weight, p=2, dim=1)  # L2 normalize weights

        # Compute cosine similarity: (batch_size, num_classes)
        cosine = F.linear(x, weight)

        # AM-Softmax: apply margin to the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = self.s * (cosine - one_hot * self.m)

        # Apply softmax and compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        return loss


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


class Conformer(nn.Module):
    def __init__(
        self, input_dim, dim_feedforward=256, dropout=0.1, kernel_size=31, nhead=8
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.feed_forward = FeedForward(input_dim, dim_feedforward, dropout)
        self.self_attention = MultiheadSelfAttention(
            input_dim, num_heads=nhead, dropout=dropout
        )
        self.convolution = ConvolutionModule(input_dim)

    def forward(self, x):
        x = self.feed_forward(x)
        x = self.self_attention(x)  # out: torch.Size([32, 128, 160])
        x = self.convolution(x)
        x = self.feed_forward(x)
        x = self.layer_norm(x)
        return x  # 返回处理后的特征


class FeedForward(nn.Module):
    def __init__(self, input_dim, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x = self.swish(x)  # 按照原论文不需要二次激活
        x = self.dropout(x)
        return 0.5 * x + residual


class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # x shape:
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return x + residual


class ConvolutionModule(nn.Module):
    def __init__(self, input_dim, kernel_size=3, padding=1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        # self.pointwise_conv = nn.Conv1d(
        #     input_dim, 2 * input_dim, kernel_size=kernel_size, padding=padding
        # )  # 和下方的函数应该起的是一样的作用，因此选择下方的做法
        self.pointwise1 = nn.Linear(in_features=input_dim, out_features=input_dim * 2)
        self.pointwise2 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.glu = nn.GLU(dim=-1)
        self.depthwise_conv = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=input_dim,
        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        # x shape: (batch_size, input_dim, sequence_length)
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x = self.glu(x)  # Apply GLU activation(截断一半)
        x = x.transpose(1, 2)  # Change to (batch_size, sequence_length, input_dim)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.transpose(1, 2)
        x = self.pointwise2(x)
        x = self.Dropout(x)

        return x + residual
