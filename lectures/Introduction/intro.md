# Machine Learning 学习笔记 - Regression (v16)

## 1. 机器学习基础概念
- **Machine Learning ≈ Looking for Function**  
  ML的核心就是寻找一个能够解决问题的函数(function)
- 典型应用场景:
  - Speech Recognition (语音识别)
  - Image Recognition (图像识别)
  - Playing Go (围棋对弈)

## 2. 不同类型的函数
### 2.1 Regression (回归)
- 输出: scalar value (标量)
- 例子:
  - 预测PM2.5数值
  - 垃圾邮件过滤(Spam filtering)

### 2.2 Classification (分类)
- 输出: 从给定的classes中选择正确的类别
- 例子:
  - 围棋落子位置预测(19x19个可能的落点)
  - 图像识别("猫"/"狗")

### 2.3 Structured Learning (结构化学习)
- 输出: 具有结构的复杂对象
  - 例如生成图片/文档

## 3. 寻找函数的步骤 (ML Framework)
### 3.1 定义模型 (Model with Unknown Parameters)
- 基于domain knowledge设计模型
- 包含:
  - weight (权重)
  - bias (偏置)
  - feature (特征)

### 3.2 定义损失函数 (Loss Function)
- 衡量模型预测的好坏
- 常用loss:
  - Cross-entropy (交叉熵)
  - Mean Squared Error (均方误差)
- Error Surface (误差曲面): 可视化loss随参数变化

### 3.3 优化 (Optimization)
- **Gradient Descent (梯度下降)**:
  - 通过计算梯度更新参数
  - 可能遇到的问题:
    - Local minima (局部最小值)
    - Global minima (全局最小值)

- **Batch Processing (批处理)**:
  - 1 epoch = 遍历所有数据一次
  - 例子:
    - N=10,000样本，B=100 → 100 updates/epoch

## 4. 模型复杂度问题
### 4.1 Linear Models的局限性
- 在unseen data上表现差 (如2021年的预测数据)
- 原因: Model Bias (模型偏差)

### 4.2 更灵活的模型
- **Piecewise Linear (分段线性)**:
  - 使用多个sigmoid函数组合
  - 可以调整:
    - slope (斜率)
    - shift (偏移)
    - height (高度)

- **Activation Functions (激活函数)**:
  - Sigmoid
  - ReLU (Rectified Linear Unit) → 实际效果更好

## 5. 神经网络 (Neural Networks)
### 5.1 基本结构
- 多层hidden layers
- 每层包含多个neurons (神经元)
- 深度模型示例:
  - AlexNet (8层)
  - VGG (16/19层)
  - GoogleNet (22层)
  - Residual Net (152层)

### 5.2 Deep vs Fat
- 为什么选择deep而不是wide?
  - 实验证明deep networks性能更好
  - 但可能带来overfitting (过拟合)问题

### 5.3 过拟合问题
- 表现:
  - Training data上表现好
  - Unseen data上表现差
- 解决方案:
  - 需要合适的model selection策略

## 6. 关键知识点总结
| 概念 | 英文 | 说明 |
|------|------|------|
| 回归 | Regression | 输出连续值 |
| 分类 | Classification | 输出离散类别 |
| 梯度下降 | Gradient Descent | 优化算法 |
| 批次大小 | Batch Size | 每次更新的样本数 |
| 激活函数 | Activation Function | ReLU/Sigmoid |

## 7. 学习资源
- [Machine Learning Basics](https://youtu.be/Dr-WRlEFefw)
- [Backpropagation详解](https://youtu.be/ibJpTrp5mcE)

> 注：实际应用中需要balance model complexity和generalization performance
