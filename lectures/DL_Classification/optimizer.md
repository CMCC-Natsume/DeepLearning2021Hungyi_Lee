
# Optimizer Techniques (v4) 学习笔记

## 1. 核心问题：Error Surface特性
- **Rugged Error Surface** (崎岖的误差曲面)
  - 传统梯度下降容易卡在：
    - Critical points (临界点)
    - Local minima (局部最小值)
  - 即使convex surface也会训练困难

## 2. 自适应学习率 (Adaptive Learning Rate)
### 2.1 基本思想
- **不同参数需要不同学习率**：
  - 公式：$\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta}{\sigma_i^t} g_i^t$
  - $\sigma_i^t$反映历史梯度大小

### 2.2 具体方法
| 方法        | 计算公式                                                                 | 特点                                                                 |
|-------------|--------------------------------------------------------------------------|----------------------------------------------------------------------|
| **Adagrad** | $\sigma_i^t = \sqrt{\frac{1}{t+1}\sum (g_i^t)^2}$                       | 所有历史梯度平方平均                                                 |
| **RMSProp** | $\sigma_i^t = \sqrt{\alpha(\sigma_i^{t-1})^2 + (1-\alpha)(g_i^t)^2}$    | 指数加权移动平均 ($\alpha$通常取0.9)                                 |
| **Adam**    | 结合Momentum和RMSProp                                                   | 最常用优化器 ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)       |

> Adam伪代码关键步骤：
> 1. 计算一阶矩估计：$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
> 2. 计算二阶矩估计：$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
> 3. 偏差校正：$\hat{m}_t = m_t/(1-\beta_1^t)$
> 4. 参数更新：$\theta_t \leftarrow \theta_{t-1} - \alpha \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$

## 3. 学习率调度 (Learning Rate Scheduling)
### 3.1 常见策略
- **Learning Rate Decay**：
  - 随训练步数增加逐渐降低$\eta^t$
  - 原因：接近最优解时需要更精细调整

- **Warm Up**：
  - 初期逐步提高学习率（如ResNet先用0.01 warmup 400步）
  - 解决早期$\sigma_i^t$估计方差大的问题

### 3.2 典型应用
- **Transformer**：
  ```math
  lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
  ```
- **RAdam**：
  - 改进Adam的warmup阶段
  - 论文：https://arxiv.org/abs/1908.03265

## 4. 优化技术对比
| 技术                | 核心改进                          | 数学表达                          |
|---------------------|-----------------------------------|-----------------------------------|
| Vanilla Gradient Descent | 基础算法                    | $\theta_i^{t+1} \leftarrow \theta_i^t - \eta g_i^t$ |
| Momentum            | 考虑梯度方向的历史累积            | $m_t = \beta m_{t-1} + g_t$       |
| Adaptive Methods    | 自适应调整各参数学习率            | $\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta}{\sigma_i^t} m_i^t$ |

## 5. 前沿发展
- **Error Surface改造**：
  - 思想："If the mountain won't move, build a road around it"
  - 参考论文：https://arxiv.org/abs/1712.09913

## 6. 学习资源
- [优化算法视频讲解(中文)](https://youtu.be/4pUmZ8hXIHM)
- [Adam原论文](https://arxiv.org/pdf/1412.6980.pdf)
- [ResNet论文](https://arxiv.org/abs/1512.03385)

> 注：实际应用中需根据error surface特性选择优化策略，通常Adam+Learning Rate Scheduling是安全选择
```
