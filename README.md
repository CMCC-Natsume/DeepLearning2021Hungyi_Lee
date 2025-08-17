<h1 align="center">李宏毅机器学习2021课程课件/作业分享（暂停更新）</h1>

---

## 💻参考资料/链接

- #### 🔥2021课程主页：[点击此处](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)

- #### 💻2021课程视频（Bilibili）：[点击此处](https://www.bilibili.com/video/BV1Wv411h7kN)

- #### 📚李宏毅老师的Youtube首页：[点击此处](https://www.youtube.com/@HungyiLeeNTU)

<br>

#### 以下为课程作业与主题的对应关系

>目前只做了前四个HW

| | Topic | | Topic |
|---|---|---|---|
| x | Colab Tutorial | HW9 | Explainable AI |
| x | PyTorch Tutorial | HW10 | Attack |
| HW1 | Regression | HW11 | Adaptation |
| HW2 | Classification | HW12 | Reinforcement Learning |
| HW3 | CNN | HW13 | Network Compression |
| HW4 | Self-attention | HW14 | Life-long Learning |
| HW5 | Transformer | HW15 | Meta Learning |
| HW6 | Generative Model | | |
| HW7 | BERT | | |
| HW8 | Auto-encoder | | |

---
## 🧾食用方法

本项目的具体执行方案很多地方都与示例代码的方案一致，环境构建工具使用的是`uv`。如果你也想和作者一样使用uv作为自己的环境构建工具，可以点击链接参考[官方文档](https://docs.astral.sh/uv/reference/environment/)，或进入其[github仓库](https://github.com/astral-sh/uv)参考用法。

---

## 🗂️项目结构/内容

>由于**文件内容较多**，这里只介绍了一些代表性的项目结构：
>（比如Homework模块只挑出了HW2和HW3作为**代表**显示大概的内容）

```
DL2021
├── Homework
│   ├── HW2
│   │   ├── HW02.pdf
│   │   └── project2
│   │       ├── README.md
│   │       ├── dataProcess.py
│   │       ├── graphMaking.py
│   │       ├── main.py
│   │       ├── model.py
│   │       ├── project_test.py
│   │       ├── pyproject.toml
│   │       ├── savedGraph
│   │       ├── savedModel
│   │       ├── uv.lock
│   │       ├── 修改经验01.md
│   │       └── 修改经验02.md
│   │
│   ├── HW3
│   │   ├── HW03.pdf
│   │   ├── experiments
│   │   │   ├── image-1.png
│   │   │   ├── image.png
│   │   │   ├── 修改经验01.md
│   │   │   └── 修改经验02.md
│   │   ├── project3
│   │   │   ├── README.md
│   │   │   ├── pyproject.toml
│   │   │   ├── savedGraph
│   │   │   ├── saved_models
│   │   │   ├── src
│   │   │   ├── test
│   │   │   └── uv.lock
│   │   └── submission
│   │
│   ├── HW……
│   │   └── HW…….pdf
│   │
│   └── resources
│
├── lectures
│   ├── CNN&Self_Attention
│   ├── DL_Classification
│   ├── GAN
│   ├── Introduction
│   ├── Normalization&Seq2Seq
│   ├── PAC_Learning
│   └── ……
│
└── 杂记
```

---

## 🌟一些作者想说的话

个人认为2021DL这份学习计划涉及的内容面还是比较广的，对应来说作业中就会有不少工程细节（虽然有示例代码提供），所以作者只会更新自己感兴趣的作业的对应代码。
另外，课程中理论部分偏少，总让本人有种不太安心的感觉，个人建议可以配合其他课程/相关论文/开源项目进行学习。
