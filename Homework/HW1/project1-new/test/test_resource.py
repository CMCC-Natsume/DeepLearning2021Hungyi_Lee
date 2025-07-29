import csv

import numpy
import torch
from termcolor import colored

data_root = "Homework/resources/HW1/"


def csv_fileReader(path: str) -> numpy.ndarray:
    with open(path) as file:
        csv_list = list(csv.reader(file))
        data = numpy.array(csv_list)
        data = data[1:, 1:]
        data = data.astype(float)
        return data


print()
# 01.csv文件形状
print(colored("三个csv文件的形状为:", "blue"))
print(f"Train.csv Shape:\t\t{csv_fileReader(data_root + 'covid.train.csv').shape}")
print(f"Test.csv Shape:\t\t\t{csv_fileReader(data_root + 'covid.test.csv').shape}")
print(
    f"SampleSub.csv Shape:\t\t{csv_fileReader(data_root + 'sampleSubmission.csv').shape}"
)

# 02.data/target数据样式
print(colored("\n测试Dataset中的数据导入部分", "yellow"))
my_data = csv_fileReader(data_root + "covid.train.csv")
data = torch.tensor(my_data[:, :-1], dtype=torch.float32)
targets = torch.tensor(my_data[:, -1], dtype=torch.float32)

print(f"Data Shape:\t\t{data.shape}")
print(f"Data Type:\t\t{data.dtype}")
print(f"data缩略图如下\n{data}\n")

print(f"Targets Shape:\t\t{targets.shape}")
print(f"Targets Type:\t\t{targets.dtype}")
print(f"Targets缩略图如下:\n{targets}\n")

# 03.测试是否能正确进行标准化的采样
print(colored("测试标准化处理(从index40开始需要标准化): 38列-->43列", "green"))
print(f"标准化处理前--Data缩略图如下:\n{data[0:3, 38:44]}\n")
data[:, 40:] = (data[:, 40:] - data[:, 40:].mean(dim=0)) / data[:, 40:].std(dim=0)
print(f"标准化处理后--Data缩略图如下:\n{data[0:3, 38:44]}\n")
