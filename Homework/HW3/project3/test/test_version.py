import matplotlib
import numpy
import pandas
import torch
import torchvision
import tqdm

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("Matplotlib 版本:", matplotlib.__version__)
print("NumPy 版本:", numpy.__version__)
print("Pandas 版本:", pandas.__version__)
print("Torchvision 版本:", torchvision.__version__)
print("tqdm 版本:", tqdm.__version__)
