import os

from PIL import Image
from torchvision import transforms

# 创建输出文件夹
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取一张示例图片
img = Image.open("pil.png")  # 替换为你的图片路径

# 创建两个不同的旋转
rotate_no_expand = transforms.RandomRotation(
    degrees=30, expand=False, fill=(0, 0, 0, 0)
)
rotate_expand = transforms.RandomRotation(degrees=30, expand=True, fill=(0, 0, 0, 0))

# 随机应用一次
rotated_no_expand = rotate_no_expand(img)
rotated_expand = rotate_expand(img)

# 保存原始图片
img.save(os.path.join(output_dir, "original.png"))

# 保存未扩展旋转的图片
rotated_no_expand.save(os.path.join(output_dir, "rotated_no_expand.png"))

# 保存扩展旋转的图片
rotated_expand.save(os.path.join(output_dir, "rotated_expand.png"))
