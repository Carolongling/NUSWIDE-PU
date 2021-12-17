import os
import numpy as np
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
# import torch.cuda
import torchvision.transforms as transforms
from load_data import load_object_from_zip
from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda(3)  # 将模型从CPU发送到GPU,如果没有GPU则删除该行，指定显卡3
    return model


# 特征提取
def extract_feature(model, image):
    result_npy = []
    model.eval()  # 必须要有，不然会影响特征提取结果
    # image = image.cuda(3)  # 如果只是在cpu上跑的话要将这行去掉
    result = model(Variable(image))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    tmp1 = torch.tensor(result_npy)
    image_feature = tmp1.view(-1, 512 * 14 * 14)  # 出来的是二维 几张图片 另一个是具体的内容
    return image_feature
