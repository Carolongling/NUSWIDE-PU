# 从预训练好的 BERT 模型中提取特征向量，即 Feature Extraction 方法
import torch.nn
from load_data import load_object_from_zip
from torch.autograd import Variable
LAYER1_NODE = 8192

def txt_net_strucuture(text_input, dimy):
    fc1W = Variable(torch.randn(1, dimy, 1, LAYER1_NODE) * 0.01, requires_grad=True)
    fc1b = Variable(torch.randn(1, LAYER1_NODE) * 0.01, requires_grad=True)
    conv1 = torch.nn.Conv2d(text_input, fc1W, kernel_size = 2, stride = [1,1,1,1], padding='VALID')
    output = torch.nn.ReLU(conv1+torch.squeeze(fc1b))
    return output

if __name__ == "__main__":
    Tags = load_object_from_zip('tags')
    ydim = Tags.shape[1]
    print(ydim)
    tag = Tags[1]  # 先试一试用第一张图
    # img1 = img.imtranspose(0, 3, 2, 1)
    # tag1 = torch.from_numpy(tag).float()
    tmp = txt_net_strucuture(tag,ydim)
    print(tmp.shape)