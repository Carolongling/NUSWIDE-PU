# NUSWIDE-PU

处理后的NUS-WIDE源数据集：取前26个类的image、tag、label做使用
文件名：FLICKR-25K.mat
链接：https://pan.baidu.com/s/1freqirqJqvWhzQuPZ6g2FA 提取码: ii10

1.下载FLICKR-25K.mat
2.读取数据，终端运行python load_data.py，会对image和tag，label数据进行序列化，会生成image，tag，label的序列文件
3.运行主文件：python NUSWIDE_SPY.py

图像预训练模型对应文件：vgg.py
文本预训练模型对应文件：nn.py
spy文件：pu_learning.py
主运行文件：NUSWIDE_SPY.py

公司服务器：172.16.0.163
已配置好的环境：conda activate VFL_SPY
文件位置：/home/centos/zll/VFL_SPY
