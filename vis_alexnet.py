from curses import mouseinterval
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and testing datasets.
pic_dir = '/home/yzhao/anaconda3/bin/test_all/sunshine.jpg'

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()


# 单张图像送入
# 构建网络
# 提取中间层
# 可视化特征图

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)


def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (224, 224))
    skimage.io.imsave('new4.jpg', img256)

    img = img256.copy()
    ax = plt.subplot()
    ax.set_title('new-image')
    # ax.axis('off')
    plt.imshow(img)

    plt.show()


# 整个AlexNet的 input_size=(3*227*227); output_size=(192*6*6).
class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=0),# input_size=(3*227*227); output_size=(48*55*55).
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),# input_size=(48*55*55); output_size=(48*27*27).
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),# input_size=(48*27*27); output_size=(128*27*27).
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),# input_size=(128*27*27); output_size=(128*13*13).
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),# input_size=(128*13*13); output_size=(192*13*13).
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),# input_size=(192*13*13); output_size=(192*13*13).
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),# input_size=(192*13*13); output_size=(128*13*13).
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),# input_size=(128*13*13); output_size=(192*6*6).
        )

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x=torch.flatten(x,start_dim=1)
 
        return x


# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        # 打印alexnet模型
        print('start---------\n',self.submodule._modules.items())
        print('---------over')
        
        # name是,conv1~conv5;
        # module是,conv1~conv5对应的层.
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            print(module)
            
            x = module(x)
            print('name', name)
            
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def get_feature():  # 特征可视化
    # 输入数据
    img = get_picture(pic_dir, transform) # 输入的图像是【3,256,256】
    # 插入维度,插入的这个是batchsize
    img = img.unsqueeze(0)  # 【1,3,256,256】
    img = img.to(device)

    # 特征输出
    net = AlexNet().to(device)
    # net.load_state_dict(torch.load('./model/net_050.pth'))
    exact_list = ['conv2']
    # myexactor输出的话,输出的是一整个Alexnet
    myexactor = FeatureExtractor(net, exact_list)  # 输出是一个网络
    
    x = myexactor(img)


    # 特征输出可视化
    for i in range(32):  # 可视化了32通道
        ax = plt.subplot(6, 6, i + 1) # 画图，(行数, 列数, i+1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title('new—conv5-image')

        # cmap是颜色图谱，jet是：蓝-青-黄-红；
        # impshow(X,cmap)的X是：要绘制的图像或数组。
        plt.imshow(x[0].data.cpu().numpy()[0,i,:,:],cmap='jet')

    plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型

# 训练
if __name__ == "__main__":
    #get_picture_rgb(pic_dir)
    get_feature()
    
# if __name__ == "__main__":
#     AlexNet1=AlexNet()
#     x = torch.randn(1, 3, 224, 224)
#     out = AlexNet1(x)