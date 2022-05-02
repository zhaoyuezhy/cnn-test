from torch import nn
import torch
import torch.nn.functional as F

'''
    说明:
    1.LeNet是5层网络
    2.nn.ReLU(inplace=True)  参数为True是为了从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
    3.本模型的维度注释均省略了N(batch_size)的大小,即input(3, 32, 32)-->input(N, 3, 32, 32)
    4.nn.init.xavier_uniform_(m.weight)
     用一个均匀分布生成值,填充输入的张量或变量,结果张量中的值采样自U(-a, a)，
     其中a = gain * sqrt( 2/(fan_in + fan_out))* sqrt(3),
     gain是可选的缩放因子,默认为1
     'fan_in'保留前向传播时权值方差的量级,'fan_out'保留反向传播时的量级
    5.nn.init.constant_(m.bias, 0)
      为所有维度tensor填充一个常量0
'''


class LeNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)  # input(3, 32, 32)  output(16, 28, 28)
        x = self.relu(x)  # 激活函数
        x = self.maxpool1(x)  # output(16, 14, 14)
        x = self.conv2(x)  # output(32, 10, 10)
        x = self.relu(x)  # 激活函数
        x = self.maxpool2(x)  # output(32, 5, 5)
        x = torch.flatten(x, start_dim=1)  # output(32*5*5) N代表batch_size
        x = self.fc1(x)  # output(120)
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # output(84)
        x = self.relu(x)  # 激活函数
        x = self.fc3(x)  # output(num_classes)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

