import torch.nn as nn
import torch.nn.functional as F
from ..nn_model import NNModel
from ..nn_model_abc import AbstractNNModel
from ..nn_model_args import NNModelArgs


class SeparableConv2d(nn.Module):
    """
    实现深度可分离卷积：
      - 先进行 depth_wise 卷积 (groups=in_channels)，只在通道内做卷积
      - 再进行 point_wise 卷积 (1x1)，将通道数映射到 out_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_channels,
                                    bias=bias)
        self.point_wise = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=bias)
        return

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8993089&tag=1
class NNModel_ThinMobileNet(NNModel):
    """
    根据表中给出的结构（TABLE II）实现的示例网络：
      Input: (B, 3, 32, 32)
      1) Conv2d/s2 -> out: (B, 32, 16, 16)
      2) SeparableConv2d/s2 -> out: (B, 32, 8, 8)
      3) SeparableConv2d/s1 -> out: (B, 128, 8, 8)
      4) SeparableConv2d/s1 -> out: (B, 128, 8, 8)
      5) SeparableConv2d/s2 -> out: (B, 256, 4, 4)
      6) SeparableConv2d/s1 -> out: (B, 256, 4, 4)
      7) SeparableConv2d/s2 -> out: (B, 512, 2, 2)
      8) SeparableConv2d/s1 -> out: (B, 512, 2, 2)
      9) SeparableConv2d/s2 -> out: (B, 1024, 1, 1)
      10) Global average pool -> out: (B, 1024, 1, 1)
      11) FC & Softmax -> out: (B, num_classes)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = None
        self.sep_conv2 = None
        self.sep_conv3 = None
        self.sep_conv4 = None
        self.sep_conv5 = None
        self.sep_conv6 = None
        self.sep_conv7 = None
        self.sep_conv8 = None
        self.sep_conv9 = None
        self.avg_pool = None
        self.fc = None
        return

    # override
    def create_args(self):
        args = super().create_args()
        args.num_classes = 10
        return args

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        # 1) 普通卷积：stride=2, out_channels=32
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)

        # 2) SeparableConv2d/s2 -> out_channels=32
        self.sep_conv2 = SeparableConv2d(in_channels=32,
                                         out_channels=32,
                                         stride=2,  # s2
                                         padding=1,
                                         bias=False)

        # 3) SeparableConv2d/s1 -> out_channels=128
        self.sep_conv3 = SeparableConv2d(in_channels=32,
                                         out_channels=128,
                                         stride=1,
                                         padding=1,
                                         bias=False)

        # 4) SeparableConv2d/s1 -> out_channels=128
        self.sep_conv4 = SeparableConv2d(in_channels=128,
                                         out_channels=128,
                                         stride=1,
                                         padding=1,
                                         bias=False)

        # 5) SeparableConv2d/s2 -> out_channels=256
        self.sep_conv5 = SeparableConv2d(in_channels=128,
                                         out_channels=256,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        # 6) SeparableConv2d/s1 -> out_channels=256
        self.sep_conv6 = SeparableConv2d(in_channels=256,
                                         out_channels=256,
                                         stride=1,
                                         padding=1,
                                         bias=False)

        # 7) SeparableConv2d/s2 -> out_channels=512
        self.sep_conv7 = SeparableConv2d(in_channels=256,
                                         out_channels=512,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        # 8) SeparableConv2d/s1 -> out_channels=512
        self.sep_conv8 = SeparableConv2d(in_channels=512,
                                         out_channels=512,
                                         stride=1,
                                         padding=1,
                                         bias=False)

        # 9) SeparableConv2d/s2 -> out_channels=1024
        self.sep_conv9 = SeparableConv2d(in_channels=512,
                                         out_channels=1024,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        # 10) 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 11) 全连接层 (假设分类到 num_classes=10)
        self.fc = nn.Linear(in_features=1024,
                            out_features=args.num_classes,
                            bias=True)
        return self

    # override
    def forward(self, x):
        # 1) Conv2d/s2
        x = self.conv1(x)
        x = F.relu(x)

        # 2) SepConv/s2
        x = self.sep_conv2(x)
        x = F.relu(x)

        # 3) SepConv/s1
        x = self.sep_conv3(x)
        x = F.relu(x)

        # 4) SepConv/s1
        x = self.sep_conv4(x)
        x = F.relu(x)

        # 5) SepConv/s2
        x = self.sep_conv5(x)
        x = F.relu(x)

        # 6) SepConv/s1
        x = self.sep_conv6(x)
        x = F.relu(x)

        # 7) SepConv/s2
        x = self.sep_conv7(x)
        x = F.relu(x)

        # 8) SepConv/s1
        x = self.sep_conv8(x)
        x = F.relu(x)

        # 9) SepConv/s2
        x = self.sep_conv9(x)
        x = F.relu(x)

        # 10) Global AvgPool
        x = self.avg_pool(x)

        # Flatten for FC
        x = x.view(x.size(0), -1)

        # 11) FC
        x = self.fc(x)

        # 训练时可用 CrossEntropyLoss 直接将 logits 输入损失函数
        # 推断时可用 softmax
        return x
