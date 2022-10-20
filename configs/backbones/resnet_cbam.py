import torch
import torch.nn as nn


from .resnet import ResNet


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNetCBAM(ResNet):

    """ResNet backbone + CBAM.
    https://github.com/luuuyi/CBAM.PyTorch
    https://www.sciencedirect.com/science/article/pii/S0040816621001932
    （对这个主干网络的简短描述）

    Args:
        depth(int): Network depth, from {18, 34, 50, 101, 152}.
        ...
        （参数文档）
    """

    def __init__(self, **kwargs):
        # 调用基类 ResNet 的初始化函数
        super(ResNetCBAM, self).__init__(**kwargs)

        # 其他特殊的初始化流程
        if self.depth == 18 or self.depth == 34:
            # for stage 0
            self.ca_0 = ChannelAttention(64)
            self.sa_0 = SpatialAttention()
            # for stage 1
            self.ca_1 = ChannelAttention(128)
            self.sa_1 = SpatialAttention()
            # for stage 2
            self.ca_2 = ChannelAttention(256)
            self.sa_2 = SpatialAttention()
            # for stage 3
            self.ca_3 = ChannelAttention(512)
            self.sa_3 = SpatialAttention()
        elif self.depth == 50 or self.depth == 101 or self.depth == 152:
            # for stage 0
            self.ca_0 = ChannelAttention(256)
            self.sa_0 = SpatialAttention()
            # for stage 1
            self.ca_1 = ChannelAttention(512)
            self.sa_1 = SpatialAttention()
            # for stage 2
            self.ca_2 = ChannelAttention(1024)
            self.sa_2 = SpatialAttention()
            # for stage 3
            self.ca_3 = ChannelAttention(2048)
            self.sa_3 = SpatialAttention()

    def forward(self, x):
        """
        重写 forward()
        """
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            ca = getattr(self, f'ca_{i}')
            sa = getattr(self, f'sa_{i}')
            x = res_layer(x)
            x = ca(x) * x
            x = sa(x) * x
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
