from adaptive_conv import adaConv2d, get_inference_time
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from Tadaptive_conv2 import adaTrConv2d
from segnet import SegNet


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__

    # for every Conv2d layer in a model..
    if classname.find('Conv2d') != -1:
        # get the number of the inputs
        n = m.out_channels * m.kernel_size[0] * m.kernel_size[1]

        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(1)

class adaModule(nn.Module):
    """
    paper module
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
    ):
        super(adaModule, self).__init__()

        self.conv = adaConv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.scales_conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales_conv.apply(weights_init_uniform_rule)
        self.scales_net = nn.Sequential(self.scales_conv,
                                        nn.ReLU())



    def forward(self, input: Tensor) -> Tensor:
        scales = self.scales_net(input)
        return self.conv(input, scales=scales)


class adaTrModule(nn.Module):
    """
    my experimental module
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
    ):
        super(adaTrModule, self).__init__()

        self.conv = adaTrConv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.scales_conv = nn.Conv2d(in_channels, out_channels, 3)
        self.scales_conv.apply(weights_init_uniform_rule)
        self.scales_net = nn.Sequential(self.scales_conv,
                                        nn.ReLU())



    def forward(self, input: Tensor) -> Tensor:
        scales = self.scales_net(input)
        return self.conv(input, scales=scales)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rand_t = torch.rand(5, 3, 7, 7).to(device)

    test_conv = adaModule(3, 64, kernel_size=3, dilation=1, padding=0, stride=1).to(device)
    print(get_inference_time(test_conv, device))