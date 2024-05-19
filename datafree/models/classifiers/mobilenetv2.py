from torch import nn
from torch import Tensor
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, List
# import torch.nn as nn
from collections import OrderedDict


__all__ = ['MobileNetV2', 'mobilenet_v2']


# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }


# def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# class ConvBNActivation(nn.Sequential):
#     def __init__(
#         self,
#         in_planes: int,
#         out_planes: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#         activation_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             activation_layer = nn.ReLU6
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             norm_layer(out_planes),
#             activation_layer(inplace=True)
#         )


# # necessary for backwards compatibility
# ConvBNReLU = ConvBNActivation


# class InvertedResidual(nn.Module):
#     def __init__(
#         self,
#         inp: int,
#         oup: int,
#         stride: int,
#         expand_ratio: int,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers: List[nn.Module] = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)

#     def forward(self, x: Tensor) -> Tensor:
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)


# class MobileNetV2(nn.Module):
#     def __init__(
#         self,
#         num_classes: int = 1000,
#         width_mult: float = 1.0,
#         inverted_residual_setting: Optional[List[List[int]]] = None,
#         round_nearest: int = 8,
#         block: Optional[Callable[..., nn.Module]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         """
#         MobileNet V2 main class
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#             block: Module specifying inverted residual building block for mobilenet
#             norm_layer: Module specifying the normalization layer to use
#         """
#         super(MobileNetV2, self).__init__()

#         if block is None:
#             block = InvertedResidual

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         input_channel = 32
#         last_channel = 1280

#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]

#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))

#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)

#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )

#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # This exists since TorchScript doesn't support inheritance, so the superclass method
#         # (this one) needs to have a name other than `forward` that can be accessed in a subclass
#         x = self.features(x)
#         # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
#         x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
#         x = self.classifier(x)
#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)

#把channel变为8的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
 
 
#定义基本的ConvBN+Relu
class baseConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,groups=1,stride=1):
        super(baseConv, self).__init__()
        pad=kernel_size//2
        relu=nn.ReLU6(inplace=True)
        if kernel_size==1 and in_channels>out_channels:
            relu=nn.Identity()
        self.baseConv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=pad,groups=groups,bias=False),
            nn.BatchNorm2d(out_channels),
            relu
        )
 
    def forward(self,x):
        out=self.baseConv(x)
        return out
 
 
#定义残差结构
class residual(nn.Module):
    def __init__(self,in_channels,expand_rate,out_channels,stride):         #输入和输出channel都要调整到8的整数倍
        super(residual, self).__init__()
        expand_channel=int(expand_rate*in_channels)     #升维后的channel
 
        conv1=baseConv(in_channels, expand_channel, 1, stride=stride)
        if expand_rate==1:
            #此时没有1*1卷积升维
            conv1=nn.Identity()
 
        #channel1
        self.block1=nn.Sequential(
            conv1,
            baseConv(expand_channel,expand_channel,3,groups=expand_channel,stride=stride),
            baseConv(expand_channel,out_channels,1)
        )
 
        if stride==1 and in_channels==out_channels:
            self.has_res=True
        else:
            self.has_res=False
 
    def forward(self,x):
        if self.has_res:
            return self.block1(x)+x
        else:
            return self.block1(x)
 
 
#定义mobilenetv2
class MobileNet_v2(nn.Module):
    def __init__(self,theta=1,num_classes=10,init_weight=True):
        super(MobileNet_v2, self).__init__()
        #[inchannel,t,out_channel,stride]
        net_config=[[32,1,16,1],
                    [16,6,24,2],
                    [24,6,32,2],
                    [32,6,64,2],
                    [64,6,96,1],
                    [96,6,160,2],
                    [160,6,320,1]]
        repeat_num=[1,2,3,4,3,3,1]
 
        module_dic=OrderedDict()
 
        module_dic.update({'first_Conv':baseConv(3,_make_divisible(theta*32),3,stride=2)})
 
        for idx,num in enumerate(repeat_num):
            parse=net_config[idx]
            for i in range(num):
                module_dic.update({'bottleneck{}_{}'.format(idx,i+1):residual(_make_divisible(parse[0]*theta),parse[1],_make_divisible(parse[2]*theta),parse[3])})
                parse[0]=parse[2]
                parse[-1]=1
 
        module_dic.update({'follow_Conv':baseConv(_make_divisible(theta*parse[-2]),_make_divisible(1280*theta),1)})
        module_dic.update({'avg_pool':nn.AdaptiveAvgPool2d(1)})
 
        self.module=nn.Sequential(module_dic)
 
        self.linear=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(_make_divisible(theta*1280),num_classes)
        )
        #初始化权重
        if init_weight:
            self.init_weight()
 
    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)
 
 
    def forward(self,x):
        out=self.module(x)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

def mobilenet_v2(num_classes,pretrained=False):
    model=MobileNet_v2(num_classes=num_classes)
    return model
# def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
#     """
#     Constructs a MobileNetV2 architecture from
#     `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = MobileNetV2(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model