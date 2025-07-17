import math
import numpy as np
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)\

"""******************************************  MELON  ******************************************"""

class RepConv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

class MELON(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(MELON, self).__init__()

        self.c = int(c2 * e)
        self.mid = int(self.c * scale)

        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = DWConv(self.mid, self.mid, 3)

    def forward(self, x):
        y0 = self.cv0(x)
        y1 = self.cv1(x)
        y = [y0,y1]
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y0 = self.cv0(x)
        y1 = self.cv1(x)
        y = [y0,y1]
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))
"""******************************************  MELON  ******************************************"""

"""******************************************  FAMNet  ******************************************"""
class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class FAMNet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first_1 = Conv(c1, self.c, 3, 1)
        self.cv_first_2 = Conv(c1, self.c, 1, 1)
        self.cv_final = Conv((4 + 2*n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((1, 1), (1, 1)), e=1.0) for _ in range(n))
        self.t = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (1, 1)), e=1.0) for _ in range(n))

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y_f1 = self.cv_first_1(x)
        y_f2 = self.cv_first_2(x)
        y_f3 = [y_f1]
        y_f4 = [y_f2]
        y=[y_f1,y_f2]
        y = torch.cat(y,1)
        y0 = [self.cv_block_1(y)]
        y1 = [self.cv_block_2(y)]

        y_f3.extend(m(y_f3[-1]) for m in self.m)
        y_f4.extend(t(y_f4[-1]) for t in self.t)
        y = y0+y1+y_f3+y_f4

        return self.cv_final(torch.cat(y, 1))

"""******************************************  FAMNet  ******************************************"""

class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

"""******************************************  C4PMS  ******************************************"""

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PMSBlock(nn.Module):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.conv1 = DWConv(c, c, 3)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.conv2 = DWConv(2*c, c, 1)
        self.add = shortcut

    def forward(self, x):
        ylist = [self.conv1(x),self.attn(x)]
        y = torch.cat(ylist,1)
        z = x + self.conv2(y) if self.add else self.conv2(y)
        return z

class C4PMS(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 3, 1)
        self.cv2 = Conv(c1 + self.c, c1//2, 1)
        self.cv3 = Conv(self.c, c1//2, 1)

        self.m = nn.Sequential(*(PMSBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a0 = self.cv0(x)
        a1 = self.m(a0)
        a2 = torch.cat((a1, x), 1)
        a3 = self.cv2(a2)
        b0 = self.cv1(x)
        b1 = b0 + a0
        b2 = self.cv3(b1)
        return torch.cat((b2, a3), 1)

class FAMNet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first_1 = Conv(c1, self.c, 3, 1)
        self.cv_first_2 = Conv(c1, self.c, 1, 1)
        self.cv_final = Conv((4 + 2*n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((1, 1), (1, 1)), e=1.0) for _ in range(n))
        self.t = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (1, 1)), e=1.0) for _ in range(n))

        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1), Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y_f1 = self.cv_first_1(x)
        y_f2 = self.cv_first_2(x)
        y_f3 = [y_f1]
        y_f4 = [y_f2]
        y=[y_f1,y_f2]
        y = torch.cat(y,1)
        y0 = [self.cv_block_1(y)]
        y1 = [self.cv_block_2(y)]

        y_f3.extend(m(y_f3[-1]) for m in self.m)
        y_f4.extend(t(y_f4[-1]) for t in self.t)
        y = y0+y1+y_f3+y_f4

        return self.cv_final(torch.cat(y, 1))

"""******************************************  C4PMS  ******************************************"""










