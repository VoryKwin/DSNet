import torch
from torch import nn


# inputs: x(b c h w) z(b m d)
#         x(1, 1028, 7, 7), z(1, 3136, 96)
# output: z(b m d)
#         z(1, 3136, 96)
class MtoT(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.z_to_q = nn.Linear(96, 1028)
        self.attend = nn.Softmax(dim=-1)  # 在最后一个维度上计算 softmax
        self.scale = 1028 ** -0.5
        self.final_adjust = nn.Linear(1028, 96)

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        # x:(1,1,49,1028)
        q = self.z_to_q(z)  # (1,3136,1028)
        q = q.view(b, 1, 3136, 1028)
        # 此时 q:(1,1,3136,1028) x.transpose(2, 3):(1,1,1028,49)
        dots = q @ x.transpose(2, 3) * self.scale  # ([1, 1, 3136, 49])
        attn = self.attend(dots)  # ([1, 1, 3136, 49])
        # x:(1,1,49,1028)
        out = attn @ x  # ([1, 1, 3136, 1028])
        out = out.squeeze(1)  # 去掉第二个维度，即变为 ([1, 3136, 1028])
        out = self.final_adjust(out)  # ([1, 3136, 96])
        # print((z + out).shape)
        return z + out  # ([1, 3136, 96])


# inputs: x(b c h w) z(b m d)
#         x(1, 24, 56, 56),z(1, 49, 768)
# output: x(b c h w)
#         x(1, 24, 56, 56)
class TtoM(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.z_to_k = nn.Linear(768, 24)
        self.z_to_v = nn.Linear(768, 24)
        self.attend = nn.Softmax(dim=-1)
        self.scale = 24 ** -0.5

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)  # ([1, 1, 3136, 24])
        k = self.z_to_k(z).view(b, 1, 49, 24)  # ([1, 1, 49, 24])
        v = self.z_to_v(z).view(b, 1, 49, 24)  # ([1, 1, 49, 24])
        dots = q @ k.transpose(2, 3) * self.scale  # ([1, 1, 3136, 49])
        attn = self.attend(dots)  # ([1, 1, 3136, 49])
        out = attn @ v  # ([1, 1, 3136, 24])
        # print('out1', out.shape)
        out = out.squeeze(1)  # ([1, 3136, 24])
        # print('out2', out.shape)
        out = out.view(b, 24, 56, 56)  # ([b, 24, 56, 56])
        # print('out3', out.shape)
        return x + out
