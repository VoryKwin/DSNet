import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.TransUNet_Part import PatchMerging, PatchExpand, FinalPatchExpand_X4, BasicLayer, BasicLayer_up, \
    PatchEmbed
from models.utils.MobileUNet_Part import InvertedResidualBlock
from models.utils.Bridge_Part import MtoT, TtoM


class DSNetSys(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1, embed_dim=96,
                 depths=[2, 2, 2, 2],  # encoder部分的SwinTransformer
                 depths_decoder=[1, 2, 2, 2],  # decoder部分的SwinTransformer
                 num_heads=[1, 2, 4, 8],  # 可能是指对应stage里都用相应的head数
                 window_size=7,  # 窗口尺寸
                 mlp_ratio=4.,  # 隐藏层维度与嵌入维度之间的比例
                 # 若ratio=4，则输入维度96 隐藏层维度384（96 * 4）输出维度：96
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 final_upsample="expand_first",
                 **kwargs):
        super().__init__()
        print(
            "DSNet expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths, depths_decoder, drop_path_rate, num_classes))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # stage4输出feature的channel
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # 96 * 2 【在哪用？好像这句没有用】
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches；将图像划分为非重叠的图像块
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding；绝对位置嵌入 默认无
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成一个列表，每层 SwinTransformer 使用的drop_path_rate
        # torch.linspace(start, end, step) 从0增长至drop_path_rate sum(depths)=8 dpr有8个结果
        # dpr = ([0.0000, 0.0143, 0.0286, 0.0429, 0.0571, 0.0714, 0.0857, 0.1000])  一个tensor
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # TransUNet--Encoding Part
        # (包含swin transformer block + 一个后面的PatchMerging)
        self.layers = nn.ModuleList()
        # i_layer依次为0，1，2，3；self.num_layers = 4
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               # depths=[2, 2, 2, 2],设i_layer = 0, depth[:1]=[], sum(depth[:1])=0
                               # depths[:1]=[2], sum(depths[:1])=2
                               # dpr[0:2],是取连两个组成一对tensor([0.0000, 0.0143])
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               # 只有前3个后面跟PatchMerging ， self.num_layers - 1 = 3
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # TransUNet--Decoding Part
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        # 循环4次
        for i_layer in range(self.num_layers):  # 是不是创建一个 self.num_delayers = len(depths_decoder)比较好
            # concat_linear 不清楚作用
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         # self.num_layers=4, 4-1-1=2，所以取值顺序2，1，0
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # norm
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # 最终输出
        if self.final_upsample == "expand_first":
            # 如果 self.final_upsample 的值为 "expand_first"，执行以下操作

            # 创建一个 FinalPatchExpand_X4 模块，用于将特征图上采样 4 倍。
            # input_resolution 参数表示输入特征图的分辨率，dim_scale=4 表示将特征维度放大 4 倍，dim=embed_dim 表示上采样后的特征维度。
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.pos_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.cos_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.sin_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.width_output = nn.Conv2d(in_channels=112, out_channels=self.num_classes, kernel_size=1,
                                          bias=False)

        # 调用 self._init_weights 方法，该方法用于初始化模型的权重。
        self.apply(self._init_weights)

        # ForMobileUNet--Encoding Part
        self.conv3x3 = self.depthwise_conv(in_chans, 32, p=1, s=2)
        self.irb_bottleneck1 = self.irb_bottleneck(32, 16, 1, 1, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(16, 24, 2, 2, 6)
        self.irb_bottleneck3 = self.irb_bottleneck(24, 32, 3, 2, 6)
        self.irb_bottleneck4 = self.irb_bottleneck(32, 96, 4, 2, 6)
        self.irb_bottleneck5 = self.irb_bottleneck(96, 1028, 3, 2, 6)
        # ForMobileUNet--Decoding Part
        self.D_irb1 = self.irb_bottleneck(1028, 96, 1, 2, 6, True)
        self.conv01 = nn.Conv2d(192, 96, 1)
        self.D_irb2 = self.irb_bottleneck(96, 32, 1, 2, 6, True)
        self.conv02 = nn.Conv2d(64, 32, 1)
        self.D_irb3 = self.irb_bottleneck(32, 24, 1, 2, 6, True)
        self.conv03 = nn.Conv2d(48, 24, 1)
        self.D_irb4 = self.irb_bottleneck(24, 16, 1, 2, 6, True)
        self.conv04 = nn.Conv2d(32, 16, 1)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)

        # ForBridge
        self.MtoT = MtoT()
        self.TtoM = TtoM()

    # ForMobileUNet
    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        """
        optimized convolution by combining depthwise convolution and
        pointwise convolution.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    # ForMobileUNet
    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        """
        create a series of inverted residual blocks.
        """
        convs = []
        # 第一个irb
        xx = InvertedResidualBlock(in_c, out_c, s, t, deconvolve=d)
        convs.append(xx)
        if n > 1:
            for i in range(1, n):
                # 之后重复的irb都加了skip connection
                xx = InvertedResidualBlock(out_c, out_c, 1, t, deconvolve=d)
                convs.append(xx)
        conv = nn.Sequential(*convs)
        return conv

    # ForMobileUNet
    def get_count(self, model):
        # simple function to get the count of parameters in a model.
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num

    def _init_weights(self, m):
        # 初始化权重函数，在初始化线性层（nn.Linear）和 LayerNorm 层（nn.LayerNorm）时进行不同的处理
        if isinstance(m, nn.Linear):
            # 对线性层进行截断正态分布初始化，标准差为 0.02
            nn.init.trunc_normal_(m.weight, std=0.02)
            # 如果线性层有偏置项（bias），将其初始化为常数 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 对 LayerNorm 层的偏置项和权重进行初始化，偏置项初始化为常数 0，权重初始化为常数 1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def up_x4(self, x, d5):  # x:([2, 3136, 96]), d5:([2, 16, 224, 224])
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # 最终的上采样方式为 "expand_first"，对特征图进行 4 倍上采样
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W  ([1, 96, 224, 224])
            # 两路结果concatenate
            x = torch.cat((x, d5), dim=1)  # ([1, 112, 224, 224])
            # 通过四个卷积层分别输出预测的抓取位置、余弦角、正弦角和抓取宽度
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def forward(self, x):
        # x, x_downsample = self.forward_features(x)
        # x = self.forward_up_features(x, x_downsample, x6, x5, x4, x3)
        # pos_output, cos_output, sin_output, width_output = self.up_x4(x)
        # return pos_output, cos_output, sin_output, width_output

        # MobileUNet 前半段 为了获得x6
        x1 = self.conv3x3(x)  # (32, 112, 112)
        x2 = self.irb_bottleneck1(x1)  # (16,112,112)
        x3 = self.irb_bottleneck2(x2)  # (24,56,56)
        x4 = self.irb_bottleneck3(x3)  # (32,28,28)
        x5 = self.irb_bottleneck4(x4)  # (96,14,14)
        x6 = self.irb_bottleneck5(x5)  # (1028,7,7)
        x = self.patch_embed(x)
        # x6: torch.Size([2, 1028, 7, 7])
        # x: torch.Size([2, 3136, 48])

        # 是否使用绝对位置编码
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.MtoT(x6, x)
        # x = self.pos_drop(x) # droprate默认为0
        x_downsample = []
        # 通过多层 Transformer Encoder 进行特征提取，将中间层的输出保存在 x_downsample 中
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        # 对输出特征进行 LayerNorm 归一化，得到 B × L × C 的输出特征
        x = self.norm(x)  # B L C

        d1 = torch.cat((self.D_irb1(x6), x5), dim=1)
        d1 = self.conv01(d1)
        d2 = torch.cat((self.D_irb2(d1), x4), dim=1)
        d2 = self.conv02(d2)
        d3 = torch.cat((self.D_irb3(d2), x3), dim=1)
        d3 = self.conv03(d3)
        x_for_MobileUNet = self.TtoM(d3, x)
        d4 = torch.cat((self.D_irb4(x_for_MobileUNet), x2), dim=1)
        d4 = self.conv04(d4)
        d5 = self.DConv4x4(d4)  # d5: torch.Size([1, 3, 224, 224])

        # Dencoder 阶段，通过多层 Transformer Decoder 进行特征融合和上采样
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # 在每一层中，将当前层的特征与 x_downsample 中对应层的特征进行拼接
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        # 对输出特征进行 LayerNorm 归一化，得到 B × L × C 的输出特征
        x = self.norm_up(x)  # B L C
        pos_output, cos_output, sin_output, width_output = self.up_x4(x, d5)
        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        # 计算损失
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        # p_loss = F.mse_loss(pos_pred, y_pos)
        # cos_loss = F.mse_loss(cos_pred, y_cos)
        # sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)
        # F.smooth_l1_loss
        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def flops(self):
        # 计算模型的 FLOPs（浮点运算数）
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (
                2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
