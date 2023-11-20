import numpy as np
import torch
import torch.utils.data
import random


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    一个抽象的用于抓取数据集的类，作为具体数据集类的父类使用
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: 图像的输出尺寸(square)
        :param include_depth: 是否包含深度
        :param include_rgb: 是否包含RGB
        :param random_rotate: 是否使用随机旋转 random rotations
        :param random_zoom: 是否使用随机缩放 random zooms
        :param input_only: 是否只返回不带标签(no labels)的数据
        """
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')
        # 保证至少要有一个输入信息

    @staticmethod  # 表示该方法是一个静态方法
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            # s.shape求numpy数组s的行列，len判断是不是二维数组（矩阵），也可以用 s.ndim
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
            # np.expand_dims(s, 0) 将矩阵 s 在维度 0 上进行扩展，在最前面添加一个新的维度，使其变为 3 维数组。
            # astype(np.float32) 转换为 np.float32
            # torch.from_numpy() 将 NumPy 数组转换为 PyTorch 张量，得到一个 3 维的张量。
        else:
            return torch.from_numpy(s.astype(np.float32))
            # astype(np.float32) 转换为 np.float32
            # torch.from_numpy() 将 NumPy 数组转换为 PyTorch 张量，得到一个与输入数组维度相同的张量。

    # 定义了三个抽象方法，强制子类必须实现这些方法，否则触发 NotImplementedError 异常
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    # 类如果定义了 __getitem__ 方法，那么该类的实例对象就可以像列表（list）一样使用索引访问 obj[idx]
    def __getitem__(self, idx):
        # 如果使用随机选抓
        if self.random_rotate:
            # 随机选择一个角度作赋值rot [0，90°，180°，270°]
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        # 如果使用随机缩放
        if self.random_zoom:
            # 随机选择一个缩放值赋值zoom_factor(0.5,1.0)
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # 加载深度图
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # 加载RGB图
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # 加载抓取框
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # draw 方法用于将抓取框绘制成图像
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        """
        对width_img宽度图像单独进行归一化处理。
        np.clip 将宽度图像中的像素值限制在 [0.0, 150.0] 的范围内;
        然后将图像中的所有像素值除以 150.0 进行归一化，将宽度值缩放到 [0.0, 1.0] 的范围内。
        """
        width_img = np.clip(width_img, 0.0, 150.0) / 150.0

        # 如果同时使用深度和RGB
        if self.include_depth and self.include_rgb:
            # 将拓展维度后的深度图和RGB图，在axis=0（理解为外侧第一个维度也就是channel维度）
            x = self.numpy_to_torch(np.concatenate((np.expand_dims(depth_img, 0), rgb_img), 0))
        # 只使用depth
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        # 只使用rgb
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        # 数据集中物体抓取框的数量，然后将该数量作为数据集的大小返回
        return len(self.grasp_files)
