import os  # 处理文件和目录操作
import glob  # 匹配文件路径名
import random

from utils.data import get_dataset
from utils.data.grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


# 父类定义在grasp_data.py里
class CornellDataset(GraspDatasetBase):
    """
    Cornell数据集包装器 Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, ds_zoom=0, ds_shuffle=False, **kwargs):
        """
        :param file_path: 数据集路径
        :param start: 若要划分数据集，选定开始位置，[0,1]，使用这之后的数据，若0.5则使用后50%；
        :param end: 若要划分数据集，选定结束位置；
        :param ds_rotate: 若要划分数据集，选定旋转系数，[0,1]，若0.8则前面80%的数据移动至末尾处
        :param kwargs: kwargs for GraspDatasetBase
        """
        # (**kwargs)接受所有关键字参数传递给父类
        super(CornellDataset, self).__init__(**kwargs)
        self.length = len(self.grasp_files)
        # 返回一个列表，每个cpos.txt文件的路径
        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        # print(graspf[0:9])

        # 是否打乱
        if ds_shuffle:
            print('!!!!!!!!!进入IW规则')
            random.seed(2022)
            random.shuffle(graspf)
            # print(graspf)
        elif not ds_shuffle:
            print('!!!!!!!!!进入OW规则')
            graspf.sort()

        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        if ds_rotate:
            # 距离l = 10,ds_rotate=0.2,10*0.2=2,把2至结束和开始至2重新拼接赋值给graspf
            graspf = graspf[int(l * ds_rotate):] + graspf[:int(l * ds_rotate)]

        # 返回每个d.tiff文件的路径0
        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        # 默认使用全部数据
        self.grasp_files = graspf[int(l * start):int(l * end)]  # 全是 cpos.txt路径 的一个list
        self.depth_files = depthf[int(l * start):int(l * end)]  # 全是 d.tiff  路径 的一个list
        self.rgb_files = rgbf[int(l * start):int(l * end)]  # 全是 r.png   路径 的一个list

    def _get_crop_attrs(self, idx):
        # 在确保裁剪区域不超出图像的左右边界的前提下，以抓取框的中心为中心进行裁剪，返回center, left, top（以左上角为起点裁切224*224）
        # 用self.grasp_files[idx]这个cpos.txt文件，实例化一个GraspRectangle类的对象，里面是一组一组的四点坐标
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        # 从抓取框信息中获取抓取框的中心坐标
        center = gtbbs.center  # (cy, cx)
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)  # 中心旋转
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)  # 缩放
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
