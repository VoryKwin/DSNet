import os
import glob

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image

class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        # (**kwargs)接受所有关键字参数传递给父类
        super(CornellDataset, self).__init__(**kwargs)
        self.length = len(self.grasp_files)
        # 返回一个列表，每个cpos.txt文件的路径
        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        # print(graspf[0:9])
        graspf.sort()
        # 是否打乱
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
        # gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        # center = gtbbs.center
        # left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        # top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        # return center, left, top
        return 0
    #
    def get_gtbb(self, idx, rot=0, zoom=1.0):
    #     gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
    #     center, left, top = self._get_crop_attrs(idx)
    #     gtbbs.rotate(rot, center)
    #     gtbbs.offset((-top, -left))
    #     gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
    #     return gtbbs
        return 0

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # depth_img.rotate(rot, center)
        # depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
