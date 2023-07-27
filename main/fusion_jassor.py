from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.draw import disk as sk_disk

from .utils import gaussian_kernel


def correlated(
        coarse: torch.Tensor,  # (1, 1024, 1024)
        fine: torch.Tensor,  # (1, 1024, 1024)
        classify: torch.Tensor,  # (2, 1024, 1024)
        divide: torch.Tensor,  # (2, 1024, 1024)
        device: str,
        image: np.ndarray = None,
) -> List[Tuple[int, int, int, float]]:
    """
    根据 detect 设定 class
    """
    thresh = 0.30

    # 粗图 + 精图 -> 联合图
    combo: torch.Tensor = coarse * fine
    # 联合图转换至网络形式
    combo: torch.Tensor = combo[None, :, :, :]
    # 类型图 -> 类型热图
    classify = combo * classify[None, :, :, :]

    # 范围截断
    combo = combo * heatmap_nms(combo, device=device)

    # 获得点列
    points = get_pts_from_hm(combo, thresh, device=device)

    # 获得类型列
    classes = get_cls_pts_from_hm(points, classify, device=device)

    # 获得概率列
    possibility = [combo[0, 0, y, x] for x, y in points]

    points = [(int(x), int(y), int(c), float(p)) for (x, y), c, p in zip(points, classes, possibility)]

    return points


def heatmap_nms(combo: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    kernel = gaussian_kernel(size=17, steep=3, device=device)[None, None, :, :]  # 初始值 9
    kernel = kernel / kernel.sum()

    heat = torch.conv2d(combo, kernel, bias=None, stride=1, padding=8, dilation=1, groups=1)

    pooled = torch.nn.functional.max_pool2d(heat, 9, stride=1, padding=4)
    # h找到0和最大值的点为1
    highest = (heat >= pooled) * (combo > 0.)
    # 将h最大值的点膨胀
    # 知道了，这个的问题在于距离不稳定
    # area = torch.nn.functional.max_pool2d(highest * 1., 3, stride=1, padding=1)
    # 原代码使用以下 kernel 迭代扩张两次
    # torch 里用这玩意不太方便，但考虑到 max 约等于 lim p -> 无穷 p 范数
    # 故这里直接使用一个高次范数代替 max
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=torch.float32, device=device)[None, None, :, :]
    area = torch.conv2d((highest * 1.) ** 10, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1) ** 0.1
    area = torch.conv2d(area ** 10, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1) ** 0.1
    return area


def get_pts_from_hm(combo: torch.Tensor, prob: float = 0.5, device: str = 'cpu') -> List[Tuple[int, int]]:
    kernel = gaussian_kernel(size=17, steep=2, device=device)[None, None, :, :]  # 初始值 9
    kernel = kernel / kernel.sum()

    heat = torch.conv2d(combo, kernel, bias=None, stride=1, padding=8, dilation=1, groups=1)
    heat = torch.conv2d(heat, kernel, bias=None, stride=1, padding=8, dilation=1, groups=1)
    heat = torch.conv2d(heat, kernel, bias=None, stride=1, padding=8, dilation=1, groups=1)
    pooled = torch.nn.functional.max_pool2d(heat, 9, stride=1, padding=4)
    # 最高点图 -> 最高点列
    highest: torch.Tensor = (heat >= pooled) & (combo >= prob)
    _, _, ys, xs = torch.where(highest)
    return list(zip(xs, ys))


def get_cls_pts_from_hm(points: List[Tuple[int, int]], classify: torch.Tensor, device: str = 'cpu') -> List[int]:
    # 权重图
    # kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float64, device=device) / 16
    kernel = gaussian_kernel(size=9, steep=3, device=device)[None, None, :, :]
    kernel = kernel / kernel.sum()
    # kernel = torch.concat([kernel, kernel], dim=1)
    bc = torch.conv2d(classify[:, :1, :, :], kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)[0, 0, :, :]
    tc = torch.conv2d(classify[:, 1:, :, :], kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)[0, 0, :, :]
    # 类型热图 -> 概率叠加图
    classes = []
    for (x, y) in points:
        c = int(bc[y, x] <= tc[y, x]) + 1
        classes.append(int(c))
    return classes
