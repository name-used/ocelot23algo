from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    用一系列脑洞大开的卷积设定 point_detect 和 class
    """

    # 粗图 + 精图 -> 联合图
    combo: torch.Tensor = coarse * fine
    # 联合图转换至网络形式
    combo: torch.Tensor = combo[None, :, :, :]

    # 准备一个高斯核（对卷积归一化）
    kernel = gaussian_kernel(size=9, steep=4, device=device)
    kernel = kernel / kernel.sum()
    # 先乘一个热图出来
    heat: torch.Tensor = combo.clone()
    # 然后池化
    pooled: torch.Tensor = F.max_pool2d(heat, 9, stride=1, padding=4)
    # 迭代几次
    for _ in range(5):
        # 获得最高顶点
        highest = (heat == pooled).type(torch.float32)
        # 最高点 -> 最高平台
        area = F.max_pool2d(highest, 9, stride=1, padding=4)
        # 只保留平台内的概率
        heat *= area
        # 高斯模糊
        heat = torch.conv2d(heat, kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        # 再池化
        pooled = F.max_pool2d(heat, 9, stride=1, padding=4)

    # 最高点图 -> 最高点列
    highest: torch.Tensor = (heat == pooled) & (combo > 0.3)
    _, _, ys, xs = torch.where(highest)

    # from .show import PPlot
    # import cv2
    # point_map = np.zeros(shape=image.shape, dtype=np.uint8)
    # for x, y in zip(xs, ys):
    #     cv2.circle(point_map, (int(x), int(y)), 3, [255, 255, 0], 3)
    # PPlot().add(
    #     image,
    #     classify[0, ...].cpu().numpy(),
    #     divide[0, ...].cpu().numpy(),
    #     point_map,
    # ).show()

    # 类型图 -> 类型热图
    detect_cls = combo * classify[None, :, :, :] * divide[None, :, :, :]
    # divide_cls = combo * divide
    # 权重图
    # kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float64, device=device) / 16
    kernel = gaussian_kernel(size=9, steep=3, device=device)
    kernel = kernel / kernel.sum()
    # 类型热图 -> 概率叠加图
    for c in [0, 1]:
        detect_cls[None, :, c, :, :] = F.conv2d(
            input=detect_cls[None, :, c, :, :],
            weight=kernel,
            bias=None,
            stride=1,
            padding=4,
            dilation=1,
            groups=1,
        )
    points = []
    for (x, y) in zip(xs, ys):
        prob_bc, prob_tc = detect_cls[0, :, y, x]
        c = int(prob_bc <= prob_tc) + 1
        # p = max(prob_bc, prob_tc) / (prob_bc + prob_tc)
        p = combo[0, 0, y, x]
        points.append(
            (
                int(x),
                int(y),
                int(c),
                float(p),
            )
        )
    return points
