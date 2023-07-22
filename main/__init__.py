from typing import Tuple, List
import cv2
import numpy as np
import torch

from .merger import Merger
from .fusion import correlated


detector = torch.jit.load('./weights/detector-jit.pth')
divider = torch.jit.load('./weights/divider-jit.pth')


def detect(cell: np.ndarray[np.uint8], tissue: np.ndarray[np.uint8], offset: Tuple[int, int]) -> List[Tuple[int, int, int, float]]:

    # 首先执行目标检测任务，生成检测热图
    # shapes 用来告诉 merger 输出多少个结果，每个结果的 channel 是多少，这会决定 Merger 的融合参数控制
    # lambdas 用来告诉 merger 这几个结果应该分别咋处理
    with Merger(model=detector, shapes=(2, 1, 2), lambdas=(
        lambda coarse_patch: coarse_patch.softmax(dim=1)[:, 1:, :, :],
        lambda fine_patch: fine_patch.clamp(0, 1),
        lambda classify_patch: classify_patch.clamp(0, 1),
    )) as merger:
        coarse_heat_map, fine_heat_map, classify_heat_map = merger.tail()

    # 然后执行组织分割任务，生成分割热图
    # shapes 用来告诉 merger 输出多少个结果，每个结果的 channel 是多少，这会决定 Merger 的融合参数控制
    # lambdas 用来告诉 merger 这几个结果应该分别咋处理
    with Merger(model=divider, shapes=(2,), lambdas=(
        lambda coarse_patch: coarse_patch,
    )) as merger:
        divide_heat_map = merger.tail()

    # 然后乱七八糟的融合方法各种上，并且返回结果
    x, y = offset
    divide_heat_map_cells_like = cv2.resize(divide_heat_map[y: y+256, x: x+256, :], dsize=(1024, 1024))
    return correlated(
        coarse=coarse_heat_map,
        fine=fine_heat_map,
        classify=classify_heat_map,
        divide=divide_heat_map_cells_like,
    )
