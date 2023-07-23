from typing import List, Tuple
import torch
import torch.nn.functional as F


def correlated(
        coarse: torch.Tensor,               # (1024, 1024, 1)
        fine: torch.Tensor,                 # (1024, 1024, 1)
        classify: torch.Tensor,             # (1024, 1024, 2)
        divide: torch.Tensor,               # (1024, 1024, 2)
) -> List[Tuple[int, int, int, float]]:
    # 粗图 + 精图 -> 联合图
    combo: torch.Tensor = coarse * fine
    # 联合图转换至网络形式
    combo: torch.Tensor = combo[None, None, :, :, 0]
    # 联合图 -> 最高点图
    pooled: torch.Tensor = F.max_pool2d(combo, 9, stride=1, padding=4)
    highest: torch.Tensor = (combo == pooled) & (combo > 0)
    # # 最高点图 -> 最高点九宫格图
    # highest_area: torch.Tensor = F.max_pool2d(highest.type(torch.float32), 3, stride=1, padding=1)
    # 最高点图 -> 最高点列
    _, _, ys, xs = torch.where(highest)

    # 类型图 -> 类型热图
    detect_cls = combo * classify
    divide_cls = combo * divide
    # 类型热图 -> 概率叠加图
    detect_cls = F.conv2d(
        input=detect_cls,
        weight=torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16,
        bias=None,
        stride=1,
        padding="valid",
        dilation=1,
        groups=1,
    )
    points = []
    for (x, y) in zip(xs, ys):
        prob_bc, prob_tc = detect_cls[y, x, :]
        c = int(prob_bc <= prob_tc)
        p = max(prob_bc, prob_tc) / (prob_bc + prob_tc)
        points.append((x, y, c, p))
    return points
