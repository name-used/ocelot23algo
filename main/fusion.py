from typing import List, Tuple
import torch
import torch.nn.functional as F


def correlated(
        coarse: torch.Tensor,               # (1, 1024, 1024)
        fine: torch.Tensor,                 # (1, 1024, 1024)
        classify: torch.Tensor,             # (2, 1024, 1024)
        divide: torch.Tensor,               # (2, 1024, 1024)
        device: str,
) -> List[Tuple[int, int, int, float]]:
    # 粗图 + 精图 -> 联合图
    combo: torch.Tensor = coarse * fine
    # 联合图转换至网络形式
    combo: torch.Tensor = combo[None, :, :, :]
    # 联合图 -> 最高点图
    pooled: torch.Tensor = F.max_pool2d(combo, 9, stride=1, padding=4)
    highest: torch.Tensor = (combo == pooled) & (combo > 0.2)
    # # 最高点图 -> 最高点九宫格图
    # highest_area: torch.Tensor = F.max_pool2d(highest.type(torch.float32), 3, stride=1, padding=1)
    # 最高点图 -> 最高点列
    _, _, ys, xs = torch.where(highest)

    # 类型图 -> 类型热图
    detect_cls = combo * classify[None, :, :, :]
    # divide_cls = combo * divide
    # 权重图
    weights = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float64, device=device) / 16
    # 类型热图 -> 概率叠加图
    for c in [0, 1]:
        detect_cls[None, :, c, :, :] = F.conv2d(
            input=detect_cls[None, :, c, :, :],
            weight=weights[None, None, :, :],
            bias=None,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )
    points = []
    for (x, y) in zip(xs, ys):
        prob_bc, prob_tc = detect_cls[0, :, y, x]
        c = int(prob_bc <= prob_tc)
        p = max(prob_bc, prob_tc) / (prob_bc + prob_tc)
        points.append(
            (
                int(x),
                int(y),
                int(c) + 1,
                float(p),
            )
        )
    return points
