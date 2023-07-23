from typing import Tuple, List
import cv2
import numpy as np
import torch

from .merge import Merger
from .fusion import correlated

detector = torch.jit.load('./weights/detector-jit.pth')
# divider = torch.jit.load('./weights/divider-jit.pth')


def detect(cell: np.ndarray[np.uint8], tissue: np.ndarray[np.uint8], offset: Tuple[int, int]) -> List[Tuple[int, int, int, float]]:
    # 基本参数
    width = 1024
    height = 1024
    kernel = 512
    step = 256
    vision = 256
    steep = 4
    device = 'cuda:0'

    # 基本操作
    detector.to(device)
    detector.eval()
    # divider.to(device)
    # divider.eval()

    # 首先执行目标检测任务，生成检测热图
    with torch.no_grad():
        # model_return_channels 用来告诉 merger 输出多少个结果，每个结果的 channel 是多少，这会决定 Merger 的融合参数控制
        merger = Merger(
                model_return_channels=(1, 1, 2),
                width=width,
                height=height,
                kernel_size=kernel,
                kernel_steep=steep,
                device=device,
        )
        # 遍历截图
        for y in range(0, height - kernel + 1, step):
            for x in range(0, width - kernel + 1, step):
                # 图像预处理
                image = cell[y: y + kernel, x: x + kernel, :] / 255
                image = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
                # 预测流程
                coarse_patch: torch.Tensor
                fine_patch: torch.Tensor
                classify_patch: torch.Tensor
                coarse_patch, fine_patch, classify_patch = detector(image)
                # 预测后处理
                coarse_patch = coarse_patch.softmax(dim=1)[:, 1:, :, :]
                fine_patch = fine_patch.clamp(0, 1)
                classify_patch = classify_patch.clamp(0, 1)
                # 融合
                merger.set(
                    patches_group=(coarse_patch, fine_patch, classify_patch),
                    grids=[(x, y)],
                )
        coarse_heat_map, fine_heat_map, classify_heat_map = merger.tail()

    # # 然后执行组织分割任务，生成分割热图
    # with torch.no_grad():
    #     # model_return_channels 用来告诉 merger 输出多少个结果，每个结果的 channel 是多少，这会决定 Merger 的融合参数控制
    #     merger = Merger(
    #             model_return_channels=(1,),
    #             width=width,
    #             height=height,
    #             kernel_size=kernel,
    #             kernel_steep=steep,
    #             device=device,
    #     )
    #     # 遍历截图
    #     for y in range(0, height - kernel + 1, step):
    #         for x in range(0, width - kernel + 1, step):
    #             # 图像预处理
    #             image = (tissue[y: y + kernel, x: x + kernel, :] / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #             image = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
    #             # 预测流程
    #             divide_heat_patch: torch.Tensor = detector(image)
    #             # 预测后处理
    #             divide_heat_patch = divide_heat_patch.permute(0, 2, 3, 1)
    #             # 融合
    #             merger.set(
    #                 patches_group=[(divide_heat_patch,)],
    #                 grids=[(x, y)]
    #             )
    #     divide_heat_map, = merger.tail()

    # 最后乱七八糟的融合方法各种上，并且返回结果
    with torch.no_grad():
        x, y = offset
        # divide_heat_map_cells_like = cv2.resize(divide_heat_map[y: y + vision, x: x + vision, :], dsize=(width, height))
        return correlated(
            coarse=coarse_heat_map,
            fine=fine_heat_map,
            classify=classify_heat_map,
            # divide=divide_heat_map_cells_like,
            divide=None,
            device=device,
        )
