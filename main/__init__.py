import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

from config import output_root
from .merge import Merger
from .fusion_liner_gauss import correlated    # *
# from .fusion_liner import correlated
# from .fusion_weighter import correlated
# from .fusion_wentai import correlated
# from .fusion_jassor import correlated
# from .fusion_xuge import correlated


detector = torch.jit.load('./weights/detector-jit.pth')
divider = torch.jit.load('./weights/divider-jit.pth')


def detect(cell: np.ndarray[np.uint8], tissue: np.ndarray[np.uint8], offset: Tuple[int, int], cache_code: str = None) -> List[Tuple[int, int, int, float]]:
    # 基本参数
    width = 1024
    height = 1024
    kernel = 512
    step = 256
    vision = 256
    steep = 3
    device = 'cuda:1'

    # 基本操作
    detector.to(device)
    detector.eval()
    divider.to(device)
    divider.eval()

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
        # 开发时有缓存用缓存，实际跑的时候没这东西
        if cache_code and \
                os.path.exists(rf'{output_root}/predict/{cache_code}_coarse.weight') and \
                os.path.exists(rf'{output_root}/predict/{cache_code}_fine.weight') and \
                os.path.exists(rf'{output_root}/predict/{cache_code}_classify.weight'):
            coarse_heat_map = torch.load(rf'{output_root}/predict/{cache_code}_coarse.weight', map_location=device)
            fine_heat_map = torch.load(rf'{output_root}/predict/{cache_code}_fine.weight', map_location=device)
            classify_heat_map = torch.load(rf'{output_root}/predict/{cache_code}_classify.weight', map_location=device)
        else:
            # 遍历截图
            patches = []
            grids = []
            for y in range(0, height - kernel + 1, step):
                for x in range(0, width - kernel + 1, step):
                    # 图像预处理
                    image = cell[y: y + kernel, x: x + kernel, :] / 255
                    patches.append(image)
                    grids.append((x, y))
            image = torch.tensor(patches, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            # 在这里做升降采样
            image = F.interpolate(image, size=(kernel // 2, kernel // 2), mode='bilinear').contiguous()

            # image = torch.concat(patches, dim=0).contiguous()
            # 预测流程
            coarse_patch: torch.Tensor
            fine_patch: torch.Tensor
            classify_patch: torch.Tensor
            coarse_patch, fine_patch, classify_patch = detector(image)

            # 在这里做升降采样
            coarse_patch = F.interpolate(coarse_patch, size=(kernel, kernel), mode='bilinear', align_corners=False).contiguous()
            fine_patch = F.interpolate(fine_patch, size=(kernel, kernel), mode='bilinear', align_corners=False).contiguous()
            classify_patch = F.interpolate(classify_patch, size=(kernel, kernel), mode='bilinear', align_corners=False).contiguous()

            # 预测后处理
            coarse_patch = coarse_patch.softmax(dim=1)[:, 1:, :, :]
            fine_patch = fine_patch.clamp(0, 1)
            classify_patch = classify_patch.clamp(0, 1)
            # 融合
            merger.set(
                patches_group=(coarse_patch, fine_patch, classify_patch),
                grids=grids,
            )
            coarse_heat_map, fine_heat_map, classify_heat_map = merger.tail()
            if cache_code:
                torch.save(coarse_heat_map, rf'{output_root}/predict/{cache_code}_coarse.weight')
                torch.save(fine_heat_map, rf'{output_root}/predict/{cache_code}_fine.weight')
                torch.save(classify_heat_map, rf'{output_root}/predict/{cache_code}_classify.weight')

    # 然后执行组织分割任务，生成分割热图
    with torch.no_grad():
        # model_return_channels 用来告诉 merger 输出多少个结果，每个结果的 channel 是多少，这会决定 Merger 的融合参数控制
        merger = Merger(
                model_return_channels=(2,),
                width=width,
                height=height,
                kernel_size=kernel,
                kernel_steep=steep,
                device=device,
        )
        # 开发时有缓存用缓存，实际跑的时候没这东西
        if cache_code and os.path.exists(rf'{output_root}/predict/{cache_code}_divide.weight'):
            divide_heat_map = torch.load(rf'{output_root}/predict/{cache_code}_divide.weight', map_location=device)
        else:
            # 遍历截图
            patches = []
            grids = []
            for y in range(0, height - kernel + 1, step):
                for x in range(0, width - kernel + 1, step):
                    # 图像预处理
                    image = (tissue[y: y + kernel, x: x + kernel, :] / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    patches.append(image)
                    grids.append((x, y))
            image = torch.tensor(patches, dtype=torch.float32, device=device).permute(0, 3, 1, 2).contiguous()
            # 预测流程
            divide_heat_patch: torch.Tensor = divider(image)
            # 融合
            merger.set(
                patches_group=(divide_heat_patch,),
                grids=grids,
            )
            divide_heat_map, = merger.tail()
            if cache_code:
                torch.save(divide_heat_map, rf'{output_root}/predict/{cache_code}_divide.weight')

    # 最后乱七八糟的融合方法各种上，并且返回结果
    with torch.no_grad():
        x, y = offset
        divide_heat_map_cells_like = F.interpolate(divide_heat_map[None, :, y: y + vision, x: x + vision], size=(width, height), mode='bilinear', align_corners=False)
        divide_heat_map_cells_like = divide_heat_map_cells_like[0, :, :, :]

        return correlated(
            coarse=coarse_heat_map,
            fine=fine_heat_map,
            classify=classify_heat_map,
            divide=divide_heat_map_cells_like,
            device=device,
            image=cell,
        )
