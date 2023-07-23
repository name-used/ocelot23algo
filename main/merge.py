from typing import Tuple, List, Iterable
import cv2
import numpy as np
import torch


class Merger(object):
    def __init__(
            self,
            model_return_channels,
            width: int,
            height: int,
            kernel_size: int,
            kernel_steep: float,
            device: str,
    ):
        # self.kns = {self.__kernel__(zoom, steep) for zoom, steep in kernel_params.items()}
        self.k = kernel_size
        self.w = width
        self.h = height
        # target 用来存放多个返回结果，helper 只存一个高斯核
        self.targets: List[torch.Tensor] = [
            torch.zeros(self.h, self.w, channel, dtype=torch.float64, device=device)
            for channel in model_return_channels
        ]
        self.helper: torch.Tensor = torch.zeros(1, self.h, self.w, dtype=torch.float64, device=device) + 1e-17
        # kernel 用于 同预测结果相乘 (1, )
        self.kernel = self.gaussian_kernel(size=kernel_size, steep=kernel_steep, device=device)

    def set(self, patches_group: List[Iterable[torch.Tensor]], grids: List[Tuple[int, int]]) -> None:
        # 拆 returns
        for target, patches in zip(self.targets, patches_group):
            helper = self.helper
            # 高斯融合
            patches = patches * self.kernel
            # 贴片
            for (x, y), patch in zip(grids, patches):
                target[y: y+self.k, x: x+self.k, :] += patch[y: y+self.k, x: x+self.k, :]
                helper[y: y+self.k, x: x+self.k, :] += self.kernel / len(self.targets)

    def tail(self) -> List[torch.Tensor]:
        return [target / self.helper for target in self.targets]

    @staticmethod
    def gaussian_kernel(size: int = 3, steep: float = 2, device: str = 'cpu') -> torch.Tensor:
        """
        provide an square matrix that matches the gaussian function
        this may used like an kernel of weight
        """
        x = cv2.getGaussianKernel(ksize=size, sigma=size / steep)
        # the numbers are too small ~ and there is no influence on multiple
        x /= np.average(x)
        x = np.matmul(x, x.T)
        return torch.tensor(x, dtype=torch.float64, device=device).unsqueeze(2).unsqueeze(0)
