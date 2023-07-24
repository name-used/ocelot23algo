import torch


def gaussian_kernel(size: int = 3, steep: float = 2, device: str = 'cpu') -> torch.Tensor:
    """
    provide an square matrix that matches the gaussian function
    this may used like an kernel of weight
    :param size:    就是高斯核的尺寸
    :param steep:   描述高斯核的陡峭程度，由于 sigma 必须结合 size 才有意义，因此剥离出 steep 来描述它
    :param device:
    """
    sigma = size / steep
    kernel_seed = torch.tensor([[
        -(x - size // 2) ** 2 / float(2 * sigma ** 2)
        for x in range(size)
    ]], dtype=torch.float64, device=device)
    kernel_1d = torch.exp(kernel_seed)
    # the numbers are too small ~ and there is no influence on multiple
    kernel = torch.matmul(kernel_1d.T, kernel_1d)
    kernel /= kernel.mean()
    return kernel[None, None, :, :]
