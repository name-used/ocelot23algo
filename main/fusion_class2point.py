from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.draw import disk as sk_disk

from .utils import gaussian_kernel


def correlated(
        coarse: torch.Tensor,               # (1, 1024, 1024)
        fine: torch.Tensor,                 # (1, 1024, 1024)
        classify: torch.Tensor,             # (2, 1024, 1024)
        divide: torch.Tensor,               # (2, 1024, 1024)
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
    # classify = combo * classify[None, :, :, :] * divide[None, :, :, :]
    # 转入 numpy 交给泰哥代码
    combo = combo[0, 0, :, :, None].cpu().numpy()
    classify = classify[0, :, :, :].permute(1, 2, 0).cpu().numpy()

    # 范围截断
    combo[:, :, 0] = combo[:, :, 0] * heatmap_nms(combo[:, :, 0])

    # 获得点列
    points = get_pts_from_hm(combo, thresh)

    # 完备化
    points = [(x, y, get_cls_pts_from_hm((x, y), classify)+1, combo[y, x, 0]) for y, x in points]

    # 去除无分类点
    points = [p for p in points if p[2]]

    return points


def heatmap_nms(hm: np.ndarray):
    # a = (hm * 255).astype(np.int32)
    # a1 = cv2.blur(hm * 255, (3, 3)).astype(np.int32)
    # a2 = cv2.blur(hm * 255, (5, 5)).astype(np.int32)
    # a3 = cv2.blur(hm * 255, (7, 7)).astype(np.int32)
    # h = a + a1 + a2 + a3
    # h = (h / 4).astype(np.float32)
    device = 'cuda:1'
    kernel = gaussian_kernel(size=17, steep=3, device=device)[None, None, :, :]    # 初始值 9
    kernel = kernel / kernel.sum()
    h = torch.tensor(hm, dtype=torch.float64, device=device)[None, None, :, :]
    h = torch.conv2d(h, kernel, bias=None, stride=1, padding=8, dilation=1, groups=1)
    h = h[0, 0, :, :].detach().cpu().numpy()

    ht = torch.tensor(h)[None, None, ...]
    # htm = torch.nn.functional.max_pool2d(ht, 3, stride=1, padding=1)
    # htm = torch.nn.functional.max_pool2d(ht, 7, stride=1, padding=3)
    # htm = torch.nn.functional.max_pool2d(ht, 21, stride=1, padding=10)
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()
    # h找到0和最大值的点为1
    h = (h >= hmax).astype(np.float32)
    # ohb为检测有结果的像素点，h就是得到最大值的像素点
    ohb = (hm > 0.).astype(np.float32)
    h = h * ohb
    # 将h最大值的点膨胀
    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h


def get_pts_from_hm(hm: np.ndarray, prob: float=0.5):
    '''
    从热图中生成中心点
    :param hm:
    :param prob:
    :return:
    '''
    assert hm.ndim == 3
    assert hm.shape[2] == 1
    hm_bin = (hm > prob).astype(np.uint8)
    contours = find_contours(hm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, keep_invalid_contour=True)
    boxes = np.array([make_bbox_from_contour(c) for c in contours], np.float32)
    if len(boxes) > 0:
        pts = (boxes[:, :2] + boxes[:, 2:]) / 2
        pts = np.round(pts).astype(np.int64)
        pts = list(pts)
    else:
        pts = []
    return pts


def find_contours(im, mode, method, keep_invalid_contour=False):
    '''
    cv2.findContours 的包装，区别是会自动转换cv2的轮廓格式到我的格式，和会自动删除轮廓点少于3的无效轮廓
    :param im:
    :param mode: 轮廓查找模式，例如 cv2.RETR_EXTERNAL cv2.RETR_TREE, cv2.RETR_LIST
    :param method: 轮廓优化方法，例如 cv2.CHAIN_APPROX_SIMPLE cv2.CHAIN_APPROX_NONE
    :param keep_invalid_contour: 是否保留轮廓点少于3的无效轮廓。
    :return:
    '''
    # 简化轮廓，目前用于缩放后去除重叠点，并不进一步简化
    # epsilon=0 代表只去除重叠点
    contours, _ = cv2.findContours(im, mode=mode, method=method)
    # 删除轮廓点少于3的无效轮廓
    valid_contours = []
    if not keep_invalid_contour:
        for c in contours:
            if len(c) >= 3:
                valid_contours.append(c)
    else:
        valid_contours.extend(contours)
    valid_contours = tr_cv_to_my_contours(valid_contours)
    return valid_contours


def tr_cv_to_my_contours(cv_contours):
    '''
    轮廓格式转换，转换opencv格式到我的格式
    :param cv_contours:
    :return:
    '''
    out_contours = [c[:, 0, ::-1] for c in cv_contours]
    return out_contours


def make_bbox_from_contour(contour):
    '''
    求轮廓的外接包围框
    :param contour:
    :return:
    '''
    min_y = np.min(contour[:, 0])
    max_y = np.max(contour[:, 0])
    min_x = np.min(contour[:, 1])
    max_x = np.max(contour[:, 1])
    bbox = np.array([min_y, min_x, max_y, max_x])
    return bbox


def get_cls_pts_from_hm(point, cls_hm: np.ndarray):
    hw = cls_hm.shape[:2]
    probs = np.zeros([cls_hm.shape[2]], np.float32)
    rr, cc = sk_disk(point, radius=3, shape=hw)
    for c in range(cls_hm.shape[2]):
        probs[c] = cls_hm[rr, cc, c].sum()
    if probs.sum() > 0.0:
        return np.argmax(probs)
    else:
        return -1
