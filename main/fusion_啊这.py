from typing import List, Tuple
import cv2
import numpy as np
import torch


def correlated(
    coarse: torch.Tensor,               # (1024, 1024, 1)
    fine: torch.Tensor,                 # (1024, 1024, 1)
    classify: torch.Tensor,             # (1024, 1024, 2)
    divide: torch.Tensor,               # (1024, 1024, 2)
) -> List[Tuple[int, int, int, float]]:

    combo = coarse * fine
    detect_cls = combo * classify
    divide_cls = divide * classify

    combo[:, :, 0] = combo[:, :, 0] * what(combo[:, :, 0])
    points = how(combo, 0.2)
    classes = why(points, detect_cls)

    return [(x, y, c, 1) for (x, y), c in zip(points, classes)]


def what(hm: np.ndarray):
    a = (hm * 255).astype(np.int32)
    a1 = cv2.blur(hm, (3, 3)).astype(np.int32)
    a2 = cv2.blur(hm, (5, 5)).astype(np.int32)
    a3 = cv2.blur(hm, (7, 7)).astype(np.int32)
    ohb = (hm > 0.).astype(np.float32)

    h = a + a1 + a2 + a3

    h = (h / 4).astype(np.float32)

    ht = torch.tensor(h)[None, None, ...]
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()
    # h找到0和最大值的点为1
    h = (h >= hmax).astype(np.float32)
    # ohb为检测有结果的像素点，h就是得到最大值的像素点
    h = h * ohb
    # 将h最大值的点膨胀
    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h


def how(hm: np.ndarray, prob: float=0.5):
    '''
    从热图中生成中心点
    :param hm:
    :param prob:
    :return:
    '''
    assert hm.ndim == 3
    assert hm.shape[2] == 1
    hm_bin = (hm > prob).astype(np.uint8)
    contours = contour_tool.find_contours(hm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, keep_invalid_contour=True)
    boxes = np.array([contour_tool.make_bbox_from_contour(c) for c in contours], np.float32)
    if len(boxes) > 0:
        pts = (boxes[:, :2] + boxes[:, 2:]) / 2
        pts = np.round(pts).astype(np.int64)
        pts = list(pts)
    else:
        pts = []
    return pts


def why(det_pts, cls_hm: np.ndarray):
    '''
    从热图中生成中心点
    :param hm:
    :param prob:
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    pts_cls = np.zeros([len(det_pts)], np.int64)
    hw = cls_hm.shape[:2]
    for i, pt in enumerate(det_pts):
        probs = np.zeros([cls_hm.shape[2]], np.float32)

        # 我晕，下面这段代码其实就是把 cls_hm 上与 pt 距离 < 3 的点加在一起

        rr, cc = sk_disk(pt, radius=3, shape=hw)
        for c in range(cls_hm.shape[2]):
            probs[c] = cls_hm[rr, cc, c].sum()

        cls = np.argmax(probs)
        pts_cls[i] = cls
    return pts_cls
