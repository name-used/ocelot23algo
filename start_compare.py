import gc
import json
import os
import sys
from typing import List, Tuple, Dict, Any
import os
import io
# from line_profiler import LineProfiler
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from time import time
from threading import Condition
import torch
import torch.nn.functional as F

from config import output_root
from evaluation.eval import _preprocess_distance_and_confidence, _calc_scores
from main.fusion_wentai import correlated as method_1
from main.fusion_liner import correlated as method_2
from main.show import PPlot


def main():
    results: Dict[str, Any] = {}
    with open(rf'/media/predator/totem/jizheng/ocelot2023/submit_test/visual/log.txt', 'a+') as output:
        with Timer(output=output) as T:
            for code in metadata.keys():
                T.track(f' -> start {code}')
                # 从 metadata 中生成 cell 图在 tissue 图里的偏移量，其它参数都是固定的，不用管
                offset_x = round((metadata[code]['patch_x_offset'] - 0.125) * 1024)
                offset_y = round((metadata[code]['patch_y_offset'] - 0.125) * 1024)

                # 细胞检测标签
                with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
                    labels: List[Tuple[int, int, int]] = []
                    for line in f.readlines():
                        x, y, c = map(int, line.split(','))
                        labels.append((x, y, c))
                # 预测权重
                coarse = torch.load(rf'{output_root}/predict/{code}_coarse.weight')
                fine = torch.load(rf'{output_root}/predict/{code}_fine.weight')
                classify = torch.load(rf'{output_root}/predict/{code}_classify.weight')
                divide = torch.load(rf'{output_root}/predict/{code}_divide.weight')
                # 叉上来
                divide = F.interpolate(divide[None, :, offset_y: offset_y + 256, offset_x: offset_x + 256], size=(1024, 1024), mode='bilinear')
                divide = divide[0, :, :, :]

                # 两种后处理方法
                predicts1 = method_1(
                    coarse=coarse,
                    fine=fine,
                    classify=classify,
                    divide=divide,
                    device='cuda:0',
                    image=None,
                )
                predicts2 = method_2(
                    coarse=coarse,
                    fine=fine,
                    classify=classify,
                    divide=divide,
                    device='cuda:0',
                    image=None,
                )

                results[code] = {
                    'predicts1': predicts1,
                    'predicts2': predicts2,
                    'labels': labels,
                    'divide': divide[1, :, :].cpu().numpy(),
                }

            # 先做整体评价
            for group, group_name in [
                (available, 'available'),
                (trains, 'train'),
                (valids, 'valid'),
                (blocks, 'blocked'),
            ]:
                T.track(f' ------------------------- {group_name} in total ------------------------------------')
                f1_mix_1, f1_bc_1, f1_tc_1, mf1_1 = evaluate(
                    prs=[results[code]['predicts1'] for code in group],
                    gts=[results[code]['labels'] for code in group]
                )
                f1_mix_2, f1_bc_2, f1_tc_2, mf1_2 = evaluate(
                    prs=[results[code]['predicts2'] for code in group],
                    gts=[results[code]['labels'] for code in group]
                )
                T1 = T.tab()
                T1.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1))
                T1.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_2, f1_bc_2, f1_tc_2, mf1_2))
                T1.track(f'change -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' %
                         (f1_mix_2 - f1_mix_1, f1_bc_2 - f1_bc_1, f1_tc_2 - f1_tc_1, mf1_2 - mf1_1))

            for change_min, change_max in [
                (0.0, 0.05),
                (0.05, 0.1),
                (0.1, 0.2),
                (0.2, 0.3),
                (0.4, 1),
            ]:
                target = f'{int(change_min * 100)}_{int(change_max * 100)}'
                with open(rf'/media/predator/totem/jizheng/ocelot2023/submit_test/visual/{target}.txt', 'w+') as f:
                    T1.track(f' ------------------------- {change_min} -> {change_max} ------------------------------------')
                    T2 = T1.tab()
                    for code in available:
                        T1.track(f'evaluating {code}')
                        f1_mix_1, f1_bc_1, f1_tc_1, mf1_1 = evaluate(
                            prs=[results[code]['predicts1']],
                            gts=[results[code]['labels']]
                        )
                        f1_mix_2, f1_bc_2, f1_tc_2, mf1_2 = evaluate(
                            prs=[results[code]['predicts2']],
                            gts=[results[code]['labels']]
                        )
                        T2.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1))
                        T2.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_2, f1_bc_2, f1_tc_2, mf1_2))
                        T2.track(f'change -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' %
                                 (f1_mix_2 - f1_mix_1, f1_bc_2 - f1_bc_1, f1_tc_2 - f1_tc_1, mf1_2 - mf1_1))
                        if change_min < mf1_1 - mf1_2 < change_max:
                            visual(code=code, target=target, info=results[code])
                            f.writelines([
                                f'code = {code}\t',
                                f'(f1_mix_1, f1_bc_1, f1_tc_1, mf1_1) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1),
                                f'(f1_mix_2, f1_bc_2, f1_tc_2, mf1_2) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_2, f1_bc_2, f1_tc_2, mf1_2),
                                f'(f1_mix_+, f1_bc_+, f1_tc_+, mf1_+) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_2 - f1_mix_1, f1_bc_2 - f1_bc_1, f1_tc_2 - f1_tc_1, mf1_2 - mf1_1),
                                '\n',
                            ])

            # 再做局部评价
            for group, group_name in [(trains, 'train'), (valids, 'valid')]:
                T1.track(f' ------------------------- {group_name} for every ------------------------------------')
                T2 = T1.tab()
                with open(rf'/media/predator/totem/jizheng/ocelot2023/submit_test/visual/{group_name}.txt', 'w+') as f:
                  for code in group:
                        T1.track(f'evaluating {code}')
                        f1_mix_1, f1_bc_1, f1_tc_1, mf1_1 = evaluate(
                            prs=[results[code]['predicts1']],
                            gts=[results[code]['labels']]
                        )
                        f1_mix_2, f1_bc_2, f1_tc_2, mf1_2 = evaluate(
                            prs=[results[code]['predicts2']],
                            gts=[results[code]['labels']]
                        )
                        T2.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1))
                        T2.track(f'pre1 -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' % (f1_mix_2, f1_bc_2, f1_tc_2, mf1_2))
                        T2.track(f'change -> \t f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f' %
                                 (f1_mix_2 - f1_mix_1, f1_bc_2 - f1_bc_1, f1_tc_2 - f1_tc_1, mf1_2 - mf1_1))
                        if mf1_2 - mf1_1 < 0:
                            # visual(code=code, target=group_name, info=results[code])
                            f.writelines([
                                f'code = {code}\t',
                                f'(f1_mix_1, f1_bc_1, f1_tc_1, mf1_1) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1),
                                f'(f1_mix_2, f1_bc_2, f1_tc_2, mf1_2) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_1, f1_bc_1, f1_tc_1, mf1_1),
                                f'(f1_mix_+, f1_bc_+, f1_tc_+, mf1_+) = (%.2f, %.2f, %.2f, %.2f)\t' % (f1_mix_2 - f1_mix_1, f1_bc_2 - f1_bc_1, f1_tc_2 - f1_tc_1, mf1_2 - mf1_1),
                                '\n',
                            ])


class Timer(object):
    def __init__(self, start: int = 0, indentation: int = 0, output: io.TextIOWrapper = sys.stdout):
        self.con = Condition()
        self.start = start or time()
        self.indentation = indentation
        self.output = output

    def __enter__(self):
        with self.con:
            self.output.writelines('#%s enter at %.2f seconds\n' % ('\t' * self.indentation, time() - self.start))
        return self.tab()

    def track(self, message: str):
        with self.con:
            self.output.writelines('#%s %s -> at time %.2f\n' % ('\t' * self.indentation, message, time() - self.start))

    def tab(self):
        with self.con:
            return Timer(start=self.start, indentation=self.indentation+1, output=self.output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.con:
            self.output.writelines('#%s exit at %.2f seconds\n' % ('\t' * self.indentation, time() - self.start))
            return False


def evaluate(prs: List[List[Tuple[int, int, int, float]]], gts: List[List[Tuple[int, int, int]]]) -> Tuple[float, float, float, float]:
    dist = 15
    # 先计算正常结果
    prs = [[(x, y, c, p) for x, y, c, p in pr] for pr in prs]
    # gts = [[(x, y, c) for x, y, c in gt] for gt in gts]
    all_sample_result = _preprocess_distance_and_confidence(prs, gts)
    pre_bc, rec_bc, f1_bc = _calc_scores(all_sample_result, 1, dist)
    pre_tc, rec_tc, f1_tc = _calc_scores(all_sample_result, 2, dist)
    # 再忽略类型计算
    prs = [[(x, y, 1, p) for x, y, c, p in pr] for pr in prs]
    gts = [[(x, y, 1) for x, y, c in gt] for gt in gts]
    all_sample_result = _preprocess_distance_and_confidence(prs, gts)
    pre_mix, rec_mix, f1_mix = _calc_scores(all_sample_result, 1, dist)

    mf1 = (f1_bc + f1_tc) / 2
    return f1_mix, f1_bc, f1_tc, mf1


def visual(code: str, target: str, info: Dict[str, List[tuple]]):
    os.makedirs(rf'{output_root}/visual/{target}', exist_ok=True)
    # 图片
    cell = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    # 标签
    label = cell.copy()
    for x, y, c in info['labels']:
        cv2.circle(label, (x, y), 3, cls_color[c], 3)
    # 分割
    divide = info['divide']
    divide = np.stack([divide, divide, divide], axis=2)
    # 第一预测
    pre1 = cell.copy()
    for x, y, c, p in info['predicts1']:
        cv2.circle(pre1, (x, y), 3, cls_color[c], 3)
    # 第二预测
    pre2 = cell.copy()
    for x, y, c, p in info['predicts2']:
        cv2.circle(pre2, (x, y), 3, cls_color[c], 3)

    PPlot().title(
        'label', 'divide', 'pre1', 'pre2'
    ).add(
        label, divide, pre1, pre2
    ).save(fname=rf'{output_root}/visual/{target}/{code}.png', dpi=1000)
    gc.collect()


with open(r'/media/predator/totem/jizheng/ocelot2023/metadata.json', 'r+') as f:
    metadata = json.load(f)
    metadata = metadata['sample_pairs']

available = list(metadata.keys())
trains = ['023', '044', '008', '094', '045', '093', '025', '009', '065', '001', '076', '062', '103', '098', '087', '090', '049', '073', '086', '100',
          '067', '097', '015', '002', '038', '119', '064', '120', '043', '112', '072', '022', '099', '046', '053', '068', '012', '075', '057', '066',
          '056', '101', '069', '092', '004', '084', '052', '050', '039', '021', '016', '070', '018', '032', '108', '114', '058', '029', '003', '122',
          '027', '010', '113', '047', '081', '104', '020', '116', '054', '059', '085', '088', '115', '040', '042', '071', '111', '107', '014', '030',
          '017', '051', '118', '080', '110', '105', '078', '005', '091', '019', '063', '033', '117', '079', '035', '106', '013', '148', '189', '159',
          '137', '169', '165', '185', '198', '183', '166', '193', '153', '134', '181', '151', '186', '132', '161', '147', '157', '127', '163', '155',
          '201', '125', '164', '177', '171', '123', '131', '145', '196', '173', '191', '138', '204', '154', '190', '156', '184', '208', '142', '130',
          '179', '197', '126', '139', '146', '194', '158', '176', '174', '180', '199', '192', '167', '200', '207', '172', '160', '136', '202', '175',
          '188', '195', '144', '209', '263', '242', '246', '228', '266', '247', '249', '216', '212', '258', '239', '286', '226', '215', '276', '243',
          '222', '229', '245', '224', '254', '235', '253', '233', '240', '260', '278', '277', '227', '231', '280', '269', '282', '211', '264', '220',
          '274', '259', '234', '213', '217', '271', '289', '265', '255', '287', '221', '214', '230', '250', '275', '225', '288', '256', '290', '244',
          '232', '210', '262', '219', '238', '261', '284', '313', '326', '309', '323', '297', '310', '308', '299', '320', '321', '301', '337', '296',
          '318', '293', '315', '325', '291', '307', '322', '329', '331', '336', '312', '305', '300', '302', '292', '334', '333', '306', '327', '311',
          '303', '330', '298', '344', '339', '340', '351', '338', '358', '368', '367', '350', '352', '361', '342', '369', '354', '372', '357', '366',
          '349', '360', '353', '356', '355', '346', '371', '341', '365', '343', '387', '390', '391', '394', '388', '383', '380', '374', '385', '389',
          '379', '384', '392', '377', '386', '398', '397', '375', '378', '395']
valids = ['034', '060', '095', '041', '077', '006', '048', '036', '082', '061', '083', '055', '109', '011', '089', '028', '037', '007', '024', '102',
          '096', '031', '074', '121', '026', '206', '178', '162', '141', '128', '135', '182', '203', '170', '124', '133', '168', '143', '150', '187',
          '205', '149', '129', '140', '283', '279', '285', '248', '281', '241', '267', '237', '236', '268', '257', '252', '273', '223', '272', '251',
          '270', '218', '314', '294', '304', '328', '332', '335', '324', '317', '295', '316', '319', '363', '347', '359', '364', '362', '348', '345',
          '370', '373', '400', '396', '382', '376', '381', '399', '393']
blocks = [14, 38, 51, 79, 129, 131, 133, 135, 138, 140, 144, 147, 152, 168, 172, 173, 180, 181, 223, 244, 252, 255, 256, 267, 279,
          286, 287, 292, 294, 307, 314, 315, 323, 325, 334, 341, 345, 376, 393, 396, 397]
blocks = [str(x).zfill(3) for x in blocks]
available = [str(x).zfill(3) for x in available if x not in blocks]
trains = [str(x).zfill(3) for x in trains if x not in blocks]
valids = [str(x).zfill(3) for x in valids if x not in blocks]

cls_color = {
    1: (153, 204, 153),
    2: (255, 0, 0),
}

main()
