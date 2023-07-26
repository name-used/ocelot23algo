import gc
import json
import os
from typing import List, Tuple, Dict, Any
# from line_profiler import LineProfiler
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from time import time
from threading import Condition

from config import output_root, target
from user.inference import Model
from evaluation.eval import _preprocess_distance_and_confidence, _calc_scores


def main():
    # os.makedirs(rf'{output_root}/predict', exist_ok=True)
    # os.makedirs(rf'{output_root}/points', exist_ok=True)
    # os.makedirs(rf'{output_root}/visual', exist_ok=True)

    with open(r'/media/predator/totem/jizheng/ocelot2023/metadata.json', 'r+') as f:
        metadata = json.load(f)

    model = Model(metadata=metadata['sample_pairs'], developing=True)

    results: Dict[str, Any] = {}
    with Timer() as T:
        for code in metadata['sample_pairs'].keys():
            # T.track(f' -> start {code}')
            process(results, model, code, T=T)

    with open(rf'{output_root}/{target}_score.txt', 'w+') as f:
        available = list(metadata['sample_pairs'].keys())
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

        f1_mix_all, f1_bc_all, f1_tc_all, mf1_all = evaluate(
            prs=[results[code]['predicts'] for code in available],
            gts=[results[code]['labels'] for code in available]
        )
        f1_mix_train, f1_bc_train, f1_tc_train, mf1_train = evaluate(
            prs=[results[code]['predicts'] for code in trains],
            gts=[results[code]['labels'] for code in trains]
        )
        f1_mix_valid, f1_bc_valids, f1_tc_valids, mf1_valids = evaluate(
            prs=[results[code]['predicts'] for code in valids],
            gts=[results[code]['labels'] for code in valids]
        )
        f1_mix_blocks, f1_bc_blocks, f1_tc_blocks, mf1_blocks = evaluate(
            prs=[results[code]['predicts'] for code in blocks],
            gts=[results[code]['labels'] for code in blocks]
        )
        f.writelines([
            '--------------------全体评分--------------------\n',
            'f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % (f1_mix_all, f1_bc_all, f1_tc_all, mf1_all),
            '--------------------训练集评分--------------------\n',
            'f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % (f1_mix_train, f1_bc_train, f1_tc_train, mf1_train),
            '--------------------验证集评分--------------------\n',
            'f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % (f1_mix_valid, f1_bc_valids, f1_tc_valids, mf1_valids),
            '--------------------损坏集评分--------------------\n',
            'f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % (f1_mix_blocks, f1_bc_blocks, f1_tc_blocks, mf1_blocks),
            '--------------------逐图片评分（训练集）--------------------\n',
            *[f'{code} --> f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % results[code]['score'] for code in trains],
            '--------------------逐图片评分（验证集）--------------------\n',
            *[f'{code} --> f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % results[code]['score'] for code in valids],
            '--------------------逐图片评分（损坏集）--------------------\n',
            *[f'{code} --> f1_mix: %.2f \t f1_bc: %.2f \t f1_tc: %.2f \t mf1: %.2f\n' % results[code]['score'] for code in blocks],
        ])


class Timer(object):
    def __init__(self, start: int = 0, indentation: int = 0):
        self.con = Condition()
        self.start = start or time()
        self.indentation = indentation

    def __enter__(self):
        with self.con:
            print('#%s enter at %.2f seconds' % ('\t' * self.indentation, time() - self.start))
        return self.tab()

    def track(self, message: str):
        with self.con:
            print('#%s %s -> at time %.2f' % ('\t' * self.indentation, message, time() - self.start))

    def tab(self):
        with self.con:
            return Timer(start=self.start, indentation=self.indentation+1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.con:
            print('#%s exit at %.2f seconds' % ('\t' * self.indentation, time() - self.start))
            return False


def process(results: dict, model: Model, code: str, T: Timer = None) -> None:
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 改融合方法时不允许走这个缓存 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if os.path.exists(rf'{output_root}/points/{code}.txt'):
    #
    #     image_cell = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
    #     image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)
    #
    #     with open(rf'{output_root}/points/{code}.txt', 'r+') as f:
    #         predicts: List[Tuple[int, int, int, float]] = []
    #         for line in f.readlines():
    #             x, y, c, p = line.split(',')
    #             predicts.append((int(x), int(y), int(c), float(p)))
    #
    #     with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
    #         labels: List[Tuple[int, int, int]] = []
    #         for line in f.readlines():
    #             x, y, c = map(int, line.split(','))
    #             labels.append((x, y, c))
    #
    #     f1_mix, f1_bc, f1_tc, mf1 = evaluate(prs=[predicts], gts=[labels])
    #     T.track(f' -> loaded {code} f1_bc - {f1_bc} f1_tc - {f1_tc} mf1 - {mf1}')
    #
    #     results[code] = {
    #         'predicts': predicts,
    #         'labels': labels,
    #         'score': (f1_mix, f1_bc, f1_tc, mf1),
    #     }
    #     visual(code=code, image=image_cell, predicts=predicts, labels=labels)
    #     return

    image_cell = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
    image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)

    image_tissue = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/tissue/image/{code}.jpg')
    image_tissue = cv2.cvtColor(image_tissue, cv2.COLOR_BGR2RGB)

    with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
        labels: List[Tuple[int, int, int]] = []
        for line in f.readlines():
            x, y, c = map(int, line.split(','))
            labels.append((x, y, c))

    predicts = model(cell_patch=image_cell, tissue_patch=image_tissue, pair_id=code)

    # with open(rf'{output_root}/points/{code}.txt', 'w+') as f:
    #     f.writelines([
    #         f'{x},{y},{c},{round(p, 2)}\n' for x, y, c, p in predicts
    #     ])

    f1_mix, f1_bc, f1_tc, mf1 = evaluate(prs=[predicts], gts=[labels])
    T.track(f' -> predicted {code} - f1_mix - {f1_mix} - f1_bc - {f1_bc} - f1_tc - {f1_tc} - mf1 - {mf1}')

    results[code] = {
        'predicts': predicts,
        'labels': labels,
        'score': (f1_mix, f1_bc, f1_tc, mf1),
    }

    visual(code=code, image=image_cell, predicts=predicts, labels=labels)


def evaluate(prs: List[List[Tuple[int, int, int, float]]], gts: List[List[Tuple[int, int, int]]]) -> Tuple[float, float, float, float]:
    dist = 18
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


# def evaluate(prs: List[List[Tuple[int, int, int, float]]], gts: List[List[Tuple[int, int, int]]]) -> Tuple[float, float, float, float]:
#     # 先计算正常结果
#     all_sample_result = _preprocess_distance_and_confidence(prs, gts)
#     pre_bc, rec_bc, f1_bc = _calc_scores(all_sample_result, 1, 15)
#     pre_tc, rec_tc, f1_tc = _calc_scores(all_sample_result, 2, 15)
#     # 再忽略类型计算
#     prs = [[(x, y, 0, p) for x, y, c, p in pr] for pr in prs]
#     gts = [[(x, y, 0) for x, y, c in gt] for gt in gts]
#     all_sample_result = _preprocess_distance_and_confidence(prs, gts)
#     pre_mix, rec_mix, f1_mix = _calc_scores(all_sample_result, 0, 15)
#
#     mf1 = (f1_bc + f1_tc) / 2
#     return f1_mix, f1_bc, f1_tc, mf1


def visual(code: str, image: np.ndarray, predicts: List[Tuple[int, int, int, float]], labels: List[Tuple[int, int, int]]):
    pass
    # cls_color = {
    #     1: (153, 204, 153),
    #     2: (255, 0, 0),
    # }
    # predict_visual = image.copy()
    # for x, y, c, p in predicts:
    #     cv2.circle(predict_visual, (x, y), 3, cls_color[c], 3)
    #
    # label_visual = image.copy()
    # for x, y, c in labels:
    #     cv2.circle(label_visual, (x, y), 3, cls_color[c], 3)
    #
    # if os.path.exists(rf'{output_root}/visual/{code}.png'):
    #     return
    # fig = plt.figure(code)
    # plt.subplot(1, 2, 1)
    # plt.title('predict')
    # plt.imshow(predict_visual)
    # plt.subplot(1, 2, 2)
    # plt.title('label')
    # plt.imshow(label_visual)
    # # plt.show()
    # fig.savefig(fname=rf'{output_root}/visual/{code}.png', dpi=1000)
    # plt.cla()
    # plt.clf()
    # plt.close('all')
    # gc.collect()


main()
