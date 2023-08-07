import gc
import json
import os
from typing import List, Tuple, Dict, Any
# from line_profiler import LineProfiler
import PIL.Image
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
            if code != '400':
                continue
            T.track(f' -> start {code}')
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
        hard = [
            17, 199, 216, 239, 243, 250, 259, 280, 300, 310, 346, 351, 354, 368,
            26, 61, 74, 89, 135, 187, 203, 236, 237, 251, 270, 272, 281, 285, 294,
            295, 304, 317, 319, 332, 359, 362, 364, 373, 393, 399
        ]
        hard = [str(index).zfill(3) for index in hard]

        blocks = [14, 38, 51, 79, 129, 131, 133, 135, 138, 140, 144, 147, 152, 168, 172, 173, 180, 181, 223, 244, 252, 255, 256, 267, 279,
                  286, 287, 292, 294, 307, 314, 315, 323, 325, 334, 341, 345, 376, 393, 396, 397]
        blocks = [str(x).zfill(3) for x in blocks]

        available = [str(x).zfill(3) for x in available if x not in blocks]
        trains = [str(x).zfill(3) for x in trains if x not in blocks and x not in hard]
        valids = [str(x).zfill(3) for x in valids if x not in blocks and x not in hard]
        hard = [str(x).zfill(3) for x in hard if x not in blocks]

        for group, group_name in [
                (available, 'available'),
                (trains, 'train'),
                (valids, 'valid'),
                (hard, 'hard'),
                (blocks, 'blocked'),
        ]:
            metrics = evaluate(
                prs=[results[code]['predicts'] for code in group],
                gts=[results[code]['labels'] for code in group]
            )
            f.writelines([
                f' -------------------- {group_name} 评分-------------------- \n',
                'f1_mix: %.4f \t f1_bc: %.4f \t f1_tc: %.4f \t mf1: %.4f\n' % metrics,
            ])

        for group, group_name in [
            (available, 'available'),
            (trains, 'train'),
            (valids, 'valid'),
            (hard, 'hard'),
            (blocks, 'blocked'),
        ]:
            f.writelines([
                f' -------------------- 逐图片评分 ({group_name}) -------------------- \n',
            ])
            for code in available:
                if code not in group: continue
                metrics = evaluate(
                    prs=[results[code]['predicts']],
                    gts=[results[code]['labels']]
                )
                f.writelines([
                    'f1_mix: %.4f \t f1_bc: %.4f \t f1_tc: %.4f \t mf1: %.4f\n' % metrics,
                ])


class Timer(object):
    def __init__(self, start: int = 0, indentation: int = 0):
        self.con = Condition()
        self.start = start or time()
        self.indentation = indentation

    def __enter__(self):
        with self.con:
            print('#%s enter at %.4f seconds' % ('\t' * self.indentation, time() - self.start))
        return self.tab()

    def track(self, message: str):
        with self.con:
            print('#%s %s -> at time %.4f' % ('\t' * self.indentation, message, time() - self.start))

    def tab(self):
        with self.con:
            return Timer(start=self.start, indentation=self.indentation+1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.con:
            print('#%s exit at %.4f seconds' % ('\t' * self.indentation, time() - self.start))
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

    # image_cell = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
    # image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)
    image_cell = np.array(PIL.Image.open(rf'/media/predator/totem/jizheng/ocelot2023/test_submit/cell.tif'))

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

    result = {"type": "Multiple points", "points": [
        {"name": "image_0", "point": [246, 1018, 2], "probability": 0.35667151724261015},
            {"name": "image_0", "point": [144, 1000, 2], "probability": 0.3700178278910613},
            {"name": "image_0", "point": [847, 993, 1], "probability": 0.4897167708231436},
            {"name": "image_0", "point": [301, 990, 1], "probability": 0.5957599504928234},
            {"name": "image_0", "point": [7, 986, 1], "probability": 0.5256677062503247},
            {"name": "image_0", "point": [269, 972, 1], "probability": 0.6741500811347951},
            {"name": "image_0", "point": [164, 953, 1], "probability": 0.6348970161446135},
            {"name": "image_0", "point": [4, 952, 1], "probability": 0.3180947088688164},
            {"name": "image_0", "point": [92, 950, 1], "probability": 0.6350390883489858},
            {"name": "image_0", "point": [207, 941, 1], "probability": 0.67376227851792},
            {"name": "image_0", "point": [30, 921, 1], "probability": 0.349892005361781},
            {"name": "image_0", "point": [317, 917, 1], "probability": 0.44398134907784564},
            {"name": "image_0", "point": [437, 915, 1], "probability": 0.5501556125786075},
            {"name": "image_0", "point": [252, 889, 1], "probability": 0.39926971274155676},
            {"name": "image_0", "point": [204, 882, 1], "probability": 0.6097194534412012},
            {"name": "image_0", "point": [306, 881, 1], "probability": 0.4129047478479904},
            {"name": "image_0", "point": [25, 868, 1], "probability": 0.6741966408637909},
            {"name": "image_0", "point": [1015, 866, 1], "probability": 0.4141620752695641},
            {"name": "image_0", "point": [455, 849, 1], "probability": 0.6504288273932518},
            {"name": "image_0", "point": [637, 837, 1], "probability": 0.6603035464428639},
            {"name": "image_0", "point": [159, 836, 1], "probability": 0.6035069869674388},
            {"name": "image_0", "point": [368, 826, 1], "probability": 0.4772213071870304},
            {"name": "image_0", "point": [230, 822, 1], "probability": 0.30771329903695843},
            {"name": "image_0", "point": [908, 819, 1], "probability": 0.5837563346111099},
            {"name": "image_0", "point": [152, 817, 1], "probability": 0.30752484217433107},
            {"name": "image_0", "point": [199, 816, 1], "probability": 0.6749978924527582},
            {"name": "image_0", "point": [570, 801, 1], "probability": 0.5779000862957151},
            {"name": "image_0", "point": [542, 793, 1], "probability": 0.6127968395638931},
            {"name": "image_0", "point": [922, 792, 1], "probability": 0.5729169922209394},
            {"name": "image_0", "point": [328, 756, 1], "probability": 0.6259490753880931},
            {"name": "image_0", "point": [143, 750, 1], "probability": 0.65124707921504},
            {"name": "image_0", "point": [297, 747, 1], "probability": 0.6037934108304238},
            {"name": "image_0", "point": [642, 742, 1], "probability": 0.6527367422080159},
            {"name": "image_0", "point": [479, 742, 1], "probability": 0.6269350841793205},
            {"name": "image_0", "point": [435, 722, 1], "probability": 0.598218276382241},
            {"name": "image_0", "point": [412, 721, 1], "probability": 0.4474786347929404},
            {"name": "image_0", "point": [624, 708, 1], "probability": 0.6093212856970491},
            {"name": "image_0", "point": [923, 705, 1], "probability": 0.4474896329173147},
            {"name": "image_0", "point": [8, 705, 1], "probability": 0.396557851694698},
            {"name": "image_0", "point": [821, 697, 1], "probability": 0.6219074144097638},
            {"name": "image_0", "point": [250, 694, 1], "probability": 0.465422490107734},
            {"name": "image_0", "point": [600, 681, 1], "probability": 0.4712818238906962},
            {"name": "image_0", "point": [744, 674, 1], "probability": 0.6949361045958965},
            {"name": "image_0", "point": [716, 659, 1], "probability": 0.6706858235844109},
            {"name": "image_0", "point": [314, 649, 1], "probability": 0.37300786575945105},
            {"name": "image_0", "point": [926, 648, 1], "probability": 0.6162076975278842},
            {"name": "image_0", "point": [669, 648, 1], "probability": 0.4387533222251688},
            {"name": "image_0", "point": [999, 636, 1], "probability": 0.6514388674816114},
            {"name": "image_0", "point": [772, 634, 1], "probability": 0.40153128009380296},
            {"name": "image_0", "point": [587, 626, 1], "probability": 0.6430122997867431},
            {"name": "image_0", "point": [827, 616, 1], "probability": 0.4680893372117122},
            {"name": "image_0", "point": [351, 600, 1], "probability": 0.6556184798189675},
            {"name": "image_0", "point": [847, 568, 1], "probability": 0.3727153011045247},
            {"name": "image_0", "point": [638, 565, 1], "probability": 0.6250417706153387},
            {"name": "image_0", "point": [958, 561, 1], "probability": 0.5431122397478046},
            {"name": "image_0", "point": [991, 538, 1], "probability": 0.6404798562818816},
            {"name": "image_0", "point": [797, 538, 1], "probability": 0.52303754970636},
            {"name": "image_0", "point": [584, 517, 1], "probability": 0.48238819448607617},
            {"name": "image_0", "point": [614, 476, 1], "probability": 0.3957769546547073},
            {"name": "image_0", "point": [879, 441, 1], "probability": 0.5537727516548726},
            {"name": "image_0", "point": [468, 434, 1], "probability": 0.5525132504538329},
            {"name": "image_0", "point": [727, 425, 1], "probability": 0.5537139994011369},
            {"name": "image_0", "point": [347, 425, 1], "probability": 0.5650179832967219},
            {"name": "image_0", "point": [637, 420, 1], "probability": 0.641779471862861},
            {"name": "image_0", "point": [263, 416, 1], "probability": 0.6101205419854439},
            {"name": "image_0", "point": [831, 412, 1], "probability": 0.6300556738120962},
            {"name": "image_0", "point": [306, 374, 1], "probability": 0.3232457719117156},
            {"name": "image_0", "point": [235, 372, 1], "probability": 0.5903020176273062},
            {"name": "image_0", "point": [1001, 363, 1], "probability": 0.47394391272643543},
            {"name": "image_0", "point": [251, 343, 1], "probability": 0.3346457513396063},
            {"name": "image_0", "point": [362, 230, 1], "probability": 0.4538018182574449},
            {"name": "image_0", "point": [535, 227, 1], "probability": 0.48168726725644745},
            {"name": "image_0", "point": [572, 204, 1], "probability": 0.6161888961369923},
            {"name": "image_0", "point": [352, 196, 1], "probability": 0.3127758453735342},
            {"name": "image_0", "point": [15, 170, 1], "probability": 0.7052349753436502},
            {"name": "image_0", "point": [57, 158, 1], "probability": 0.5156896928736288},
            {"name": "image_0", "point": [594, 153, 1], "probability": 0.5994958359243968},
            {"name": "image_0", "point": [127, 127, 1], "probability": 0.5817544565533197},
            {"name": "image_0", "point": [77, 107, 1], "probability": 0.6509113009749117},
            {"name": "image_0", "point": [836, 102, 1], "probability": 0.552578725290019},
            {"name": "image_0", "point": [382, 85, 1], "probability": 0.5833780724624711},
            {"name": "image_0", "point": [345, 83, 1], "probability": 0.5874185589646144},
            {"name": "image_0", "point": [348, 26, 1], "probability": 0.5507228551349095},
            {"name": "image_0", "point": [525, 8, 1], "probability": 0.3744287977294678}
    ],"version": {"major": 1, "minor": 0}}
    rst = [(*r['point'], r['probability']) for r in result['points']]
    r1 = evaluate(prs=[rst], gts=[labels])
    r2 = evaluate(prs=[predicts], gts=[labels])
    print(r1)
    print(r2)

    T.track(f' -> predicted {code} - f1_mix - {f1_mix} - f1_bc - {f1_bc} - f1_tc - {f1_tc} - mf1 - {mf1}')

    results[code] = {
        'predicts': predicts,
        'labels': labels,
        'score': (f1_mix, f1_bc, f1_tc, mf1),
    }

    visual(code=code, image=image_cell, predicts=predicts, labels=labels)


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
