import json
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt

from user.inference import Model as Processor
from evaluation.eval import _preprocess_distance_and_confidence, _calc_scores


def main():
    with open(r'D:\jassorRepository\OCELOT_Dataset\jassor\metadata.json', 'r+') as f:
        metadata = json.load(f)

    for code in metadata['sample_pairs'].keys():

        image_cell = cv2.imread(rf'D:\jassorRepository\OCELOT_Dataset\jassor\cell\image\{code}.jpg')
        image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)

        image_tissue = cv2.imread(rf'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image\{code}.jpg')
        image_tissue = cv2.cvtColor(image_tissue, cv2.COLOR_BGR2RGB)

        with open(rf'D:\jassorRepository\OCELOT_Dataset\jassor\cell\label_origin\{code}.csv', 'r+') as f:
            label_txt = f.readlines()
        labels = [
            tuple(map(int, line.split(','))) for line in label_txt
        ]

        predicts = Processor(metadata=metadata['sample_pairs'])(cell_patch=image_cell, tissue_patch=image_tissue, pair_id=code)

        f1_bc, f1_tc, mf1 = evaluate(pr=predicts, gt=labels)

        print(code, f1_bc, f1_tc, mf1)
        # print(score, detect_result)

        cls_color = {
                0: (153, 204, 153),
                1: (255, 0,   0),
            }
        predict_visual = image_cell.copy()
        for x, y, c, p in predicts:
            cv2.circle(predict_visual, (x, y), 3, cls_color[c - 1], 3)

        label_visual = image_cell.copy()
        for x, y, c in labels:
            cv2.circle(label_visual, (x, y), 3, cls_color[c - 1], 3)

        plt.subplot(1, 2, 1)
        plt.title('predict')
        plt.imshow(predict_visual)
        plt.subplot(1, 2, 2)
        plt.title('label')
        plt.imshow(label_visual)
        plt.show()


def evaluate(pr: List[Tuple[int, int, int, float]], gt: List[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence([pr], [gt])

    pre_bc, rec_bc, f1_bc = _calc_scores(all_sample_result, 1, 15)
    pre_tc, rec_tc, f1_tc = _calc_scores(all_sample_result, 2, 15)
    mf1 = (f1_bc + f1_tc) / 2
    return f1_bc, f1_tc, mf1


main()
