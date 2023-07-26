from typing import List, Tuple
import numpy as np

from main import detect


class Model:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, developing=False):
        self.metadata = metadata
        self.developing = developing

    def __call__(self, cell_patch: np.ndarray[np.uint8], tissue_patch: np.ndarray[np.uint8], pair_id: str) -> List[Tuple[int, int, int, float]]:

        # 试着加点断言，看看会不会报错，以确定提交数据是否具备一致性
        # tissue - cell 比例关系锁定为四倍
        assert round(self.metadata[pair_id]['tissue']['resized_mpp_x'] / self.metadata[pair_id]['cell']['resized_mpp_x']) == 4
        assert round(self.metadata[pair_id]['tissue']['resized_mpp_y'] / self.metadata[pair_id]['cell']['resized_mpp_y']) == 4
        # 横纵长锁定为 1024
        assert round(
            (self.metadata[pair_id]['tissue']['x_end'] - self.metadata[pair_id]['tissue']['x_start']) *
            self.metadata[pair_id]['mpp_x'] / self.metadata[pair_id]['tissue']['resized_mpp_x']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['tissue']['y_end'] - self.metadata[pair_id]['tissue']['y_start']) *
            self.metadata[pair_id]['mpp_y'] / self.metadata[pair_id]['tissue']['resized_mpp_y']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['cell']['x_end'] - self.metadata[pair_id]['cell']['x_start']) *
            self.metadata[pair_id]['mpp_x'] / self.metadata[pair_id]['cell']['resized_mpp_x']
        ) == 1024
        assert round(
            (self.metadata[pair_id]['cell']['y_end'] - self.metadata[pair_id]['cell']['y_start']) *
            self.metadata[pair_id]['mpp_y'] / self.metadata[pair_id]['cell']['resized_mpp_y']
        ) == 1024

        # 从 metadata 中生成 cell 图在 tissue 图里的偏移量，其它参数都是固定的，不用管
        # 在上述确定的比例关系下，cell 在 tissue 中占据视野为 256*256
        # 而 (x, y) 在 metadata 中表示中心点，我们需要它表示左上角
        # 因此 (x, y) = (x, y) - (256, 256) / 2
        # x = round(self.metadata[pair_id]['patch_x_offset'] * 1024 - 128)
        # y = round(self.metadata[pair_id]['patch_y_offset'] * 1024 - 128)
        x = round((self.metadata[pair_id]['patch_x_offset'] - 0.125) * tissue_patch.shape[1])
        y = round((self.metadata[pair_id]['patch_y_offset'] - 0.125) * tissue_patch.shape[0])

        return detect(
            cell=cell_patch,
            tissue=tissue_patch,
            offset=(x, y),
            cache_code=self.developing and pair_id,
        )

    # def __call__(self, cell_patch: np.ndarray[np.uint8], tissue_patch: np.ndarray[np.uint8], pair_id: str) -> List[Tuple[int, int, int, float]]:
    #     """This function detects the cells in the cell patch. Additionally
    #     the broader tissue context is provided.
    #
    #     NOTE: this implementation offers a dummy inference example. This must be
    #     updated by the participant.
    #
    #     Parameters
    #     ----------
    #     cell_patch: np.ndarray[uint8]
    #         Cell patch with shape [1024, 1024, 3] with values from 0 - 255
    #     tissue_patch: np.ndarray[uint8]
    #         Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
    #     pair_id: str
    #         Identification number of the patch pair
    #
    #     Returns
    #     -------
    #         List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    #     """
    #     # Getting the metadata corresponding to the patch pair ID
    #     meta_pair = self.metadata[pair_id]
    #
    #     #############################################
    #     #### YOUR INFERENCE ALGORITHM GOES HERE #####
    #     #############################################
    #
    #     # The following is a dummy cell detection algorithm
    #     prediction = np.copy(cell_patch[:, :, 2])
    #     prediction[(cell_patch[:, :, 2] <= 40)] = 1
    #     xs, ys = np.where(prediction.transpose() == 1)
    #     class_id = [1] * len(xs) # Type of cell
    #     probs = [1.0] * len(xs) # Confidence score
    #
    #     # xs = [100]
    #     # ys = [200]
    #     # class_id = [0]
    #     # probs = [0.8]
    #
    #     #############################################
    #     ####### RETURN RESULS PER SAMPLE ############
    #     #############################################
    #
    #     # We need to return a list of tuples with 4 elements, i.e.:
    #     # - int: cell's x-coordinate in the cell patch
    #     # - int: cell's y-coordinate in the cell patch
    #     # - int: class id of the cell, either 1 (BC) or 2 (TC)
    #     # - float: confidence score of the predicted cell
    #     return list(zip(xs, ys, class_id, probs))
