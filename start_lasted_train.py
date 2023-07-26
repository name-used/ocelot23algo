from typing import List, Tuple

import cv2
import torch
import json
import segmentation_models_pytorch as smp

from config import output_root


def main():

    net = Net(16)
    net.to(device)

    print(net)

    inputs = torch.zeros(1, 3, 16, 16, dtype=torch.float32, device=device)

    outputs = net(inputs, None, None, None, None)

    print(inputs.shape, len(outputs), [x.shape for x in outputs])

    return

    for code in trains:
        # 预测权重
        coarse = torch.load(rf'{output_root}/predict/{code}_coarse.weight')
        fine = torch.load(rf'{output_root}/predict/{code}_fine.weight')
        classify = torch.load(rf'{output_root}/predict/{code}_classify.weight')
        divide = torch.load(rf'{output_root}/predict/{code}_divide.weight')
        # 图片
        cell = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        # 细胞检测标签
        with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
            labels: List[Tuple[int, int, int]] = []
            for line in f.readlines():
                x, y, c = map(int, line.split(','))
                labels.append((x, y, c))


class Net(torch.nn.Module):
    model_id = 'lasted'

    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            name='resnet18',
            in_channels=3,
            depth=2,
            weights=None,
            output_stride=32,
            random=True,
        )
        pass
        # self.encoder = smp.encoders.vgg.VGGEncoder(out_channels=16, batch_norm=True)

    def forward(
            self,
            image: torch.Tensor,        # (b, 3, 1024, 1024)
            coarse: torch.Tensor,       # (b, 1, 1024, 1024)
            fine: torch.Tensor,         # (b, 1, 1024, 1024)
            classify: torch.Tensor,     # (b, 2, 1024, 1024)
            divide: torch.Tensor,       # (b, 2, 1024, 1024)
    ):
        _, character_8, character_4 = self.encoder(image)
        return self.encoder(image)


device = 'cuda:0'

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
