import io
import sys
from threading import Condition
from time import time
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F
import json
import segmentation_models_pytorch as smp

from config import output_root


def main():
    net = Net()
    net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3, eps=1e-17)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, 2, 1e-5)

    # kernel 用来处理 label 的问题
    kernel = gaussian_kernel(size=16, steep=4, device=device)
    kernel = kernel - kernel.min()
    kernel = kernel / kernel.max()

    with Timer() as T:
        for i in range(40):
            scheduler.step(i)
            optimizer.zero_grad()
            for code in trains:
                # T.track(f' -> epoch {i}: training {code}')
                # 预测权重
                coarse = torch.load(rf'{output_root}/predict/{code}_coarse.weight', map_location=device)
                fine = torch.load(rf'{output_root}/predict/{code}_fine.weight', map_location=device)
                classify = torch.load(rf'{output_root}/predict/{code}_classify.weight', map_location=device)
                divide = torch.load(rf'{output_root}/predict/{code}_divide.weight', map_location=device)
                # 图片
                image = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                # 细胞检测标签
                label = torch.zeros_like(classify)
                with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
                    for line in f.readlines():
                        x, y, c = map(int, line.split(','))
                        label[c - 1, y, x] = 1
                # 组织为 batches 进行训练
                images = []
                coarses = []
                fines = []
                classifies = []
                divides = []
                gts = []
                for y in range(0, 1024 - 16 + 1, 16):
                    for x in range(0, 1024 - 16 + 1, 16):
                        images.append(image[:, y:y + 16, x:x + 16])
                        coarses.append(coarse[:, y:y + 16, x:x + 16])
                        fines.append(fine[:, y:y + 16, x:x + 16])
                        classifies.append(classify[:, y:y + 16, x:x + 16])
                        divides.append(divide[:, y:y + 16, x:x + 16])
                        gts.append(torch.tensor([
                            # 维度 0 表示点的自信度
                            (label[:, y:y + 16, x:x + 16].sum(0) * kernel).sum(),
                            # 维度 1、2 表示点的类型
                            (label[0, y:y + 16, x:x + 16] * kernel).sum(),
                            (label[1, y:y + 16, x:x + 16] * kernel).sum(),
                        ], dtype=torch.float32, device=device))

                images = torch.stack(images).type(torch.float32)
                coarses = torch.stack(coarses).type(torch.float32)
                fines = torch.stack(fines).type(torch.float32)
                classifies = torch.stack(classifies).type(torch.float32)
                divides = torch.stack(divides).type(torch.float32)
                gts = torch.stack(gts).type(torch.float32)

                prs = net(images, coarses, fines, classifies, divides)

                loss = gts * torch.log(prs + 1e-17) + (1 - gts) * torch.log(1 - prs + 1e-17)
                loss = - loss.sum()
                loss.backward()

                T.track(f' -> epoch {i}: training {code} with loss %.4f and dist %.4f' % (loss, ((prs - gts) * gts).abs().sum() / (gts.sum() + 1e-17)))
                optimizer.step()
            torch.save(net, f'./training_weights/epoch-{i}.pth')


class Net(torch.nn.Module):
    model_id = 'lasted'

    def __init__(self):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            name='resnet18',
            in_channels=3,
            depth=2,
            weights=None,
            output_stride=32,
            random=True,
        )
        # nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        # nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
        self.decoder = torch.nn.Sequential(
            # 通道卷积
            torch.nn.Conv2d(
                in_channels=64 * 6,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(64),
            # torch.nn.GELU(),
            # 分层卷积
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=0,
                dilation=1,
                groups=64,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(64),
            # torch.nn.GELU(),
            # 通道卷积
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(3),
            # 分类卷积
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=8,
                stride=1,
                padding=0,
                dilation=1,
                groups=3,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            # torch.nn.BatchNorm2d(3),
            # torch.nn.Softmax(),
            torch.nn.Sigmoid(),
        )
        # self.encoder = smp.encoders.vgg.VGGEncoder(out_channels=16, batch_norm=True)

    def forward(
            self,
            image: torch.Tensor,  # (b, 3, 1024, 1024)
            coarse: torch.Tensor,  # (b, 1, 1024, 1024)
            fine: torch.Tensor,  # (b, 1, 1024, 1024)
            classify: torch.Tensor,  # (b, 2, 1024, 1024)
            divide: torch.Tensor,  # (b, 2, 1024, 1024)
    ):
        # 首先用 encoder 对 image 进行编码 # channel == 64
        _, character_8, character_4 = self.encoder(image)
        # 插值叉上来
        character_8 = F.interpolate(character_8, size=(16, 16), mode='bilinear')
        character_4 = F.interpolate(character_4, size=(16, 16), mode='bilinear')
        # 然后扩大输入的权重
        coarse = coarse.repeat(1, 64, 1, 1)
        fine = fine.repeat(1, 64, 1, 1)
        classify = classify.repeat(1, 32, 1, 1)
        divide = divide.repeat(1, 32, 1, 1)

        # 特征粘结
        inputs = torch.concat([
            character_8, character_4, coarse, fine, classify, divide
        ], dim=1)
        # 特征运算
        return self.decoder(inputs)[:, :, 0, 0]


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
    return kernel


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
            return Timer(start=self.start, indentation=self.indentation + 1, output=self.output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.con:
            self.output.writelines('#%s exit at %.2f seconds\n' % ('\t' * self.indentation, time() - self.start))
            return False


device = 'cuda:1'

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


def demo():
    net = Net()
    net.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-2, eps=1e-17)

    inputs = torch.zeros(1, 3, 16, 16, dtype=torch.float32, device=device)
    coarse = torch.zeros(1, 1, 16, 16, dtype=torch.float32, device=device)
    fine = torch.zeros(1, 1, 16, 16, dtype=torch.float32, device=device)
    classify = torch.zeros(1, 2, 16, 16, dtype=torch.float32, device=device)
    divide = torch.zeros(1, 2, 16, 16, dtype=torch.float32, device=device)
    label = torch.zeros(1, 3, dtype=torch.float32, device=device)

    inputs[0, :, 5, 5] = coarse[0, :, 5, 5] = fine[0, :, 5, 5] = classify[0, :, 5, 5] = divide[0, :, 5, 5] = label[0, 1] = label[0, 0] = 1

    for i in range(1000):
        optimizer.zero_grad()
        outputs = net(inputs, coarse, fine, classify, divide)
        loss = label * torch.log(outputs + 1e-17) + (1 - label) * torch.log(1 - outputs + 1e-17)
        loss = - loss.sum()
        loss.backward()
        optimizer.step()

        print(i, loss, outputs.mean(), outputs.max(), outputs.min())
        print(i, loss, (outputs - label).abs().mean())


# demo()
main()
