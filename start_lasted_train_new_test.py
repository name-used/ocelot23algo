import io
import sys
from threading import Condition
from time import time
from typing import List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import json
import segmentation_models_pytorch as smp

from config import output_root


def demo():
    net = Net()
    net.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-2, eps=1e-17)

    inputs = torch.zeros(2, 3, s, s, dtype=torch.float32, device=device)
    coarse = torch.zeros(2, 1, s, s, dtype=torch.float32, device=device)
    fine = torch.zeros(2, 1, s, s, dtype=torch.float32, device=device)
    classify = torch.zeros(2, 2, s, s, dtype=torch.float32, device=device)
    divide = torch.zeros(2, 2, s, s, dtype=torch.float32, device=device)
    label = torch.zeros(2, 3, dtype=torch.float32, device=device)

    inputs[0, :, 5, 5] = coarse[0, :, 5, 5] = fine[0, :, 5, 5] = classify[0, :, 5, 5] = divide[0, :, 5, 5] = label[0, 1] = label[0, 0] = 1
    inputs[1, :, 15, 15] = coarse[1, :, 15, 15] = fine[1, :, 15, 15] = classify[1, :, 15, 15] = divide[1, :, 15, 15] = label[1, 1] = label[1, 0] = 1

    for i in range(1000):
        optimizer.zero_grad()
        outputs = net(inputs, coarse, fine, classify, divide)
        loss = label * torch.log(outputs + 1e-17) + (1 - label) * torch.log(1 - outputs + 1e-17)
        loss = - loss.sum()
        loss.backward()
        optimizer.step()

        # print(i, loss, outputs.mean(), outputs.max(), outputs.min())
        print(i, loss, (outputs - label).abs().max(), ((outputs - label) * label).abs().sum() / (label.sum() + 1e-17))


def train():
    net = Net()
    net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3, eps=1e-17)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, 2, 1e-5)

    # kernel_filter 用来均衡数据
    kernel_filter = gaussian_kernel(size=s, steep=2, device=device)

    # kernel 用来处理 label 的问题
    kernel = gaussian_kernel(size=s, steep=4, device=device)
    kernel -= kernel.min()
    kernel /= kernel.max()

    shuffled_trains = trains.copy()

    with Timer() as T:
        for i in range(max_epoch):
            np.random.shuffle(shuffled_trains)
            scheduler.step(i)
            optimizer.zero_grad()
            for code in shuffled_trains:
                # T.track(f' -> epoch {i}: training {code}')
                # 预测权重
                coarse = torch.load(rf'{output_root}/{predict_root}/{code}_coarse.weight', map_location=device)
                fine = torch.load(rf'{output_root}/{predict_root}/{code}_fine.weight', map_location=device)
                classify = torch.load(rf'{output_root}/{predict_root}/{code}_classify.weight', map_location=device)
                divide = torch.load(rf'{output_root}/{predict_root}/{code}_divide.weight', map_location=device)
                # 图片
                image = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = (image - mean) / std
                image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                # 细胞检测标签
                label_points = []
                label = torch.zeros_like(classify)
                with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
                    for line in f.readlines():
                        x, y, c = map(int, line.split(','))
                        label_points.append((x, y, c))
                        label[c - 1, y, x] = 1

                xs = torch.rand(train_batch_size)
                ys = torch.rand(train_batch_size)
                ps = torch.rand(train_batch_size)

                if len(label_points):
                    # 在有 label 的情况下围绕着 label 采样
                    lbs = (torch.rand(train_batch_size) * len(label_points)).type(torch.int64).clamp(0, len(label_points) - 1)
                    x0s = torch.tensor([label_points[lb][0] for lb in lbs])
                    y0s = torch.tensor([label_points[lb][1] for lb in lbs])

                    # 20 % 的机率全图随机
                    rule = ps < 0.2
                    xs[rule] = xs[rule] * 1009
                    ys[rule] = ys[rule] * 1009

                    # 20 % 的机率在 label 附近 30 范围随机
                    rule = (0.2 <= ps) & (ps < 0.4)
                    xs[rule] = xs[rule] * 61 - 30 + x0s[rule]
                    ys[rule] = ys[rule] * 61 - 30 + y0s[rule]

                    # 30 % 的机率在 label 附近 10 范围随机
                    rule = (0.4 <= ps) & (ps < 0.7)
                    # rule = ps < 0.8
                    xs[rule] = xs[rule] * 21 - 10 + x0s[rule]
                    ys[rule] = ys[rule] * 21 - 10 + y0s[rule]

                    # 30 % 的机率在 label 附近 3 范围随机
                    rule = 0.7 <= ps
                    xs[rule] = xs[rule] * 7 - 3 + x0s[rule]
                    ys[rule] = ys[rule] * 7 - 3 + y0s[rule]
                else:
                    xs = xs * 1009
                    ys = ys * 1009

                # 取整
                xs = (xs - s // 2).round().type(torch.int64).clamp(0, 1023 - s)
                ys = (ys - s // 2).round().type(torch.int64).clamp(0, 1023 - s)

                # 组织为 batches 进行训练
                images = [image[:, y:y + s, x:x + s] for x, y in zip(xs, ys)]
                coarses = [coarse[:, y:y + s, x:x + s] for x, y in zip(xs, ys)]
                fines = [fine[:, y:y + s, x:x + s] for x, y in zip(xs, ys)]
                classifies = [classify[:, y:y + s, x:x + s] for x, y in zip(xs, ys)]
                divides = [divide[:, y:y + s, x:x + s] for x, y in zip(xs, ys)]
                gts = [torch.tensor([
                    # 维度 0 表示点的自信度
                    (label[:, y:y + s, x:x + s].sum(0) * kernel).sum().clamp(0, 1),
                    # 维度 1、2 表示点的类型
                    (label[0, y:y + s, x:x + s] * kernel).sum().clamp(0, 1),
                    (label[1, y:y + s, x:x + s] * kernel).sum().clamp(0, 1),
                ], dtype=torch.float32, device=device)
                    for x, y in zip(xs, ys)
                ]

                images = torch.stack(images).type(torch.float32)
                coarses = torch.stack(coarses).type(torch.float32)
                fines = torch.stack(fines).type(torch.float32)
                classifies = torch.stack(classifies).type(torch.float32)
                divides = torch.stack(divides).type(torch.float32)
                gts = torch.stack(gts).type(torch.float32)

                prs = net(images, coarses, fines, classifies, divides)

                ce_loss = 3 * gts * torch.log(prs + 1e-3) + (1 - gts) * torch.log(1 - prs + 1e-3)
                loss = -ce_loss.sum()
                # for c in range(3):
                #     inter = (prs[:, c] * gts[:, c]).sum()
                #     union = (prs[:, c] + gts[:, c]).sum()
                #     dice_i_loss = 1 - 2 * inter / (union + 1e-3)
                #     dice_i_loss = dice_i_loss * (1 - torch.exp(-0.6931471805599453 * union / 0.2 / train_batch_size)) * train_batch_size
                #     loss = loss + dice_i_loss
                loss.backward()

                T.track(f' -> epoch {i + 1}: training {code} with loss %.4f and dist %.4f' % (loss, ((prs - gts) * gts).abs().sum() / (gts.sum() + 1e-17)))
                optimizer.step()
            torch.save(net, f'./training_weights/epoch-{i + 1}.pth')


def test():
    net = torch.load(f'./training_weights/epoch-100.pth')
    net.to(device)
    net.eval()

    # kernel 用来处理 label 的问题
    kernel = gaussian_kernel(size=s, steep=4, device=device)
    kernel -= kernel.min()
    kernel /= kernel.max()

    with torch.no_grad():
        with Timer() as T:
            for code in valids:
            # for code in trains:
                T.track(f' ------------------------------------ {code} ------------------------------------ ')
                # 预测权重
                coarse = torch.load(rf'{output_root}/{predict_root}/{code}_coarse.weight', map_location=device)
                fine = torch.load(rf'{output_root}/{predict_root}/{code}_fine.weight', map_location=device)
                classify = torch.load(rf'{output_root}/{predict_root}/{code}_classify.weight', map_location=device)
                divide = torch.load(rf'{output_root}/{predict_root}/{code}_divide.weight', map_location=device)
                # 图片
                image = cv2.imread(rf'/media/predator/totem/jizheng/ocelot2023/cell/image/{code}.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                # 细胞检测标签
                label_points = []
                label = torch.zeros_like(classify)
                with open(rf'/media/predator/totem/jizheng/ocelot2023/cell/label_origin/{code}.csv', 'r+') as f:
                    for line in f.readlines():
                        x, y, c = map(int, line.split(','))
                        label_points.append((x, y, c))
                        label[c - 1, y, x] = 1

                if not label_points:
                    continue

                xs, ys, cs = zip(*label_points)

                T1 = T.tab()
                for p in [0, 3, 7, 11, 15]:
                    T1.track(f'with shift {p}')
                    # 组织为 batches 进行训练
                    images = []
                    coarses = []
                    fines = []
                    classifies = []
                    divides = []
                    gts = []

                    points = [
                        (max(min(1023 - s, x - s // 2 + k * p), 0), max(min(1023 - s, y - s // 2 + l * p), 0))
                        for x, y in zip(xs, ys) for k in ([-1, 1] if p > 0 else [0]) for l in ([-1, 1] if p > 0 else [0])
                    ]

                    for x, y in points:
                        images.append(image[:, y:y + s, x:x + s])
                        coarses.append(coarse[:, y:y + s, x:x + s])
                        fines.append(fine[:, y:y + s, x:x + s])
                        classifies.append(classify[:, y:y + s, x:x + s])
                        divides.append(divide[:, y:y + s, x:x + s])
                        gts.append(torch.tensor([
                            # 维度 0 表示点的自信度
                            (label[:, y:y + s, x:x + s].sum(0) * kernel).sum().clamp(0, 1),
                            # 维度 1、2 表示点的类型
                            (label[0, y:y + s, x:x + s] * kernel).sum().clamp(0, 1),
                            (label[1, y:y + s, x:x + s] * kernel).sum().clamp(0, 1),
                        ], dtype=torch.float32, device=device))

                    images = torch.stack(images).type(torch.float32)
                    coarses = torch.stack(coarses).type(torch.float32)
                    fines = torch.stack(fines).type(torch.float32)
                    classifies = torch.stack(classifies).type(torch.float32)
                    divides = torch.stack(divides).type(torch.float32)
                    gts = torch.stack(gts).type(torch.float32)

                    prs = net(images, coarses, fines, classifies, divides)

                    T2 = T1.tab()
                    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        gt = (gts > thresh).sum(dim=0).type(torch.int64)
                        pr = (prs > thresh).sum(dim=0).type(torch.int64)
                        tp = ((gts > thresh) * (prs > thresh)).sum(dim=0).type(torch.int64)
                        T2.track(
                            f' -> thresh {thresh}\t'
                            f'sum gts == {gt.tolist()}\t'
                            f'sum prs == {pr.tolist()}\t'
                            f'sum tps == {tp.tolist()}\t'
                            f'f1 == {(2. * tp / (pr + gt + 1e-3)).tolist()}'
                        )


class Net(torch.nn.Module):
    model_id = 'lasted'

    def __init__(self):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            name='resnet18',
            in_channels=3,
            depth=2,
            weights='imagenet',
            output_stride=32,
            random=True,
            activate=None
        )
        # nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        # nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),

        self.translator = torch.nn.Sequential(
            # 通道卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=3 + 64 + 64,
                out_channels=1,
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
            torch.nn.Sigmoid(),
        )
        self.decoder = torch.nn.Sequential(
            # 通道卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=7,
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
            torch.nn.SiLU(),
            # 分层卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=4,
                dilation=1,
                groups=64,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            # 通道卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
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
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            # 分层卷积，宽度-10
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=11,
                stride=1,
                padding=0,
                dilation=1,
                groups=32,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            # 通道卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=32,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            # 分层卷积，宽度-10
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=11,
                stride=1,
                padding=0,
                dilation=1,
                groups=32,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),
            # 通道卷积，等宽变换
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
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
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU(),
            # 分类卷积，宽度-9
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=10,
                stride=1,
                padding=0,
                dilation=1,
                groups=32,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.SiLU(),
            # 通道卷积，分类
            torch.nn.Conv2d(
                in_channels=512,
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
            torch.nn.Sigmoid(),
        )
        # self.encoder = smp.encoders.vgg.VGGEncoder(out_channels=16, batch_norm=True)

    def forward(
            self,
            image: torch.Tensor,  # (b, 3, s, s)
            coarse: torch.Tensor,  # (b, 1, s, s)
            fine: torch.Tensor,  # (b, 1, s, s)
            classify: torch.Tensor,  # (b, 2, s, s)
            divide: torch.Tensor,  # (b, 2, s, s)
    ):
        # 首先用 encoder 对 image 进行编码 # channel == 64
        image_16, character_8, character_4 = self.encoder(image)
        # 插值叉上来
        character_8 = F.interpolate(character_8, size=(s, s), mode='bilinear')
        character_4 = F.interpolate(character_4, size=(s, s), mode='bilinear')
        # 特征粘结
        characters = torch.concat([
            image_16, character_8, character_4
        ], dim=1)
        # 特征合并
        character = self.translator(characters)

        # 结果粘结
        inputs = torch.concat([
            character, coarse, fine, classify, divide
        ], dim=1)

        # 结果运算
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
        -(x - (size - 1) / 2) ** 2 / float(2 * sigma ** 2)
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


predict_root = 'predict_new_detect'
device = 'cuda:1'
train_batch_size = 500
max_epoch = 100
s = 30

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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# demo()
# train()
test()
