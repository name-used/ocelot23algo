import json
import cv2
import matplotlib.pyplot as plt

from user.inference import Model as Processor


with open(r'D:\jassorRepository\OCELOT_Dataset\jassor\metadata.json', 'r+') as f:
    metadata = json.load(f)

for code in metadata['sample_pairs'].keys():

    image_cell = cv2.imread(rf'D:\jassorRepository\OCELOT_Dataset\jassor\cell\image\{code}.jpg')
    image_cell = cv2.cvtColor(image_cell, cv2.COLOR_BGR2RGB)

    image_tissue = cv2.imread(rf'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image\{code}.jpg')
    image_tissue = cv2.cvtColor(image_tissue, cv2.COLOR_BGR2RGB)

    detect_result = Processor(metadata=metadata['sample_pairs'])(cell_patch=image_cell, tissue_patch=image_tissue, pair_id=code)

    print(detect_result)

    cls_color = {
            0: (255, 0,   0),#Neoplastic
            1: (153, 204, 153),#Inflammatory
        }
    for x, y, c, p in detect_result:
        cv2.circle(image_cell, (x, y), 3, cls_color[c], 3)

    plt.imshow(image_cell)
    plt.show()
