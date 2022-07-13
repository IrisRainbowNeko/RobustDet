from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES
from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, get_label_map

from .config import *
from .augmentations import *
import torch
import cv2
import numpy as np
import yaml

yaml.warnings({'YAMLLoadWarning': False})
with open('./data/dataset.yaml', 'r', encoding='utf-8') as f:  # 打开yaml文件
    dataset_names = yaml.load(f.read(), Loader=yaml.FullLoader)


def process_names_voc(namestr, type=None, year='2007'):
    types_set = namestr.split('/')
    if type:
        names = [(type, x.split("+")) for x in types_set]
    else:
        names = [(x.split(':')[0], x.split(':')[1].split("+")) for x in types_set]

    def proc(k, x):
        st = x.find('[')
        if st != -1:
            return (x[st + 1:x.find(']', st + 1)], k, dataset_names['voc'][x[:st]])
        else:
            return (year, k, dataset_names['voc'][x])

    return [[proc(k, x) for x in v] for k, v in names]

'''def process_names_coco(namestr, type=None, year='2017'):
    type_map={'trainval':'train', 'test':'val'}
    types_set = namestr.split('/')
    if type:
        names = [(type_map[type], x.split("+")) for x in types_set]
    else:
        names = [(type_map[x.split(':')[0]], x.split(':')[1].split("+")) for x in types_set]

    def proc(k, x):
        st = x.find('[')
        if st != -1:
            year_loc=x[st + 1:x.find("]", st + 1)]
            return (f'{year}_{dataset_names["coco"][x[:st]]}', f'{k}{year}')
        else:
            return (f'{year}_{dataset_names["coco"][x]}', f'{k}{year}')

    return [[proc(k, x) for x in v] for k, v in names]'''

def process_names_coco(namestr, type=None, year='2017'):
    type_map={'trainval':'train', 'test':'val'}
    types_set = namestr.split('/')
    if type:
        names = [(type_map[type], x.split("+")) for x in types_set]
    else:
        names = [(type_map[x.split(':')[0]], x.split(':')[1].split("+")) for x in types_set]

    def proc(k, x):
        st = x.find('[')
        if st != -1:
            year_loc=x.find("]", st + 1)
            info=x[st+1:year_loc].split(',')
            return (f'{year}_{dataset_names["coco"][x[:st]]}', f'{k}{year}{"_"+info[1] if len(info)>1 else ""}')
        else:
            return (f'{year}_{dataset_names["coco"][x]}', f'{k}{year}')

    return [[proc(k, x) for x in v] for k, v in names]

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    keys=list(batch[0].keys())
    datas = list(zip(*[list(x.values()) for x in batch]))
    op_dict={
        'img':lambda x:torch.stack(x, 0),
        'img_clean':lambda x:torch.stack(x, 0),
        'target':lambda x:[torch.FloatTensor(y) for y in x],
        'size':lambda x:torch.FloatTensor(x),
        'adv_label':lambda x:torch.LongTensor(x),
    }
    return [op_dict[keys[i]](x) for i,x in enumerate(datas)]

'''class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels'''
