import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import ChainMap
import pickle

__COCO_ROOT = ['../datas/coco2017/', '/dataset/dzy/coco2017/']
for x in __COCO_ROOT:
    if osp.exists(x):
        COCO_ROOT=x
        break

IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join('data', 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        image_set [string, string]: [image folder, annotations name]
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_sets=[('train2017', 'train2017')], transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MSCOCO',
                 load_sizes=False, give_size=False, advlabel=False, clean_adv_paire=False):
        sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO

        self.datas=[]
        self.part_len=[]
        self.coco_dict = {}

        count=0
        for x in image_sets:
            if x[1] not in self.coco_dict:
                coco=COCO(osp.join(root, ANNOTATIONS, INSTANCES_SET.format(x[1])))
                self.coco_dict[x[1]]=coco

            im_keys=list(coco.imgToAnns.keys())
            count+=len(im_keys)
            self.part_len.append(count)
            self.datas.append({'root':osp.join(root, x[0]), 'coco':x[1], 'ids':im_keys})

        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        def load_size_pkl(name):
            with open(osp.join(root, f'{name}_sizes.pkl'), 'rb') as f:
                return pickle.load(f)
        #self.img_sizes = load_size_pkl('2017') if load_sizes else None

        self.load_sizes = load_sizes
        self.give_size = give_size
        self.advlabel = advlabel
        self.clean_adv_paire = clean_adv_paire


    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return self.part_len[-1]

    def get_part(self, index):
        global part
        for part, x in enumerate(self.part_len):
            if index<x:
                break
        pack=self.datas[part]
        idx=index-(0 if part==0 else self.part_len[part-1])
        return pack['root'], self.coco_dict[pack['coco']], pack['ids'][idx]

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        root, coco, img_id=self.get_part(index)

        #target = coco.imgToAnns[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        imgInfo = coco.loadImgs(img_id)[0]

        target = coco.loadAnns(ann_ids)
        path = osp.join(root, imgInfo['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, channels = img.shape

        if self.load_sizes:
            height = int(imgInfo['height'])
            width = int(imgInfo['width'])

        if self.clean_adv_paire:
            img_clean = cv2.resize(cv2.imread(osp.join(root[:root.rfind('_Attack')], path)), img.shape[1::-1])

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            if self.clean_adv_paire:
                (img, img_clean), boxes, labels = self.transform([img, img_clean], target[:, :4], target[:, 4])
            else:
                img, boxes, labels = self.transform([img], target[:, :4], target[:, 4])
                img=img[0]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        data_dict = {}
        data_dict['img'] = img
        if self.clean_adv_paire:
            data_dict['img_clean'] = img_clean
        data_dict['target'] = target
        if self.give_size:
            data_dict['size'] = (width, height)
        if self.advlabel:
            data_dict['adv_label'] = int(root.find('Attack') != -1)

        return data_dict

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
