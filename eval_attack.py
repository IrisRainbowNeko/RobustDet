"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch.backends.cudnn as cudnn
from data import *
from data import VOC_CLASSES as voc_labelmap
from data import COCO_CLASSES as coco_labelmap

from models import build_ssd, build_robust_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
from utils.utils import get_logger, Empty, Timer
import cv2
from utils.cfgParser import cfgParser

from attack import *
from robust import *

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

cfgp = cfgParser()
args=cfgp.load_cfg(['test'])
print(args)

if not os.path.exists(args.log_folder):
    os.mkdir(args.log_folder)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

exp_name=f'{os.path.basename(args.trained_model[:-4])}_{args.adv_type}_PGD-{args.atk_iters}'
logger = get_logger(os.path.join(args.log_folder, f'eval_{exp_name}.log'))

# import amp
if args.amp:
    from torch.cuda import amp
    logger.info('using amp')

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        logger.warning("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml')
#imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets', 'Main', '{}.txt')

YEAR = '2007'
devkit_path = args.dataset_root + 'VOC' + YEAR
#dataset_mean = (104, 117, 123)
set_type = 'test'

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = f'det_{exp_name}_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        logger.info('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[-1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    score_list=[]
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap, sorted_scores = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        score_list.extend(sorted_scores)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    with open(f'./confidence_{exp_name}.pkl', 'wb') as f:
        pickle.dump(score_list, f)
    logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for ap in aps:
        logger.info('{:.3f}'.format(ap))
    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('--------------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB eval code.')
    logger.info('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    logger.info(imagesetfile)
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                logger.info('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        logger.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        if args.save_tp_fp:
            with open(f'./{args.save_tp_fp}.pkl', 'wb') as f:
                pickle.dump({'fp':fp, 'tp':tp, 'npos':npos},f)
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap, sorted_scores

def test_net_attack(save_folder, net, cuda, dataset, adv_type):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    criterion_mlb = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    criterion_clsw = ClassWiseLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    dataset_mean_t = torch.tensor(DATASET_MEANS).view(1, -1, 1, 1)
    pgd = PGD(net, img_transform=(lambda x: x - dataset_mean_t, lambda x: x + dataset_mean_t))
    pgd.set_para(eps=args.step_size, alpha=lambda:args.step_size, iters=args.atk_iters)

    dag = DAG(net, img_transform=(lambda x: x - dataset_mean_t, lambda x: x + dataset_mean_t))
    dag.set_para(gamma=lambda:0.5, iters=150)

    adv_dict = {'clean': lambda:CleanGenerator(), 'cls': lambda:CLS_ADG(pgd, criterion_mlb), 'loc': lambda:LOC_ADG(pgd, criterion_mlb),
                'con': lambda:CON_ADG(pgd, criterion_mlb, rate=args.con_weights),
                'mtd': lambda:MTD(pgd, criterion_mlb),'cwat': lambda:CWAT(pgd, criterion_clsw), 'dag': lambda:AdvDataGenerator(dag, criterion_mlb)}
    adv_generator = adv_dict[adv_type.lower()]()

    _t['im_detect'].tic()
    with torch.no_grad():
        for i in range(num_images):
            #im, gt, (w, h) = dataset.pull_item(i)
            data_dict = dataset.pull_item(i)
            im = data_dict['img']
            gt = data_dict['target']
            (w, h) = data_dict['size']

            x = im.unsqueeze(0)
            if args.cuda:
                x = x.cuda()
                gt = [torch.FloatTensor(gt).cuda()]

            net.phase = 'train'
            with torch.enable_grad():
                at_img = adv_generator.generate(x, gt)
            net.phase = 'test'
            net.eval()

            with amp.autocast() if args.amp else Empty():
                detections = net(at_img)

            #cv2.imwrite('recons.png', (net.recons.detach().cpu().squeeze(0)+torch.tensor(DATASET_MEANS, device='cpu').view(-1,1,1)).permute(1,2,0).numpy()[:,:,(2,1,0)])
            #cv2.imwrite('raw.png', (im.squeeze(0)+torch.tensor(DATASET_MEANS, device='cpu').view(-1,1,1)).permute(1,2,0).numpy()[:,:,(2,1,0)])
            #0/0
            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32,copy=False)
                all_boxes[j][i] = cls_dets

            if i%100==0:
                detect_time = _t['im_detect'].toc(average=False)
                logger.info('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,num_images, detect_time/100))
                _t['im_detect'].tic()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Evaluating detections')
    if args.dataset=='VOC':
        evaluate_detections_voc(all_boxes, output_dir, dataset)
    else:
        evaluate_detections_coco(all_boxes, output_dir, dataset)

def evaluate_detections_voc(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)

def evaluate_detections_coco(box_list, output_dir, dataset):

    la_map=get_label_map(os.path.join('data', 'coco_labels.txt'))
    la_map={v:k for k,v in la_map.items()}

    bboxes = []
    probs = []
    classes = []
    image_ids = []
    for cls, data in enumerate(box_list[1:]):
        for idx, item in enumerate(data):
            if not isinstance(item, list):
                bboxes.extend(item[:,:4].tolist())
                probs.extend(item[:,4].tolist())
                classes.extend([la_map[cls+1]]*len(item))
                image_ids.extend([dataset.get_part(idx)[-1]]*len(item))

    mean_ap, detail=COCOEvaluator().evaluate(output_dir, image_ids, bboxes, classes, probs)
    logger.info(f'mean AP: {mean_ap}')
    logger.info(detail)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from io import StringIO
import json

class COCOEvaluator:
    def evaluate(self, path_to_results_dir, image_ids, bboxes, classes, probs):
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

        return self.evaluate_file(path_to_results_dir)

    def evaluate_file(self, path_to_results_dir):
        annType = 'bbox'
        path_to_coco_dir = args.dataset_root
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')

        cocoGt = COCO(path_to_annotation)
        cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'results.json'))

        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        mean_ap = cocoEval.stats[1].item()  # stats[1] records AP@[0.5]
        detail = string_stdout.getvalue()

        return mean_ap, detail

    def _write_results(self, path_to_results_dir, image_ids, bboxes, classes, probs):
        results = []
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            results.append(
                {
                    'image_id': int(image_id),  # COCO evaluation requires `image_id` to be type `int`
                    'category_id': cls,
                    'bbox': [  # format [left, top, width, height] is expected
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    'score': prob
                }
            )

        with open(os.path.join(path_to_results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    #print(get_voc_results_file_template(set_type, labelmap[3]))
    # load net
    labelmap = voc_labelmap if args.dataset=='VOC' else coco_labelmap
    cfg = voc if args.dataset == 'VOC' else coco
    #num_classes = len(labelmap) + 1                      # +1 for background
    num_classes = cfg['num_classes']
    if args.robust:
        net = build_robust_ssd('test', 300, num_classes, CFR=args.cfr, CFR_layer=args.cfr_layer, K=args.k_count, backbone=args.backbone)
    else:
        net = build_ssd('test', 300, num_classes, backbone=args.backbone)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    #torch.save(net.decoder.state_dict(), 'weights/robust_decoder_15000_2.pth')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    logger.info('Finished loading model!')

    # load data
    # evaluation
    eval_folder=(process_names_voc if args.dataset == 'VOC' else process_names_coco)(args.data_use, type='test')
    for dadv in eval_folder:
        logger.info(f'evaluate {dadv} datas')
        if args.dataset == 'VOC':
            dataset_adv = VOCDetection(args.dataset_root, dadv, BaseTransform(300),
                                       VOCAnnotationTransform(),
                                       give_size=True)
        else:
            dataset_adv = COCODetection(args.dataset_root, dadv, BaseTransform(300),
                                       COCOAnnotationTransform(), load_sizes=True,
                                       give_size=True)
        for adv_type in args.adv_type.split('/'):
            test_net_attack(args.save_folder, net, args.cuda, dataset_adv, adv_type)
