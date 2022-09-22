from data import *
from data.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models import build_ssd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.utils import *
from utils.cfgParser import cfgParser

from attack import *
from robust import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

gvars = argparse.Namespace()


def train():
    if args.dataset == 'COCO':
        cfg = coco
        image_sets = process_names_coco(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = COCODetection(root=args.dataset_root, image_sets=image_sets,
                                transform=SSDAugmentation(cfg['min_dim'],), load_sizes=True)
    elif args.dataset == 'VOC':
        cfg = voc
        image_sets=process_names_voc(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = VOCDetection(root=args.dataset_root, image_sets=image_sets,
                               transform=SSDAugmentation(cfg['min_dim']))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], amp=args.amp and args.multi_gpu)
    net = ssd_net

    if args.cuda:
        if args.multi_gpu:
            net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.basenet!='None':
        vgg_weights = torch.load(args.save_folder + args.basenet)
        logger.info('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        if args.basenet=='None':
            logger.info('Initializing vgg weights...')
            ssd_net.vgg.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda, use_focal=args.focal)

    net.train()
    # loss counters
    gvars.loc_loss = 0
    gvars.conf_loss = 0
    gvars.epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info(f'Training SSD on: {dataset.name}')
    logger.info('Using the specified args:')
    logger.info(repr(args))

    gvars.step_index = 0
    if args.amp:
        scaler = amp.GradScaler()

    data_loader = data.DataLoader(dataset, args.batch_size//2,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)
    data_loader_adv = data.DataLoader(dataset, args.batch_size//(2 if args.clean_mix else 1),
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)

    criterion_mlb = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    criterion_clsw = ClassWiseLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    dataset_mean_t = torch.tensor(DATASET_MEANS).view(1, -1, 1, 1)
    pgd = PGD(net, img_transform=(lambda x: x - dataset_mean_t, lambda x: x + dataset_mean_t))
    pgd.set_para(eps=8, alpha=lambda:8, iters=20)
    adv_dict = {'cls': (CLS_ADG, criterion_mlb), 'loc': (LOC_ADG, criterion_mlb), 'con': (CON_ADG, criterion_mlb), 'mtd': (MTD, criterion_mlb),
                'cwat': (CWAT, criterion_clsw)}
    adv_item = adv_dict[args.adv_type.lower()]
    adv_generator = adv_item[0](pgd, adv_item[1])

    # create batch iterator
    gvars.t0 = time.time()
    gvars.batch_iterator = BatchIter(data_loader, args.cuda)
    batch_iterator_adv = BatchIter(data_loader_adv, args.cuda)

    gvars.inter_iteration=0
    def inter_iter(clean_img, at_img, at_targets):

        if gvars.inter_iteration in cfg['lr_steps']:
            gvars.step_index += 1
            adjust_learning_rate(optimizer, args.gamma, gvars.step_index)

        if args.clean_mix:
            images, targets = gvars.batch_iterator.next()

            optimizer.zero_grad()
            with amp.autocast() if args.amp else Empty():
                out = net(torch.cat((images, at_img), dim=0))
                # backprop
                loss_l, loss_c = criterion(out, targets+at_targets)
                loss = loss_l + loss_c
        else:
            optimizer.zero_grad()
            with amp.autocast() if args.amp else Empty():
                out = net(at_img)
                # backprop
                loss_l, loss_c = criterion(out, at_targets)
                loss = loss_l + loss_c

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        gvars.loc_loss += loss_l.item()
        gvars.conf_loss += loss_c.item()

        if gvars.inter_iteration % 50 == 0:
            gvars.t1 = time.time()
            logger.info('iter ' + repr(gvars.inter_iteration) + ' || Loss: %.4f || timer: %.4f sec.' % (loss.item(), (gvars.t1 - gvars.t0) / 50))
            gvars.t0 = time.time()

        if gvars.inter_iteration != 0 and gvars.inter_iteration % 5000 == 0:
            logger.info(f'Saving state, iter: {gvars.inter_iteration}')
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_{gvars.inter_iteration}.pth'))
        gvars.inter_iteration+=1

    pgd.set_call_back(inter_iter)
    for iteration in range(args.start_iter//pgd.iters, args.max_iter//pgd.iters):
        # load adv train data
        images, targets = batch_iterator_adv.next()

        adv_generator.generate(images, targets)
        # at_img = torch.cat([adv_generator.generate(x.unsqueeze(0), targets) for x in images[images.size(0) // 2:, :, :, :]], dim=0)

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_final_{args.max_iter}.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)


if __name__ == '__main__':
    cfgp = cfgParser()
    args = cfgp.load_cfg(['train'])

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    suffix_dict = {'_ft': args.resume, '_nobase': args.basenet == 'None', '_advonly': not args.clean_mix}
    task_name = f'{args.dataset}_adv3-{args.adv_type}_{args.data_use}{"".join(k for k, v in suffix_dict.items() if v)}'
    logger = get_logger(os.path.join(args.log_folder, f'train_{task_name}.log'))

    # import amp
    if args.amp:
        from torch.cuda import amp

        logger.info('using amp')

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logger.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                           "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    train()
