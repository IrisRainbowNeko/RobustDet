from data import *
from data.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models import *
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
from triplet import OnlineTripletLoss, pdist_js, pdist_l2
from dconv import *
from torchsummary import summary
from utils.cfgParser import cfgParser

from attack import *
from robust import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def train():
    if args.dataset == 'COCO':
        cfg = coco
        image_sets = process_names_coco(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = COCODetection(root=args.dataset_root, image_sets=image_sets,
                                transform=SSDAugmentation(cfg['min_dim'], ))
    elif args.dataset == 'VOC':
        cfg = voc
        image_sets=process_names_voc(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = VOCDetection(root=args.dataset_root, image_sets=image_sets,
                               transform=SSDAugmentation(cfg['min_dim']))

    ssd_net = build_robust_ssd('train', cfg['min_dim'], cfg['num_classes'], amp=args.amp and args.multi_gpu,
                               CFR=args.cfr, CFR_layer=args.cfr_layer, multi_fc=args.multi_fc, K=args.k_count, backbone=args.backbone)
    net = ssd_net
    summary(net, (3, 300, 300))

    if args.cuda:
        if args.multi_gpu:
            net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.cfr:
        ssd_net.decoder.apply(weights_init)

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.basenet!='None':
        '''vgg_weights = torch.load(args.save_folder + args.basenet)
        logger.info('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)'''
        logger.info('Loading regular CNN weights...')
        ssd_net.load_weights_expand(args.basenet, weights_init_prob if args.cfr_layer<=14 else weights_init)

    if args.cuda:
        net = net.cuda()

    if not args.resume and args.basenet=='None':
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.vgg.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_mlb = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda, use_focal=args.focal)
    criterion_triplet = OnlineTripletLoss(0.6, pdist=pdist_l2 if args.multi_fc else pdist_js)

    net.train()
    #print(net)
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info(f'Training SSD on: {dataset.name}')
    logger.info('Using the specified args:')
    logger.info(repr(args))

    step_index = 0
    scaler = amp.GradScaler(enabled=args.amp)

    if torch.__version__.startswith('1.9'):
        data_loader = data.DataLoader(dataset, args.batch_size // 2,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True, generator=torch.Generator(device='cuda'))
        data_loader_adv = data.DataLoader(dataset, args.batch_size // 2,
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True, drop_last=True, generator=torch.Generator(device='cuda'))
    else:
        data_loader = data.DataLoader(dataset, args.batch_size // 2,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True)
        data_loader_adv = data.DataLoader(dataset, args.batch_size // 2,
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True, drop_last=True)

    #criterion_mlb = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    criterion_clsw = ClassWiseLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    dataset_mean_t = torch.tensor(DATASET_MEANS).view(1, -1, 1, 1)
    pgd = PGD(net, img_transform=(lambda x: x - dataset_mean_t, lambda x: x + dataset_mean_t))
    pgd.set_para(eps=args.step_size, alpha=lambda:args.step_size, iters=args.atk_iters)
    adv_dict = {'cls': (CLS_ADG, criterion_mlb), 'loc': (LOC_ADG, criterion_mlb), 'con': (CON_ADG, criterion_mlb),
                'mtd': (MTD, criterion_mlb), 'cwat': (CWAT, criterion_clsw)}
    adv_item = adv_dict[args.adv_type.lower()]
    adv_generator = adv_item[0](pgd, adv_item[1])

    # create batch iterator
    t0 = time.time()
    batch_iterator = BatchIter(data_loader, args.cuda)
    batch_iterator_adv = BatchIter(data_loader_adv, args.cuda)

    adv_label = torch.LongTensor(([0]*(args.batch_size//2))+([1]*(args.batch_size//2))).cuda()
    sum_loss=0

    if args.adc==0:
        args.adc=args.batch_size//2

    if args.adc > 0:
        adv_weights=torch.FloatTensor([1]*(data_loader_adv.batch_size+data_loader.batch_size)+[data_loader_adv.batch_size/args.adc]*(args.adc))
        adv_weights=adv_weights.unsqueeze(-1).expand(adv_weights.size(0), ssd_net.priors.size(0)).contiguous().cuda()

    inter_iteration = 0
    for iteration in range(args.start_iter//pgd.iters, args.max_iter//pgd.iters):
        # load adv train data
        images_adv, targets_adv = batch_iterator_adv.next()

        pgd.set_data(images_adv, targets_adv)
        for im_clean, at_img, at_targets in adv_generator:
            if inter_iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            images, targets = batch_iterator.next()

            # forward
            optimizer.zero_grad()
            with amp.autocast(enabled=args.amp):
                if args.adc > 0:
                    adv_pred = net.module.disc(torch.cat((images, at_img)))
                    out1 = net(torch.cat((images, at_img), dim=0), adv_pred=adv_pred)
                    if args.cfr:
                        recons1 = net.recons
                        logvar1 = net.logvar
                        mu1 = net.mu
                    loss_tri, _ = criterion_triplet(adv_pred, adv_label)

                    pred_adv = adv_pred.detach()[images.size(0):, :]
                    sample_idx = sorted(random.sample(range(at_img.size(0)), args.adc))
                    img_sample = images[torch.tensor(sample_idx)]
                    targets_sample = [targets[x] for x in sample_idx]
                    out2 = net(img_sample, adv_pred=pred_adv[torch.tensor(sample_idx)])
                    if args.cfr:
                        recons2 = net.recons
                        logvar2 = net.logvar
                        mu2 = net.mu
                    out = (torch.cat((out1[0], out2[0]), dim=0), torch.cat((out1[1], out2[1]), dim=0), out1[2])
                else:
                    adv_pred = net.module.disc(torch.cat((images, at_img)))
                    out = net(torch.cat((images, at_img), dim=0), adv_pred=adv_pred)
                    loss_tri, _ = criterion_triplet(adv_pred, adv_label)

                # backprop
                loss_l, loss_c = criterion_mlb(out, targets + at_targets + targets_sample if args.adc > 0 else targets + at_targets,
                                                   weights=adv_weights if args.adc > 0 else None)
                loss = (loss_l + loss_c + loss_tri * 3) * 0.75

                if args.cfr:
                    recons_loss = F.mse_loss(torch.cat((images, im_clean, img_sample), dim=0),
                                             torch.cat((recons1, recons2), dim=0))
                    logvar = torch.cat((logvar1, logvar2), dim=0)
                    mu = torch.cat((mu1, mu2), dim=0)
                    kld_loss = torch.mean(-0.5 * torch.mean((1 + logvar - mu ** 2 - logvar.exp()).view(images.size(0), -1), dim=1), dim=0)
                    #kld_loss = torch.mean(-0.5 * torch.mean((1 + logvar - logvar.exp()).view(images.size(0), -1), dim=1), dim=0)
                    loss = loss + (recons_loss * 0.16 + kld_loss * 5) / 255.0
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            sum_loss += loss.item()

            if inter_iteration % 50 == 0:
                t1 = time.time()
                if args.cfr:
                    logger.info('iter ' + repr(inter_iteration) + ' || Loss: %.4f || recons_loss: %.4f || kld_loss: %.4f || timer: %.4f sec.' %
                                (sum_loss / 50, recons_loss.item(), kld_loss.item(), (t1 - t0) / 50))
                else:
                    logger.info('iter ' + repr(inter_iteration) + ' || Loss: %.4f || timer: %.4f sec.' % (
                    sum_loss / 50, (t1 - t0) / 50))
                t0 = time.time()
                sum_loss = 0

            if inter_iteration != 0 and inter_iteration % 5000 == 0:
                logger.info(f'Saving state, iter: {inter_iteration}')
                torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_{inter_iteration}.pth'))
            inter_iteration += 1

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
    if isinstance(m, nn.Conv2d) or isinstance(m, DynamicConv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def weights_init_prob(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, DynamicConv2d):
        init.xavier_uniform_(m.weight, gain=1./256)
        if m.bias is not None:
            init.zeros_(m.bias)

if __name__ == '__main__':
    cfgp = cfgParser()
    args=cfgp.load_cfg(['train'])

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    suffix_dict = {'_ft': args.resume, '_nobase': args.basenet == 'None', f'_rec{args.cfr_layer}': args.cfr,
                   '_focal': args.focal, '_mlfc': args.multi_fc, '_resnet': args.backbone == 'resnet'}
    task_name = f'{args.dataset}_adv-{args.adv_type}_dconv{args.k_count}_step{args.step_size}' \
                f'_{args.data_use}{"".join(k for k, v in suffix_dict.items() if v)}'
    logger = get_logger(os.path.join(args.log_folder, f'train_{task_name}.log'))

    logger.info(repr(args))

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
