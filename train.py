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
from utils.utils import get_logger, Empty

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--data_use', default="trainval:clean",
                    type=str, help='datas use for training')
parser.add_argument('--backbone', default='vgg', type=str,
                    help='Pretrained base model')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', type=str,
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--multi_gpu', default=True, type=str2bool,
                    help='Use multiple GPU to train model')
parser.add_argument('--amp', default=False, type=str2bool,
                    help='Use automatic mixed precision for training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--focal', default=False, type=str2bool,
                    help='Use vae for training')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/', type=str,
                    help='Directory for saving checkpoint weights')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for logs')
args = parser.parse_args()

if not os.path.exists(args.log_folder):
    os.mkdir(args.log_folder)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

suffix_dict={'_ft':args.resume, '_nobase':args.basenet=='None', '_resnet':args.backbone=='resnet'}
task_name=f'{args.dataset}_{args.data_use}{"".join(k for k,v in suffix_dict.items() if v)}'
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
                               transform=SSDAugmentation(cfg['min_dim']), load_sizes=True)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], amp=args.amp and args.multi_gpu, backbone=args.backbone)
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
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info(f'Training SSD on: {dataset.name}')
    logger.info('Using the specified args:')
    logger.info(repr(args))

    step_index = 0
    if args.amp:
        scaler = amp.GradScaler()

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    t0 = time.time()
    loss_sum=0
    batch_iterator = BatchIter(data_loader, args.cuda)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = batch_iterator.next()

        # forward
        optimizer.zero_grad()
        with amp.autocast() if args.amp else Empty():
            out = net(images)
            # backprop
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        loss_sum += loss.item()

        if iteration % 50 == 0:
            t1 = time.time()
            logger.info('iter ' + repr(iteration) + ' || Loss: %.4f || timer: %.4f sec.' % (loss_sum/50, (t1 - t0)/50))
            loss_sum=0
            t0 = time.time()

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            logger.info(f'Saving state, iter: {iteration}')
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_{iteration}.pth'))

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_final_{cfg["max_iter"]}.pth'))


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
        if m.bias is not None:
            init.zeros_(m.bias)

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
