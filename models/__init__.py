from .base import *
from .ssd import *
from .robust_ssd import *

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def build_check(phase, size):
    if phase not in ("test", "train"):
        raise Exception(f'ERROR: Phase: {phase} not recognized')
    if size != 300:
        raise Exception(f'ERROR: You specified size {size}. However, currently only SSD300 (size=300) is supported!')

def build_ssd(phase, size=300, num_classes=21, amp=False, backbone='vgg'):
    build_check(phase, size)
    conv_layer=nn.Conv2d
    if backbone=='vgg':
        base_, extras_, head_ = multibox(conv_layer, vgg(base[str(size)], 3, conv_layer),
                                         add_extras(extras[str(size)], 1024, conv_layer),
                                         mbox[str(size)], num_classes)
    elif backbone=='resnet':
        base_, extras_, head_ = multibox_resnet(conv_layer, resnet(conv_layer),
                                         add_extras_resnet(conv_layer),
                                         mbox[str(size)], num_classes)
    return (SSD_amp if amp else SSD)(phase, size, base_, extras_, head_, num_classes, backbone=backbone)

def build_robust_ssd(phase, size=300, num_classes=21, amp=False, CFR=False, CFR_layer=21, multi_fc=False, K=4, backbone='vgg'):
    build_check(phase, size)
    conv_layer = lambda *args, **kwargs: DynamicConv2d(*args, use_FC=multi_fc, K=K, **kwargs)
    if backbone == 'vgg':
        base_, extras_, head_ = multibox(conv_layer, vgg(base[str(size)], 3, conv_layer),
                                         add_extras(extras[str(size)], 1024, conv_layer),
                                         mbox[str(size)], num_classes)
    elif backbone=='resnet':
        base_, extras_, head_ = multibox_resnet(conv_layer, resnet(conv_layer),
                                         add_extras_resnet(conv_layer),
                                         mbox[str(size)], num_classes)
    if CFR:
        base_[CFR_layer]=ProbConv2d(base_[CFR_layer])
        return (RobustSSD_amp if amp else RobustSSD)(phase, size, base_, extras_, head_,
                        discriminator(fc=not multi_fc, k=K), num_classes, decoder=decoder_map[CFR_layer],
                        backbone=backbone, CFR_layer=CFR_layer)
    else:
        return (RobustSSD_amp if amp else RobustSSD)(phase, size, base_, extras_, head_,
                        discriminator(fc=not multi_fc, k=K), num_classes, backbone=backbone)