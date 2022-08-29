import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc, coco
import os
from dconv import *
from torch.cuda import amp
from models import *

ssd_adv_pred = {}

class RobustSSD(nn.Module):
    """Modified from Single Shot Multibox Architecture

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, disc, num_classes, dconv_K=4, decoder=None, backbone='vgg', CFR_layer=21):
        super(RobustSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size
        self.backbone = backbone

        # SSD network
        self.vgg = nn.ModuleList(base) if backbone == 'vgg' else base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.disc = disc
        self.dconv_K=dconv_K
        self.decoder=decoder
        self.CFR_layer=CFR_layer

        #input dynamic convolution weights to each DynamicConv2d layer via hook
        for layer in self.modules():
            if isinstance(layer, DynamicConv2d) or (isinstance(layer, ProbConv2d) and isinstance(layer.conv_mean, DynamicConv2d)):
                layer.register_forward_pre_hook(lambda module, x:(x[0], ssd_adv_pred[x[0].device]))

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            Detect.set(num_classes, 0, 200, 0.01, 0.45)

    @property
    def adv_pred(self):
        return ssd_adv_pred[self.loc[0].weight.device]

    def forward(self, x, adv_pred=None):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                tensor of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        adv_pred=adv_pred if adv_pred is not None else self.disc(x) #AID预测动态卷积权重
        ssd_adv_pred[adv_pred.device] = adv_pred

        if self.backbone == 'vgg':
            # apply vgg up to conv4_3 relu
            for k in range(23):
                if k==self.CFR_layer and self.decoder is not None:
                    z, mu, logvar = self.vgg[k](x)
                    self.mu=mu
                    self.logvar=logvar
                    if self.training:
                        self.recons = self.decoder(z)
                        x=z
                    else:
                        x=mu
                else:
                    x = self.vgg[k](x)
                x=torch.where(torch.isnan(x), torch.full_like(x, 0), x)

            s = self.L2Norm(x)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                if k==self.CFR_layer and self.decoder is not None:
                    z, mu, logvar = self.vgg[k](x)
                    self.mu=mu
                    self.logvar=logvar
                    if self.training:
                        self.recons = self.decoder(z)
                        x=z
                    else:
                        x=mu
                else:
                    x = self.vgg[k](x)
            sources.append(x)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)
        elif self.backbone=='resnet':
            x=self.vgg(x)
            sources.append(x)
            for v in self.extras:
                x=v(x)
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = Detect.apply(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(x.type())                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ['.pkl', '.pth']:
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_weights_expand(self, base_file, weights_init=None):
        '''
        Conv2d:              conv
                          ↙ ↓ ↓ ↘
        DynamicConv2d:  c1  c2 c3 .. cn

        Conv:         conv      init
                       ↓        ↓
        ProbConv:  conv_mean  conv_std
        :param base_file: Model weights using regular convolution
        '''
        other, ext = os.path.splitext(base_file)
        if ext in ['.pkl', '.pth']:
            print('Loading Conv weights expand into DynamicConv layer...')
            self_dict=self.state_dict()
            module_dict=dict(self.named_modules())

            if weights_init is not None:
                for k,layer in module_dict.items():
                    if isinstance(layer, ProbConv2d):
                        layer.apply(weights_init)

            load_dict=torch.load(base_file, map_location=lambda storage, loc: storage)
            state_dict = {}
            for k, v in load_dict.items():
                if k in self_dict.keys() or k.startswith('vgg'):
                    layer=module_dict[k[:k.rfind('.')]]
                    if isinstance(layer, DynamicConv2d) or (isinstance(layer, ProbConv2d) and isinstance(layer.conv_mean, DynamicConv2d)):
                        v = v.unsqueeze(0).repeat([layer.K] + torch.ones(len(v.shape), dtype=int).tolist())
                    if isinstance(layer, ProbConv2d):
                        state_dict[k[:k.rfind('.')+1]+'conv_mean'+k[k.rfind('.'):]]=v
                        if weights_init is not None:
                            state_dict[k[:k.rfind('.')+1]+'conv_std'+k[k.rfind('.'):]]=layer.conv_std.state_dict()[k[k.rfind('.')+1:]]
                    else:
                        state_dict[k]=v

            self_dict.update(state_dict)
            self.load_state_dict(self_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class RobustSSD_amp(RobustSSD):
    def forward(self, x, adv_pred=None):
        with amp.autocast():
            output = super().forward(x, adv_pred=adv_pred)
        return output