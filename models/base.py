import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from copy import deepcopy
from . import resnet as resnet_base

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, conv_layer, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if isinstance(v, str) and v[0]=='P': # CFR support
                v=int(v[1:])
                conv2d = ProbConv2d(conv_layer(in_channels, v, kernel_size=3, padding=1))
            else:
                conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=not isinstance(conv2d, ProbConv2d))]
            else:
                layers += [conv2d, nn.ReLU(inplace=not isinstance(conv2d, ProbConv2d))]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = conv_layer(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = conv_layer(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def resnet(conv_layer, batch_norm=False):
    resnet_base.conv_layer=conv_layer
    backbone=resnet_base.resnet50()
    models.resnet50(pretrained=conv_layer==nn.Conv2d)

    layers=nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        #backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )

    return layers

class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        return x*self.scale

def decoder():
    modules = []
    modules.append(
        nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            #nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, kernel_size=3),
            nn.Tanh(),
            ScaleLayer(160.)
        )
    )

    return nn.Sequential(*modules)

decoder_large_28 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(512, 512, kernel_size=3),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(256, 256, kernel_size=3),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=3),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(32, 3, kernel_size=3),
    nn.Tanh(),
    ScaleLayer(160.)
)

decoder_large_21 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(256, 256, kernel_size=3),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=3),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(32, 3, kernel_size=3),
    nn.Tanh(),
    ScaleLayer(160.)
)

decoder_large_14 = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(32, 3, kernel_size=3),
    nn.Tanh(),
    ScaleLayer(160.)
)

decoder_large_7 = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(inplace=True),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(32, 3, kernel_size=3),
    nn.Tanh(),
    ScaleLayer(160.)
)

decoder_map={7:decoder_large_7, 14:decoder_large_14, 21:decoder_large_21, 28:decoder_large_28}

def add_extras(cfg, i, conv_layer, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [conv_layer(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [conv_layer(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(conv_layer, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [conv_layer(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [conv_layer(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [conv_layer(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [conv_layer(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def add_extras_resnet(conv_layer, input_size=[1024, 512, 512, 256, 256, 256]):
    # Extra layers added to VGG for feature scaling
    additional_blocks = []  # 存放额外卷积层的列表
    # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
    middle_channels = [256, 256, 128, 128, 128]
    # input_size[:-1]=[1024, 512, 512, 256, 256], input_size[1:]=[512, 512, 256, 256, 256]
    for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
        padding, stride = (1, 2) if i < 3 else (0, 1)
        layer = nn.Sequential(
            conv_layer(input_ch, middle_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            conv_layer(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )
        additional_blocks.append(layer)
    return additional_blocks

def multibox_resnet(conv_layer, resnet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    loc_layers += [conv_layer(1024, cfg[0] * 4, kernel_size=3, padding=1, bias=False)]
    conf_layers += [conv_layer(1024, cfg[0] * num_classes, kernel_size=3, padding=1, bias=False)]
    for k, v in enumerate(extra_layers, 1):
        loc_layers += [conv_layer(v[-3].out_channels, cfg[k] * 4, kernel_size=3, padding=1, bias=False)]
        conf_layers += [conv_layer(v[-3].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1, bias=False)]
    return resnet, extra_layers, (loc_layers, conf_layers)

def discriminator(k=4, fc=True):
    return Discriminator(models.resnet18(pretrained=True, progress=False), k, fc=fc)

class Discriminator(nn.Module):
    def __init__(self, base_net, k=4, fc=True):
        super().__init__()
        self.base_net=base_net
        if hasattr(self.base_net,'fc'):
            self.base_net.fc=nn.Linear(base_net.fc.in_features, k) if fc else nn.Sequential()
        else:
            self.base_net.classifier=nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, k),
            )
        self.fc=fc

    def forward(self, x):
        x = self.base_net(x)
        if self.fc:
            x = F.softmax(x, dim=-1)
        return x

class ProbConv2d(nn.Module):
    def __init__(self, conv_layer, eps=1e-8):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        if hasattr(conv_layer,'K'):
            self.K=conv_layer.K

        self.conv_mean=conv_layer
        self.conv_std=deepcopy(conv_layer)
        self.eps=eps

    def forward(self, x, softmax_attention=None):
        if softmax_attention is None:
            x_mean=self.conv_mean(x)
            x_log_var=self.conv_std(x)
        else:
            x_mean = self.conv_mean(x, softmax_attention)
            x_log_var = self.conv_std(x, softmax_attention)
        #x_log_var = torch.where(torch.isinf(x_log_var.exp()), torch.full_like(x_log_var, 0), x_log_var)
        x_log_var = x_log_var.clip(max=10)
        return self.reparameterize(x_mean, x_log_var)

    # 随机生成隐含向量
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar