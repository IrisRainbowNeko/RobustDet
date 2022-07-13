import torch
from torch import nn
from copy import deepcopy
from .base import Attacker
from torch.cuda import amp
from utils.utils import Empty

class FGSM(Attacker):
    def __init__(self, model, img_transform=(lambda x:x, lambda x:x), use_amp=False):
        super().__init__(model, img_transform)
        self.use_amp=use_amp

        if use_amp:
            self.scaler = amp.GradScaler()

    def set_para(self, eps=8, alpha=lambda:8, **kwargs):
        super().set_para(eps=eps, alpha=alpha, **kwargs)

    def step(self, images, labels, loss):
        with amp.autocast() if self.use_amp else Empty():
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels)

        if self.use_amp:
            self.scaler.scale(cost).backward()
        else:
            cost.backward()

        adv_images = (images + self.alpha() * images.grad.sign()).detach_()
        eta = torch.clamp(adv_images - self.ori_images, min=-self.eps, max=self.eps)
        images = self.img_transform[0](torch.clamp(self.img_transform[1](self.ori_images + eta), min=0, max=255).detach_())

        return images

    def attack(self, images, labels):
        #images = deepcopy(images)
        #self.ori_images = deepcopy(images)

        self.model.eval()

        images = self.forward(self, images, labels)

        self.model.zero_grad()
        self.model.train()

        return images