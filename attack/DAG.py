import torch
from torch import nn
from copy import deepcopy
from .base import Attacker
from torch.cuda import amp
from utils.utils import Empty
from layers.box_utils import match
import random
from torch.nn import functional as F

class DAG(Attacker):
    def __init__(self, model, img_transform=(lambda x:x, lambda x:x)):
        super().__init__(model, img_transform)
        self.call_back=None
        self.img_loader=None

    def set_para(self, eps=8, gamma=lambda:0.5, iters=150, **kwargs):
        super().set_para(eps=eps, gamma=gamma, iters=iters, **kwargs)

    def set_call_back(self, call_back):
        self.call_back=call_back

    def set_img_loader(self, img_loader):
        self.img_loader=img_loader

    def step(self, images, labels, loss):
        adv_labels, target_boxes, target_labels = labels
        images.requires_grad = True

        self.model.zero_grad()
        predictions = self.model(images)
        loc_data, conf_data, priors = predictions

        logits = conf_data.view(-1, loss.num_classes)
        active_cond = logits.argmax(dim=1) != adv_labels

        target_boxes = target_boxes[active_cond]
        logits = logits[active_cond]
        target_labels = target_labels[active_cond]
        adv_labels = adv_labels[active_cond]

        target_loss = F.cross_entropy(logits, target_labels, reduction="sum")
        adv_loss = F.cross_entropy(logits, adv_labels, reduction="sum")
        total_loss = target_loss - adv_loss
        total_loss.backward()
        image_grad = images.grad.detach()

        with torch.no_grad():
            image_perturb = (self.gamma() / image_grad.norm(float("inf"))) * image_grad
            images = self.img_transform[0](torch.clamp(self.img_transform[1](self.ori_images + image_perturb), min=0, max=255).detach_())

        return images, image_perturb

    def attack(self, images, targets):
        num = images.size(0)
        priors = self.model.priors
        num_priors = (priors.size(0))

        conf_t = torch.LongTensor(num, num_priors)
        loc_t = torch.Tensor(num, num_priors, 4)
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors
            match(self.loss.threshold, truths, defaults, self.loss.variance, labels,
                  loc_t, conf_t, idx)
        conf_t = conf_t.cuda()
        loc_t = loc_t.cuda()
        target_boxes = loc_t.view(-1, 4)
        target_labels = conf_t.view(-1)
        adv_labels = self.get_adv_labels(target_labels)

        #images = deepcopy(images)
        self.ori_images = deepcopy(images)

        self.model.eval()
        sum_perturb = torch.zeros_like(images)
        for i in range(self.iters):
            images, step_perturb = self.forward(self, images, (adv_labels, target_boxes, target_labels))
            sum_perturb += step_perturb

        self.model.train()

        images = self.img_transform[0](torch.clamp(self.img_transform[1](self.ori_images + sum_perturb), min=0, max=255).detach_())

        return images

    def get_adv_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Assign adversarial labels to a set of correct labels,
        i.e. randomly sample from incorrect classes.
        Parameters
        ----------
        labels : torch.Tensor
            [n_targets]
        Returns
        -------
        torch.Tensor
            [n_targets]
        """
        adv_labels = torch.zeros_like(labels)

        for i,la in enumerate(labels):
            incorrect_labels = [l for l in range(self.loss.num_classes) if l != la]
            adv_labels[i] = random.choice(incorrect_labels)

        return adv_labels.cuda()