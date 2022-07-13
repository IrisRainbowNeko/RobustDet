from .base import BaseAdvDataGenerator
from copy import deepcopy
import torch

class MTD(BaseAdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker)
        self.criterion = criterion
        self.loss_loc=lambda out, la: self.criterion(out, la)[0]
        self.loss_cls=lambda out, la: self.criterion(out, la)[1]

        def mtd_forward(attacker, images, labels):
            img_loc=attacker.step(deepcopy(images), labels, self.loss_loc)
            img_cls=attacker.step(images, labels, self.loss_cls)
            with torch.no_grad():
                img=img_loc if sum(self.criterion(attacker.model(img_loc), labels)) > sum(
                    self.criterion(attacker.model(img_cls), labels)) else img_cls
            return img

        self.attacker.set_forward(mtd_forward)