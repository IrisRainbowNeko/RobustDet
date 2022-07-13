import time
import logging
import sys
import numpy as np
import torch


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

class BatchIter:
    def __init__(self, loader, cuda):
        self.batch_iterator = iter(loader)
        self.batch_size=loader.batch_size
        self.loader = loader
        self.cuda = cuda

    def next(self):
        try:
            images, targets = next(self.batch_iterator)
        except StopIteration:
            self.batch_iterator = iter(self.loader)
            images, targets = next(self.batch_iterator)

        if self.cuda:
            images = images.cuda()
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
        else:
            images = images
            with torch.no_grad():
                targets = [ann for ann in targets]
        return images, targets

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(levelname)s] <%(asctime)s> %(message)s",
        datefmt='%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name if name else filename)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def getIdx(a):
    co = a.unsqueeze(0)-a.unsqueeze(1)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    return out

class Empty:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass