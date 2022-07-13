from layers.modules.multibox_loss import *
from utils import getIdx

class ClassWiseLoss(MultiBoxLoss):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining,
                 neg_pos, neg_overlap, encode_target, use_gpu=True):
        super().__init__(num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining,
                         neg_pos, neg_overlap, encode_target, use_gpu)

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='none').sum(dim=-1)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[pos_idx].view(-1, self.num_classes)
        targets_weighted = conf_t[pos]
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        loss_sum = loss_c+loss_l
        cls_idx_group=getIdx(targets_weighted)
        loss = [loss_sum[cidx].mean() for cidx in cls_idx_group]
        loss = sum(loss)/len(loss)

        return loss


