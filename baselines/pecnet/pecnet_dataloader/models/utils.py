import pickle
import torch
import numpy as np
from torch.utils import data


def initial_pos(traj_batches):
    batches = []
    for b in traj_batches:
        starting_pos = b[:, 7, :].copy() / 1000  # starting pos is end of past, start of future. scaled down.
        batches.append(starting_pos)

    return batches


class SocialDataset(data.Dataset):
    def __init__(self, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=True):
        'Initialization'
        load_name = "./social_pool_data/{0}_{1}{2}_{3}_{4}.pickle".format(set_name, 'all_' if scene is None else scene[:-2] + scene[-1] + '_', b_size, t_tresh, d_tresh)
        print(load_name)
        with open(load_name, 'rb') as f:
            data = pickle.load(f)

        traj, masks = data
        traj_new = []

        if id == False:
            for t in traj:
                t = np.array(t)
                t = t[:, :, 2:]
                traj_new.append(t)
                if set_name == "train":
                    # augment training set with reversed tracklets...
                    reverse_t = np.flip(t, axis=1).copy()
                    traj_new.append(reverse_t)
        else:
            for t in traj:
                t = np.array(t)
                traj_new.append(t)

                if set_name == "train":
                    # augment training set with reversed tracklets...
                    reverse_t = np.flip(t, axis=1).copy()
                    traj_new.append(reverse_t)

        masks_new = []
        for m in masks:
            masks_new.append(m)

            if set_name == "train":
                # add second time for the reversed tracklets...
                masks_new.append(m)

        traj_new = np.array(traj_new)
        masks_new = np.array(masks_new)
        self.trajectory_batches = traj_new.copy()
        self.mask_batches = masks_new.copy()
        self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches))  # for relative positioning
        if verbose:
            print("Initialized social dataloader...")


def box_muller_transform(x: torch.FloatTensor):
    shape = x.shape
    x = x.view(shape[:-1] + (-1, 2))
    z = torch.zeros_like(x, device=x.device)
    z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).cos()
    z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).sin()
    return z.view(shape)


def inv_box_muller_transform(z: torch.FloatTensor):
    shape = z.shape
    z = z.view(shape[:-1] + (-1, 2))
    x = torch.zeros_like(z, device=z.device)
    x[..., 0] = z.square().sum(dim=-1).div(-2).exp()
    x[..., 1] = torch.atan2(z[..., 1], z[..., 0]).div(2 * np.pi).add(0.5)
    return x.view(shape)


def evaluate_tcc(pred, gt):
    """Get TCC scores for each pedestrian"""
    pred, gt = torch.FloatTensor(pred).permute(1, 0, 2), torch.FloatTensor(gt).permute(1, 0, 2)
    pred_best = pred
    pred_gt_stack = torch.stack([pred_best.permute(1, 0, 2), gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef.clip_(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return TCCs
