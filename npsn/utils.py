import math
import random
import torch
import numpy as np


def box_muller_transform(x: torch.FloatTensor):
    r"""Box-Muller transform"""
    shape = x.shape
    x = x.view(shape[:-1] + (-1, 2))
    z = torch.zeros_like(x, device=x.device)
    z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).cos()
    z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).sin()
    return z.view(shape)


def inv_box_muller_transform(z: torch.FloatTensor):
    r"""Inverse Box-Muller transform"""
    shape = z.shape
    z = z.view(shape[:-1] + (-1, 2))
    x = torch.zeros_like(z, device=z.device)
    x[..., 0] = z.square().sum(dim=-1).div(-2).exp()
    x[..., 1] = torch.atan2(z[..., 1], z[..., 0]).div(2 * np.pi).add(0.5)
    return x.view(shape)


def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output."""

    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2, device=V.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov


def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    # Calculate ADEs and FDEs
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]

    # Calculate TCCs
    pred_best = pred[temp[:, -1, :].argmin(dim=0), :, range(pred.size(2)), :]
    pred_gt_stack = torch.stack([pred_best, gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return ADEs, FDEs, TCCs


def evaluate_tcc(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
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


def data_sampler(V_obs, V_tr, A_obs=None, A_tr=None, scale=True, rotation=True, flip=True):
    if scale:
        V_obs, V_tr, A_obs, A_tr = random_scale(V_obs, V_tr, A_obs, A_tr)
    if rotation:
        V_obs, V_tr, A_obs, A_tr = random_rotation(V_obs, V_tr, A_obs, A_tr)
    if flip:
        V_obs, V_tr, A_obs, A_tr = random_flip(V_obs, V_tr, A_obs, A_tr)
    return V_obs, V_tr, A_obs, A_tr


def random_scale(V_obs, V_tr, A_obs, A_tr, min=0.8, max=1.2):
    scale = random.uniform(min, max)
    V_obs[..., -2:] = V_obs[..., -2:] * scale
    V_tr = V_tr * scale
    return V_obs, V_tr, A_obs, A_tr


def random_rotation(V_obs, V_tr, A_obs, A_tr):
    theta = random.uniform(-math.pi, math.pi)
    theta = (theta // (math.pi / 2)) * (math.pi / 2)

    r_mat = [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
    r = torch.tensor(r_mat, dtype=torch.float, requires_grad=False).cuda()

    V_obs[..., -2:] = torch.einsum('rc,ntvc->ntvr', r, V_obs[..., -2:])
    V_tr = torch.einsum('rc,ntvc->ntvr', r, V_tr)
    return V_obs, V_tr, A_obs, A_tr


def random_flip(V_obs, V_tr, A_obs, A_tr):
    if random.random() > 0.5:
        flip = torch.cat([V_obs[..., -2:], V_tr], dim=1)
        flip = torch.flip(flip, dims=[1])
        V_obs[..., -2:] = flip[:, :8]
        V_tr = flip[:, 8:]
        if A_obs is not None:
            flip = torch.cat([A_obs, A_tr], dim=-3)
            flip = torch.flip(flip, dims=[-3])
            A_obs = flip[..., :8, :, :]
            A_tr = flip[..., 8:, :, :]
    return V_obs, V_tr, A_obs, A_tr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
