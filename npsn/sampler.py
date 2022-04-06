import torch
from torch.distributions import MultivariateNormal
from .utils import box_muller_transform


def mc_sample(mu, cov, n_sample):
    mv_norm = MultivariateNormal(mu, cov)
    sample = mv_norm.sample((n_sample,))
    return sample


def qmc_sample(mu, cov, n_sample, rng):
    mv_norm = MultivariateNormal(mu, cov)
    qr_seq = torch.stack([box_muller_transform(rng.draw(n_sample)) for _ in range(mu.size(0))], dim=1).unsqueeze(dim=2).type_as(mu)
    sample = mv_norm.loc + (mv_norm._unbroadcasted_scale_tril @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def purposive_sample(mu, cov, n_sample, loc_sample):
    mv_norm = MultivariateNormal(mu, cov)
    loc_norm = box_muller_transform(loc_sample).permute(2, 0, 1, 3).expand((n_sample,) + mu.shape)
    sample = mv_norm.loc + (mv_norm._unbroadcasted_scale_tril @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def mc_sample_fast(mu, cov, n_sample):
    r_sample = torch.randn((n_sample,) + mu.shape, dtype=mu.dtype, device=mu.device)
    sample = mu + (torch.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def qmc_sample_fast(mu, cov, n_sample, rng):
    qr_seq = torch.stack([box_muller_transform(rng.draw(n_sample)) for _ in range(mu.size(0))], dim=1).unsqueeze(dim=2).type_as(mu)
    sample = mu + (torch.cholesky(cov) @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def purposive_sample_fast(mu, cov, n_sample, loc_sample):
    loc_norm = box_muller_transform(loc_sample).permute(2, 0, 1, 3).expand((n_sample,) + mu.shape)
    sample = mu + (torch.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample
