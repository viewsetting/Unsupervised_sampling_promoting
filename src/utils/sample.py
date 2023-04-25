# modified from https://github.com/InhwanBae/NPSN/blob/main/npsn/sampler.py
import torch
import numpy as np

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

def mc_sample_fast(mu, cov, n_sample):
    r_sample = torch.randn((n_sample,) + mu.shape, dtype=mu.dtype, device=mu.device)
    sample = mu + (torch.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample

def qmc_sample_fast(mu, cov, n_sample, rng=None):
    rng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)
    qr_seq = torch.stack([box_muller_transform(rng.draw(n_sample)) for _ in range(mu.size(0))], dim=1).unsqueeze(dim=2).type_as(mu)
    sample = mu + (torch.cholesky(cov) @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample

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
