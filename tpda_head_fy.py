import math
from typing import Tuple

import torch
from torch import nn


def tau_from_cov_torch(cov: torch.Tensor) -> torch.Tensor:
    """Compute TPDA tau for each covariance matrix in cov: (..., D, D)."""
    dim = cov.shape[-1]
    det_term = torch.det(cov)
    numerator = math.gamma(dim / 2.0 + 2.0)
    denom = torch.sqrt((2.0 * torch.pi) ** dim * det_term)
    return -torch.pow(numerator / denom, 2.0 / (dim + 2.0))


def build_spherical_covariances(
    log_cov: torch.Tensor, feature_dim: int, min_scale: float = 1e-4
) -> torch.Tensor:
    """Build per-class spherical covariances Sigma_k = sigma_k^2 I."""
    var = torch.exp(log_cov) + min_scale  # (K,)
    eye = torch.eye(feature_dim, dtype=var.dtype, device=var.device).unsqueeze(0)
    return var[:, None, None] * eye


class TPDAHead(nn.Module):
    """
    TPDA classification head with trainable means/priors and fixed spherical covariances.

    Output scores are class-weighted truncated parabola densities:
        s_c(x) = pi_c * relu(-tau_c - 0.5 * Q_c(x))
    """

    def __init__(self, num_classes: int, feature_dim: int = 2):
        super().__init__()
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive (got {feature_dim}).")
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.min_scale = 1e-4

        dtype = torch.get_default_dtype()
        # Match DNLL/LDAHead initialization.
        base_scale = 6.0 / math.sqrt(2.0 * feature_dim)
        self.mu = nn.Parameter(torch.randn(num_classes, feature_dim, dtype=dtype) * base_scale)
        # Fixed (non-trainable) per-class log-variance for spherical covariances.
        self.register_buffer("log_cov", torch.zeros(num_classes, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def covariances(self) -> torch.Tensor:
        return build_spherical_covariances(
            self.log_cov, feature_dim=self.feature_dim, min_scale=self.min_scale
        )

    def priors(self) -> torch.Tensor:
        return torch.softmax(self.prior_logits, dim=0)

    def mahalanobis(self, x: torch.Tensor, cov: torch.Tensor = None) -> torch.Tensor:
        del cov  # Unused for spherical form; kept for API compatibility.
        delta = x[:, None, :] - self.mu[None, :, :]  # (B,K,D)
        var = torch.exp(self.log_cov).to(dtype=x.dtype, device=x.device) + self.min_scale
        return (delta * delta).sum(dim=-1) / var[None, :]

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cov = self.covariances()
        tau = tau_from_cov_torch(cov)
        pi = self.priors()
        maha = self.mahalanobis(x, cov=cov)
        densities = torch.relu(-tau[None, :] - 0.5 * maha)
        scores = pi[None, :] * densities
        if return_aux:
            return scores, maha, tau, pi
        return scores


def tpda_joint_fy_loss(head: TPDAHead, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Attached joint Fenchel-Young loss used in the TPDA notebook:

      l_i = 1/2 - (2/(N+4)) * sum_c pi_c^2 * tau_c
            + pi_{y_i} * (1/2 * Q_{y_i}(x_i) + tau_{y_i})
    """
    _, maha, tau, pi = head(x, return_aux=True)
    dim = x.shape[1]

    term_global = 0.5 - (2.0 / (dim + 4.0)) * torch.sum((pi**2) * tau)
    tau_y = tau.gather(0, y)
    pi_y = pi.gather(0, y)
    q_y = maha.gather(1, y[:, None]).squeeze(1)
    return term_global + pi_y * (0.5 * q_y + tau_y)
