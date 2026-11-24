"""Compact simplex q-Gaussians: head, FY loss, projected CF matching, and sampling."""

import math

import torch
from torch import nn


class SimplexQGaussianHead(nn.Module):
    """Fixed simplex q-Gaussians for finite q < 1; only priors are learned.

    p(z|c) = [(1-q) * (-tau - Q_c(z)/2)]_+^(1/(1-q)).
    q=0 is TP; q=0.5 is biweight. Sigma=variance*I is the quadratic shape.
    The default shape gives moment covariance I/D for every q.
    Forward returns joint densities. Use qgaussian_joint_fy_loss for training.
    """

    def __init__(self, C: int, D: int, s: float = 6.0,
                 variance: float | None = None, *, q: float = 0.0):
        super().__init__()
        if not isinstance(C, int) or isinstance(C, bool) or C < 2:
            raise ValueError("C must be an integer of at least 2.")
        if not isinstance(D, int) or isinstance(D, bool) or D < 1:
            raise ValueError("D must be a positive integer.")
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        if not math.isfinite(s) or s <= 0:
            raise ValueError("s must be finite and positive.")
        self.q = float(q)
        if not math.isfinite(self.q) or self.q >= 1:
            raise ValueError("q must be finite and less than 1 (compact support).")
        self.alpha = 2 - self.q
        self.power = 1 / (1 - self.q)
        if variance is None:
            # For density [(-tau - Q/2)/p]_+^p, choose moment covariance I/D.
            # Log space avoids intermediate gamma and power overflow.
            p = self.power
            log_variance = (
                0.5 * D * math.log(math.pi) + math.lgamma(p + 1)
                - p * math.log(2 * p) - math.lgamma(0.5 * D + p + 1)
                + (0.5 * D + p) * math.log1p((2 * p + 2) / D)
            ) / p
            variance = math.exp(log_variance)
        else:
            variance = float(variance)
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("variance must be finite and positive.")

        self.C = C
        self.D = D
        dtype = torch.get_default_dtype()
        variance = torch.tensor(variance, dtype=dtype)
        if not torch.isfinite(variance) or variance <= 0:
            raise ValueError(
                "variance must be finite and positive in the default torch dtype; "
                "use a higher-precision default dtype or an explicit variance."
            )
        mu = self._regular_simplex_vertices(C, D, dtype)
        mu = mu * (s / math.sqrt(2.0 * C / (C - 1)))
        self.register_buffer("mu", mu)
        self.register_buffer("variance", variance)
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @staticmethod
    def _regular_simplex_vertices(C, D, dtype):
        """Construct unit-norm simplex vertices centered at the origin."""
        eye = torch.eye(C, dtype=dtype)
        centered = eye - eye.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        simplex = centered @ vh.transpose(-2, -1)[:, : C - 1]
        simplex = simplex / simplex.norm(dim=1, keepdim=True)
        if D > C - 1:
            simplex = torch.cat([simplex, torch.zeros(C, D - C + 1, dtype=dtype)], dim=1)
        return simplex

    @property
    def cov_diag(self) -> torch.Tensor:
        """Diagonal of the fixed shape matrix, with shape (D,)."""
        return self.variance.expand(self.D)

    @property
    def covariance(self) -> torch.Tensor:
        """Shared fixed spherical shape matrix, with shape (D, D)."""
        return torch.eye(self.D, dtype=self.mu.dtype, device=self.mu.device) * self.variance

    def covariances(self) -> torch.Tensor:
        """Return the fixed shape matrix for every class: (C, D, D)."""
        return self.covariance.unsqueeze(0).expand(self.C, -1, -1)

    @property
    def support_radius(self) -> torch.Tensor:
        """Radius of each class's spherical support."""
        return (-2 * self.tau()[0] * self.variance).sqrt()

    @property
    def moment_variance(self) -> torch.Tensor:
        """Actual per-coordinate variance, distinct from the shape scale."""
        return -2 * self.tau()[0] * self.variance / (self.D + 2 * self.power + 2)

    @property
    def moment_covariance(self) -> torch.Tensor:
        """Actual within-class covariance of the normalized q-Gaussian distribution."""
        return torch.eye(self.D, dtype=self.mu.dtype, device=self.mu.device) * self.moment_variance

    def priors(self) -> torch.Tensor:
        return torch.softmax(self.prior_logits, dim=0)

    def mahalanobis(self, z: torch.Tensor) -> torch.Tensor:
        """Squared class distances in the fixed spherical metric."""
        if z.ndim != 2 or z.shape[1] != self.D:
            raise ValueError(f"Expected embeddings of shape (batch, {self.D}), got {tuple(z.shape)}.")
        delta = z[:, None, :] - self.mu.to(dtype=z.dtype)[None, :, :]
        return delta.square().sum(dim=-1) / self.variance.to(dtype=z.dtype)

    def tau(self) -> torch.Tensor:
        """Normalization thresholds, computed without determinants or gamma overflow."""
        p = self.power
        log_peak = (
            math.lgamma(self.D / 2 + p + 1) - math.lgamma(p + 1)
            - 0.5 * self.D * (math.log(2 * math.pi) + self.variance.log())
            + p * math.log(p)
        ) / (self.D / 2 + p)
        return -log_peak.exp().expand(self.C)

    def forward(self, z: torch.Tensor, return_aux: bool = False):
        """Return (batch, C) joint densities, optionally with Q, tau, and priors."""
        maha = self.mahalanobis(z)
        tau = self.tau().to(dtype=maha.dtype)
        pi = self.priors().to(dtype=maha.dtype)
        base = (-tau - 0.5 * maha) / self.power
        # Avoid infinite derivatives of x^power at zero when q < 0.
        densities = torch.where(base > 0, base.clamp_min(torch.finfo(base.dtype).tiny).pow(self.power), 0)
        scores = pi[None, :] * densities
        if return_aux:
            return scores, maha, tau, pi
        return scores


def qgaussian_joint_fy_loss(head: SimplexQGaussianHead, z, target):
    """Per-example joint FY loss with alpha=2-q, including outside support.

    Omega*(f) = 1/(alpha*(alpha-1)) + 2*a/(D+2*power+2)*sum(pi^alpha),
    with a=-tau, power=1/(1-q), and f(c,z)=pi_c^(alpha-1)*(a-Q_c(z)/2).
    At q=0 this recovers tpda_joint_fy_loss.
    """
    maha = head.mahalanobis(z)
    a = -head.tau()[0].to(dtype=maha.dtype)
    alpha = head.alpha
    b = alpha - 1
    log_pi = torch.log_softmax(head.prior_logits.to(dtype=maha.dtype), dim=0)
    global_term = 1 / (alpha * b) + 2 * a / (head.D + 2 * head.power + 2) * (alpha * log_pi).exp().sum()
    q_y = maha.gather(1, target[:, None]).squeeze(1)
    return global_term + (b * log_pi[target]).exp() * (0.5 * q_y - a)


class ClasswiseQGaussianReg(nn.Module):
    """Class-wise projected CF matching for compact q-Gaussians (q < 1).

    The target has radial power 1/(1-q); q=0 is TP and q=0.5 is biweight.
    Class c contributes (n_c/N)*n_c times its mean projected integral.
    Absent/singleton classes contribute zero. Advance step each minibatch.
    Centers and the moment scale come from the fixed head.
    """

    def __init__(self, D, num_slices=256, seed=1729, *, num_points=65, quadrature_points=64, q=0.5):
        super().__init__()
        for name, value, minimum in (
            ("D", D, 1), ("num_slices", num_slices, 1),
            ("num_points", num_points, 3), ("quadrature_points", quadrature_points, 16),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        self.q = float(q)
        if not math.isfinite(self.q) or self.q >= 1:
            raise ValueError("q must be finite and less than 1 (compact support).")
        self.power = 1 / (1 - self.q)
        self.D = D
        self.moment_dimension = D + 2 * self.power + 2
        self.num_slices = num_slices
        self.seed = int(seed)

        # Golub-Welsch rule for normalized weight (1-u^2)^alpha on [-1, 1].
        # The first eigenvector components squared are the probability weights.
        alpha = self.power + (D - 1) / 2
        k = torch.arange(1, quadrature_points, dtype=torch.float64)
        off_diagonal = (k * (k + 2 * alpha) / (4 * (k + alpha).square() - 1)).sqrt()
        jacobi = torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
        nodes, eigenvectors = torch.linalg.eigh(jacobi)
        self.register_buffer("projection_nodes", nodes * math.sqrt(self.moment_dimension))
        self.register_buffer("projection_weights", eigenvectors[0].square())
        grid = torch.linspace(-5, 5, num_points, dtype=torch.float64)
        self.register_buffer("integration_grid", grid)
        self.register_buffer("target_cf", self.projected_cf(grid))

    def projected_cf(self, frequencies):
        """Target CF for a unit-variance projection (real, even, and 1 at zero)."""
        nodes = self.projection_nodes.to(frequencies)
        weights = self.projection_weights.to(frequencies)
        return (frequencies[..., None] * nodes).cos() @ weights

    def statistics(self, embeddings, target, centers, *, radius, step=0):
        """Return (per-class statistics, counts) for fixed spherical targets.

        embeddings: (N, D); zero-based int64 target: (N,); centers: (C, D).
        radius is the head's shared support radius, not sqrt(shape variance).
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != self.D:
            raise ValueError(f"embeddings must have nonempty shape (N, {self.D}).")
        if target.ndim != 1 or target.shape[0] != embeddings.shape[0] or target.dtype != torch.long:
            raise ValueError("target must be an int64 vector of length N.")
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != self.D:
            raise ValueError(f"centers must have nonempty shape (C, {self.D}).")
        if target.device != embeddings.device:
            raise ValueError("target and embeddings must be on the same device.")
        if ((target < 0) | (target >= centers.shape[0])).any():
            raise ValueError("target labels must lie in [0, C).")
        radius = torch.as_tensor(radius, dtype=embeddings.dtype, device=embeddings.device)
        if radius.numel() != 1 or not torch.isfinite(radius).all() or (radius <= 0).any():
            raise ValueError("radius must be one finite positive scalar.")
        centers = centers.to(embeddings)
        residuals = (embeddings - centers[target]) * (math.sqrt(self.moment_dimension) / radius.reshape(()))
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed + int(step))
        directions = torch.randn(
            self.D, self.num_slices, generator=generator,
            device=embeddings.device, dtype=embeddings.dtype,
        )
        directions = nn.functional.normalize(directions, dim=0)
        grid = self.integration_grid.to(embeddings)
        phases = (residuals @ directions).unsqueeze(-1) * grid
        membership = nn.functional.one_hot(target, num_classes=centers.shape[0]).T.to(embeddings)
        counts = membership.sum(1)
        denominator = counts.clamp_min(1).view(-1, 1, 1)
        cf_real = torch.einsum("cn,nmt->cmt", membership, phases.cos()) / denominator
        cf_imag = torch.einsum("cn,nmt->cmt", membership, phases.sin()) / denominator
        # The target CF can change sign; it cannot be an integration weight.
        error = (cf_real - self.target_cf.to(embeddings)).square() + cf_imag.square()
        error = error * torch.exp(-0.5 * grid.square())
        statistics = counts * torch.trapezoid(error, grid, dim=-1).mean(dim=-1)
        return statistics * (counts >= 2).to(statistics.dtype), counts

    def forward(self, embeddings, target, centers, *, radius, step=0):
        statistics, counts = self.statistics(embeddings, target, centers, radius=radius, step=step)
        return (counts / embeddings.shape[0] * statistics).sum()


class QGaussianRegFYLoss(nn.Module):
    """Joint FY loss plus CF matching for a fixed q < 1 (default 0.5)."""

    def __init__(self, D, q_weight=1.0, num_slices=256, seed=1729, *, num_points=65, q=0.5):
        super().__init__()
        self.q_weight = float(q_weight)
        if not math.isfinite(self.q_weight) or self.q_weight < 0:
            raise ValueError("q_weight must be finite and nonnegative.")
        self.regularizer = ClasswiseQGaussianReg(D, num_slices, seed, num_points=num_points, q=q)

    def forward(self, head, embeddings, target, *, step=0):
        if head.D != self.regularizer.D:
            raise ValueError("head and regularizer dimensions must match.")
        if head.q != self.regularizer.q:
            raise ValueError("head and regularizer q values must match.")
        loss = qgaussian_joint_fy_loss(head, embeddings, target).mean()
        if self.q_weight == 0:
            return loss
        return loss + self.q_weight * self.regularizer(
            embeddings, target, head.mu, radius=head.support_radius, step=step,
        )


def sample_qgaussian_residuals(num_samples, D, *, q=0.5, radius=1.0,
                               generator=None, dtype=torch.float32):
    """Exact compact q-Gaussian residuals on CPU; q=0 samples TP.

    Squared relative radii follow Beta(D/2, 1/(1-q)+1). A normal vector
    supplies the direction and the first chi-square term; an independent
    Gamma draw supplies the remaining term. Uses the supplied generator.
    """
    q = float(q)
    if not math.isfinite(q) or q >= 1:
        raise ValueError("q must be finite and less than 1 (compact support).")
    normal = torch.randn(num_samples, D, generator=generator, dtype=dtype, device="cpu")
    shape = torch.full((num_samples, 1), 1 / (1 - q) + 1, dtype=dtype, device="cpu")
    gamma = torch._standard_gamma(shape, generator=generator)
    norm = (normal.square().sum(1, keepdim=True) + 2 * gamma).sqrt()
    return normal / norm * torch.as_tensor(radius, dtype=dtype, device="cpu")
