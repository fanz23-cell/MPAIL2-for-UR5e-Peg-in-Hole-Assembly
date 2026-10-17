"""Dynamics models for MPAIL."""

import torch
from typing import TYPE_CHECKING, Literal

from mpail2.utils import resolve_obj
from mpail2.ensemble import EnsembleMLP

if TYPE_CHECKING:
    from .configs.cfgs import DynamicsCfg

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: Tensor of shape (T, B, D).
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)).square()
        err = err + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


class Dynamics(torch.nn.Module):
    """
    Latent dynamics model. Propagates latent states through a feedforward network.
    Expects already-encoded latent states as input.
    """
    def __init__(
        self,
        dynamics_cfg: 'DynamicsCfg',
        num_envs: int,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(Dynamics, self).__init__()
        self.dtype = dtype
        self.d = device
        self.cfg = dynamics_cfg
        self.num_envs = num_envs
        self.ensemble_size = getattr(self.cfg, "ensemble_size", 1)

        # Feedforward dynamics network
        if self.ensemble_size > 1:
            self.model = EnsembleMLP(
                ensemble_size=self.ensemble_size,
                input_dim=self.cfg.latent_dim + self.cfg.action_dim,
                output_dim=self.cfg.latent_dim,
                **self.cfg.model_kwargs,
            )
        else:
            self.model = resolve_obj(self.cfg.model_factory)(
                input_dim=self.cfg.latent_dim + self.cfg.action_dim,
                output_dim=self.cfg.latent_dim,
                **self.cfg.model_kwargs,
            )

    def step(self, current_latent: torch.Tensor, control: torch.Tensor):
        """
        Compute next latent state using feedforward network.

        Args:
            current_latent: Current latent state of shape (..., latent_dim)
            control: Control input of shape (..., action_dim)

        Returns:
            next_latent: Next latent state of shape (..., latent_dim)
        """
        model_input = torch.cat((current_latent, control), dim=-1)
        return self.model(model_input)

    @staticmethod
    def sample_ensemble_rollout(zs: torch.Tensor) -> torch.Tensor:
        """Sample one ensemble member per rollout from ``zs`` with shape ``[E, ..., T+1, D]``."""
        e, *batch_dims, t_plus_1, d = zs.shape
        if len(batch_dims) == 0:
            sample_idx = torch.randint(e, (1,), device=zs.device)
            return zs[sample_idx.item()]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        zs_flat = zs.reshape(e, batch_size, t_plus_1, d)
        sample_idx = torch.randint(e, (batch_size,), device=zs.device)
        gather_idx = sample_idx.view(1, batch_size, 1, 1).expand(1, batch_size, t_plus_1, d)
        sampled = zs_flat.gather(0, gather_idx).squeeze(0)
        return sampled.reshape(*batch_dims, t_plus_1, d)

    @torch.compile
    def forward(
        self,
        z0: torch.Tensor,
        controls: torch.Tensor,
        return_type: Literal["avg", "all", "sample"] = "avg",
    ):
        """
        Forward pass for dynamics model. Rolls out latent trajectory from initial latent state.

        Args:
            z0: initial latent state of shape (B, ..., latent_dim)
            controls: controls of shape (B, ..., T, a_dim)
            tf_mask: mask for teacher forcing of shape (B, ..., T)
            latent_gt: ground truth latent states for teacher forcing of shape (B, ..., T, latent_dim)

        Returns:
            zs: latent trajectory of shape (B, ..., T+1, latent_dim)
        """
        assert return_type in {"avg", "all", "sample"}
        controls = controls.to(device=self.d, dtype=self.dtype)
        T = controls.shape[-2]
        # Match batch dimensions: if controls is (B, ..., T, a_dim), expand z0 to (B, ..., latent_dim)
        batch_shape = controls.shape[:-2]  # (B, ...)
        num_extra_dims = len(batch_shape) - (z0.ndim - 1)
        z0_expanded = z0.view(*z0.shape[:-1], *([1] * max(0, num_extra_dims)), z0.shape[-1])
        z0_expanded = z0_expanded.expand(*batch_shape, z0.shape[-1])
        if self.ensemble_size > 1:
            z0_expanded = z0_expanded.unsqueeze(0).expand(self.ensemble_size, *z0_expanded.shape)
            controls = controls.unsqueeze(0).expand(self.ensemble_size, *controls.shape)

        zs_list = [z0_expanded]

        for i in range(T):
            current_latent = zs_list[-1]

            # Step dynamics in latent space
            next_latent = self.step(current_latent, controls[..., i, :])
            zs_list.append(next_latent)

        # Stack all latent states
        zs = torch.stack(zs_list, dim=-2)
        if self.ensemble_size == 1:
            return zs
        if return_type == "all":
            return zs
        if return_type == "sample":
            return self.sample_ensemble_rollout(zs)
        return zs.mean(dim=0)
