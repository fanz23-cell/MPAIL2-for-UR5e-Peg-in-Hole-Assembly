import torch
from typing import TYPE_CHECKING, Optional

from mpail2.ensemble import EnsembleMLP

if TYPE_CHECKING:
    from .configs.cfgs import EnsembleValueCfg


class EnsembleValue(torch.nn.Module):
    """
    Ensemble of Q-networks that samples two at random during forward pass,
    similar to TD-MPC2 implementation.
    """

    def __init__(
        self,
        cfg: 'EnsembleValueCfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.num_q = cfg.ensemble_size

        self.model = EnsembleMLP(
            ensemble_size=self.num_q,
            input_dim=cfg.state_dim + cfg.action_dim,
            output_dim=1,
            **self.cfg.model_kwargs,
        ).to(device=device, dtype=dtype)

    @staticmethod
    def sample_ensemble_q(
        q_values: torch.Tensor, traj_ndim: Optional[int] = None
    ) -> torch.Tensor:
        """Sample one Q-head from ``q_values`` with shape ``[E, *batch]``.

        If ``traj_ndim`` is None, samples independently per batch element (full ``batch``).
        If set to ``k``, the first ``k`` dimensions of ``batch`` share one draw; remaining
        dimensions broadcast (e.g. time within a rollout).
        """
        e, *batch_dims = q_values.shape
        if len(batch_dims) == 0:
            sample_idx = torch.randint(0, e, (1,), device=q_values.device)
            return q_values[sample_idx.item()]
        if traj_ndim is not None:
            if traj_ndim > len(batch_dims) or traj_ndim < 0:
                raise ValueError(
                    f"traj_ndim={traj_ndim} invalid for batch_dims={batch_dims}"
                )
            traj_shape = batch_dims[:traj_ndim]
            rest_shape = batch_dims[traj_ndim:]
            sample_idx = torch.randint(0, e, traj_shape, device=q_values.device)
            e_idx = sample_idx.reshape(*traj_shape, *([1] * len(rest_shape))).expand(
                *batch_dims
            )
            grids = torch.meshgrid(
                *[torch.arange(d, device=q_values.device) for d in batch_dims],
                indexing="ij",
            )
            return q_values[(e_idx,) + tuple(grids)]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        flat = q_values.reshape(e, batch_size)
        sample_idx = torch.randint(0, e, (batch_size,), device=q_values.device)
        gather_idx = sample_idx.view(1, batch_size).expand(1, batch_size)
        sampled = flat.gather(0, gather_idx).squeeze(0)
        return sampled.reshape(*batch_dims)

    def track_q_grad(self, enable: bool = True):
        for param in self.model.parameters():
            param.requires_grad_(enable)

    def forward(
        self,
        state,
        action,
        return_type='min',
        sample_traj_ndim: Optional[int] = None,
    ):
        """
        Forward pass through ensemble.

        Args:
            state: State tensor
            action: Action tensor
            return_type: 'min' returns minimum of 2 sampled Q-values,
                        'avg' returns average of 2 sampled Q-values,
                        'all' returns all Q-values
                        'max' returns maximum of 2 sampled Q-values
            sample_traj_ndim: For ``return_type=='sample'``, number of leading batch
                dimensions that share one random head (see ``sample_ensemble_q``).

        Returns:
            Q-value(s) based on return_type
        """
        assert return_type in {'min', 'avg', 'all', 'sample', 'max'}
        _input = torch.cat([state, action], dim=-1)
        ensemble_input = _input.unsqueeze(0).expand(self.num_q, *_input.shape)
        q_values = self.model(ensemble_input).squeeze(-1)

        if return_type == 'all':
            return q_values

        if return_type == 'avg':
            return q_values.mean(dim=0)

        if return_type == 'sample':
            return self.sample_ensemble_q(q_values, traj_ndim=sample_traj_ndim)

        if return_type == 'max':
            # Match value head behavior: sample a random pair, then take optimistic max.
            qidx = torch.randperm(self.num_q, device=q_values.device)[:2]
            sampled_qs = q_values[qidx]
            return sampled_qs.max(0).values

        # Keep conservative min behavior on a random pair.
        qidx = torch.randperm(self.num_q, device=q_values.device)[:2]
        sampled_qs = q_values[qidx]
        return sampled_qs.min(0).values
