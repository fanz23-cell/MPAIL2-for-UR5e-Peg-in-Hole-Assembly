import torch
from typing import TYPE_CHECKING, Optional

from mpail2.utils import resolve_obj  # Accommodate both yaml and Configclass
from mpail2.ensemble import EnsembleMLP

if TYPE_CHECKING:
    from .configs.cfgs import RewardCfg


class Reward(torch.nn.Module):
    '''Adversarial Inverse Reinforcement Learning reward function. Evaluates state transitions
    using a learned discriminator to classify expert vs generator transitions'''

    def __init__(
        self,
        cfg : 'RewardCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.ensemble_size = getattr(self.cfg, "ensemble_size", 1)

        if self.ensemble_size > 1:
            self.model = EnsembleMLP(
                ensemble_size=self.ensemble_size,
                input_dim=cfg.state_dim * 2,  # for s and s'
                output_dim=1,
                **cfg.model_kwargs,
            ).to(device=device, dtype=dtype)
        else:
            self.model = resolve_obj(self.cfg.model_factory)(
                input_dim=cfg.state_dim * 2,  # for s and s'
                output_dim=1,
                **cfg.model_kwargs,
            ).to(device=device, dtype=dtype)

    @staticmethod
    def sample_ensemble_rewards(
        rewards: torch.Tensor, traj_ndim: Optional[int] = None
    ) -> torch.Tensor:
        """Sample from ``rewards`` with shape ``[E, *batch]``; optional shared draw per trajectory prefix."""
        e, *batch_dims = rewards.shape
        if len(batch_dims) == 0:
            sample_idx = torch.randint(0, e, (1,), device=rewards.device)
            return rewards[sample_idx.item()]
        if traj_ndim is not None:
            if traj_ndim > len(batch_dims) or traj_ndim < 0:
                raise ValueError(
                    f"traj_ndim={traj_ndim} invalid for batch_dims={batch_dims}"
                )
            traj_shape = batch_dims[:traj_ndim]
            rest_shape = batch_dims[traj_ndim:]
            sample_idx = torch.randint(0, e, traj_shape, device=rewards.device)
            e_idx = sample_idx.reshape(*traj_shape, *([1] * len(rest_shape))).expand(
                *batch_dims
            )
            grids = torch.meshgrid(
                *[torch.arange(d, device=rewards.device) for d in batch_dims],
                indexing="ij",
            )
            return rewards[(e_idx,) + tuple(grids)]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        rewards_flat = rewards.reshape(e, batch_size)
        sample_idx = torch.randint(0, e, (batch_size,), device=rewards.device)
        gather_idx = sample_idx.view(1, batch_size).expand(1, batch_size)
        sampled = rewards_flat.gather(0, gather_idx).squeeze(0)
        return sampled.reshape(*batch_dims)

    def forward_from_concat(
        self,
        state_pair: torch.Tensor,
        return_type: str = "min",
        sample_traj_ndim: Optional[int] = None,
    ) -> torch.Tensor:
        assert return_type in {"avg", "all", "sample", "min", "max"}
        if self.ensemble_size == 1:
            rewards = self.model(state_pair).squeeze(-1)
            return rewards
        else:
            ensemble_state_pair = state_pair.unsqueeze(0).expand(self.ensemble_size, *state_pair.shape)
            rewards = self.model(ensemble_state_pair).squeeze(-1)

        if return_type == "min":
            # Match value head behavior: sample a random pair, then take conservative min.
            qidx = torch.randperm(self.ensemble_size, device=rewards.device)[:2]
            sampled_qs = rewards[qidx]
            return sampled_qs.min(0).values
        elif return_type == "max":
            # Match value head behavior: sample a random pair, then take optimistic max.
            qidx = torch.randperm(self.ensemble_size, device=rewards.device)[:2]
            sampled_qs = rewards[qidx]
            return sampled_qs.max(0).values
        elif return_type == "sample":
            return self.sample_ensemble_rewards(
                rewards, traj_ndim=sample_traj_ndim
            )
        elif return_type == "avg":
            return rewards.mean(dim=0)
        elif return_type == "all":
            return rewards

    @torch.compile
    def forward(
        self,
        state,
        next_state,
        action=None,
        return_type: str = "min",
        sample_traj_ndim: Optional[int] = None,
    ):
        """state shape: (num_envs, state_dim). return_type: avg, all, max, or sample (ensemble only)."""
        _input = torch.cat([state, next_state], dim=-1)
        return self.forward_from_concat(
            _input, return_type=return_type, sample_traj_ndim=sample_traj_ndim
        )
