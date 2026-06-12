import torch
import warnings
from typing import TYPE_CHECKING

from mpail2.utils import resolve_obj  # Accommodate both yaml and Configclass
from mpail2.value import EnsembleValue

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
        self.action_dim = cfg.action_dim or 0
        self.use_action = bool(cfg.use_action)
        input_dim = cfg.state_dim * 2 + (self.action_dim if self.use_action else 0)

        self.model = resolve_obj(self.cfg.model_factory)(
            input_dim = input_dim, # for s, s', and optionally a
            output_dim = 1,
            **cfg.model_kwargs,
        ).to(device=device, dtype=dtype)

    @torch.compile
    def forward(self, state, next_state, action=None):
        '''state shape: (num_envs, state_dim)'''
        inputs = [state, next_state]
        if self.use_action:
            if action is None:
                action = torch.zeros(*state.shape[:-1], self.action_dim, device=state.device, dtype=state.dtype)
            inputs.append(action)
        _input = torch.cat(inputs, dim=-1)
        return self.model(_input).squeeze(-1)
