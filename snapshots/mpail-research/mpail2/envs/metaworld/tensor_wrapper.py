"""Vector env + torch tensors for Meta-World; preserves multi-key observation dicts."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class MetaWorldVectorTensorWrapper:
    """Wrap ``SyncVectorEnv`` (or any ``VectorEnv``): torch device/dtype, full Dict obs.

    Unlike ``VectorizedGymnasiumWrapper``, does **not** collapse dict observations to a
    single top-level ``obs`` key.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        num_envs: int,
        max_episode_length: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(env, gym.vector.VectorEnv):
            raise ValueError("MetaWorldVectorTensorWrapper requires a gym.vector.VectorEnv.")

        self.env = env
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.dtype = dtype
        self.current_iteration = 0
        self._seed = None

        u = env.unwrapped
        if not hasattr(u, "num_envs"):
            u.num_envs = num_envs
        if not hasattr(u, "max_episode_length"):
            u.max_episode_length = max_episode_length
        if not hasattr(u, "device"):
            u.device = device
        if not hasattr(u, "current_iteration"):
            u.current_iteration = 0
        if not hasattr(u, "seed"):

            def seed_method(seed):
                self._seed = seed
                return [seed]

            u.seed = seed_method

        self.action_space = env.action_space

        # Preserve the full Dict observation space (proprio + cam_* keys) rather
        # than collapsing to a single "obs" key like VectorizedGymnasiumWrapper.
        # sync_learner_dims_from_env reads single_observation_space (delegated to
        # SyncVectorEnv via __getattr__) to configure per-key encoder dimensions.
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Dict({"obs": env.observation_space})

    def _to_tensor(self, x, dtype: torch.dtype | None = None):
        if isinstance(x, dict):
            return {k: self._to_tensor(v, dtype) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(item, dtype) for item in x)
        tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        if not isinstance(tensor, torch.Tensor):
            td = dtype if dtype is not None else self.dtype
            tensor = torch.tensor(tensor, dtype=td)
        target_dtype = dtype if dtype is not None else self.dtype
        return tensor.to(dtype=target_dtype, device=self.device)

    def _obs_to_tensor_dict(self, obs) -> dict[str, torch.Tensor]:
        t = self._to_tensor(obs)
        if not isinstance(t, dict):
            return {"obs": t}
        return t

    def _tensorize_info(self, info) -> dict:
        """Convert vector-env info (dict of batched arrays) to tensors where ndarray."""
        if not isinstance(info, dict):
            return info
        out: dict = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                out[k] = self._to_tensor(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                out[k] = self._to_tensor(np.stack(v, axis=0))
            elif isinstance(v, tuple):
                out[k] = v
            else:
                out[k] = v
        return out

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_tensor = self._obs_to_tensor_dict(obs)
        info_tensor = self._tensorize_info(info)
        return obs_tensor, info_tensor

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions

        obs, reward, terminated, truncated, info = self.env.step(actions_np)

        obs_tensor = self._obs_to_tensor_dict(obs)
        reward_tensor = self._to_tensor(reward)
        terminated_tensor = self._to_tensor(terminated, dtype=torch.bool)
        truncated_tensor = self._to_tensor(truncated, dtype=torch.bool)
        info_tensor = self._tensorize_info(info)

        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info_tensor

    def seed(self, seed):
        self._seed = seed
        if hasattr(self.env.unwrapped, "seed"):
            self.env.unwrapped.seed(seed)
        return [seed]

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        return getattr(self.env, name)
