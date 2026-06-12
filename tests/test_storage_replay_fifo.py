import unittest
import importlib.util
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "mpail2_storage_under_test",
    Path(__file__).resolve().parents[1] / "mpail2" / "storage.py",
)
_STORAGE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STORAGE_MODULE)
RolloutStorage = _STORAGE_MODULE.RolloutStorage


class ReplayFifoTest(unittest.TestCase):
    def _make_storage(self):
        return RolloutStorage(
            num_envs=2,
            num_steps_per_env=3,
            actor_obs_shape={"policy": (1,)},
            critic_obs_shape={"policy": (1,)},
            action_shape=(1,),
            replay_size=6,
            replay_batch_size=2,
            device=torch.device("cpu"),
        )

    def _fill_rollout(self, storage, base):
        storage.clear()
        for step in range(storage.num_steps_per_env):
            transition = RolloutStorage.Transition()
            transition.observations = {
                "policy": torch.tensor([[base + step * 10], [base + 1 + step * 10]], dtype=torch.float32)
            }
            transition.actions = torch.zeros(storage.num_envs, 1)
            transition.rewards = torch.zeros(storage.num_envs)
            transition.dones = torch.zeros(storage.num_envs, dtype=torch.bool)
            transition.valid = torch.ones(storage.num_envs, dtype=torch.bool)
            storage.add_transitions(transition)

    def test_fifo_replaces_old_trajectories_after_buffer_is_full(self):
        storage = self._make_storage()

        self._fill_rollout(storage, 0)
        storage.process_replay_buffer()
        self.assertEqual(storage.num_trajectories_stored, 2)
        self.assertTrue(torch.equal(storage.replay_data["obs"]["policy"][0, :, 0], torch.tensor([0.0, 1.0])))

        self._fill_rollout(storage, 100)
        storage.process_replay_buffer()
        self.assertEqual(storage.num_trajectories_stored, 2)
        self.assertTrue(torch.equal(storage.replay_data["obs"]["policy"][0, :, 0], torch.tensor([100.0, 101.0])))
        self.assertTrue(torch.equal(storage.replay_data["age"], torch.tensor([0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
