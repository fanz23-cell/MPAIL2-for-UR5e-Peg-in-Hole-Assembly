#!/usr/bin/env python3
"""Collect successful ObjectAnywhereEEGrasped route-A rollouts for OmniReset.

Route A starts from OmniReset's grasped reset distribution and keeps complete
episodes only if the official teacher reaches success at least once.  The raw
file intentionally stores success, reward, done, episode ids, action norms, and
peg-hole distance so downstream slicing can prove what was kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect OmniReset route-A successful state/action rollouts.")
parser.add_argument("--policy", type=Path, required=True, help="TorchScript policy.pt exported from Stage2 teacher")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--target-success-episodes", type=int, default=20)
parser.add_argument("--max-env-steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clip-actions", type=float, default=0.0, help="Clip raw actions if >0; 0 matches RSL-RL None.")
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default="ObjectAnywhereEEGrasped")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402


def _policy_obs(obs) -> torch.Tensor:
    return obs if isinstance(obs, torch.Tensor) else obs["policy"]


def _progress_term(env):
    return env.unwrapped.reward_manager.get_term_cfg("progress_context").func


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
    params = getattr(reset_event, "params", None)
    if params is None:
        raise ValueError("Route A requires an env with reset_from_reset_states params.")
    params["reset_types"] = [args_cli.reset_type]
    params["probs"] = [1.0]
    print(f"[INFO] Route A reset override: {params['reset_types']} probs={params['probs']}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    policy = torch.jit.load(str(args_cli.policy), map_location=device).eval()

    obs, _ = env.reset()
    obs_t = _policy_obs(obs)
    episode_buffers: list[list[dict[str, torch.Tensor | int | float | bool]]] = [[] for _ in range(args_cli.num_envs)]
    episode_ids = torch.arange(args_cli.num_envs, dtype=torch.long)
    next_episode_id = args_cli.num_envs

    kept: list[dict[str, torch.Tensor | int | float | bool]] = []
    total_episodes = 0
    success_episodes = 0
    env_steps = 0

    try:
        while (
            simulation_app.is_running()
            and success_episodes < args_cli.target_success_episodes
            and env_steps < args_cli.max_env_steps
        ):
            with torch.inference_mode():
                raw_actions = policy(obs_t.to(device))
                if args_cli.clip_actions and args_cli.clip_actions > 0:
                    actions = raw_actions.clamp(-args_cli.clip_actions, args_cli.clip_actions)
                else:
                    actions = raw_actions

            next_obs, rewards, terms, truncs, _infos = env.step(actions)
            next_obs_t = _policy_obs(next_obs)
            dones = terms | truncs

            progress = _progress_term(env)
            success = progress.success.detach().clone()
            xyz_distance = progress.xyz_distance.detach().clone()
            action_norm = actions.norm(dim=-1).detach().clone()

            for env_id in range(args_cli.num_envs):
                episode_buffers[env_id].append(
                    {
                        "obs": obs_t[env_id].detach().cpu(),
                        "action": actions[env_id].detach().cpu(),
                        "next_obs": next_obs_t[env_id].detach().cpu(),
                        "reward": float(rewards[env_id].detach().cpu()),
                        "done": bool(dones[env_id].detach().cpu()),
                        "success": bool(success[env_id].detach().cpu()),
                        "xyz_distance": float(xyz_distance[env_id].detach().cpu()),
                        "action_norm": float(action_norm[env_id].detach().cpu()),
                        "episode_id": int(episode_ids[env_id]),
                        "step_in_episode": len(episode_buffers[env_id]),
                    }
                )

            done_ids = dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            for env_id in done_ids:
                total_episodes += 1
                episode = episode_buffers[env_id]
                if any(bool(row["success"]) for row in episode):
                    kept.extend(episode)
                    success_episodes += 1
                episode_buffers[env_id] = []
                episode_ids[env_id] = next_episode_id
                next_episode_id += 1

            env_steps += args_cli.num_envs
            obs_t = next_obs_t

            if env_steps % (args_cli.num_envs * 50) == 0 or done_ids:
                print(
                    f"env_steps={env_steps} total_episodes={total_episodes} "
                    f"success_episodes={success_episodes} kept_transitions={len(kept)}",
                    flush=True,
                )
    finally:
        env.close()

    if not kept:
        raise RuntimeError("No successful episodes collected; refusing to save an empty route-A dataset.")

    def stack_tensor(key: str) -> torch.Tensor:
        return torch.stack([row[key] for row in kept])  # type: ignore[arg-type]

    output = {
        "data": {
            "obs": {"policy": stack_tensor("obs").float()},
            "actions": stack_tensor("action").float(),
            "next_obs": {"policy": stack_tensor("next_obs").float()},
            "rewards": torch.tensor([row["reward"] for row in kept], dtype=torch.float32),
            "dones": torch.tensor([row["done"] for row in kept], dtype=torch.bool),
            "success": torch.tensor([row["success"] for row in kept], dtype=torch.bool),
            "xyz_distance": torch.tensor([row["xyz_distance"] for row in kept], dtype=torch.float32),
            "action_norm": torch.tensor([row["action_norm"] for row in kept], dtype=torch.float32),
            "episode_id": torch.tensor([row["episode_id"] for row in kept], dtype=torch.long),
            "step_in_episode": torch.tensor([row["step_in_episode"] for row in kept], dtype=torch.long),
        },
        "metadata": {
            "policy": str(args_cli.policy),
            "task": args_cli.task,
            "reset_type": args_cli.reset_type,
            "target_success_episodes": args_cli.target_success_episodes,
            "total_episodes_seen": total_episodes,
            "success_episodes_kept": success_episodes,
            "note": "Complete successful ObjectAnywhereEEGrasped episodes; slice before BC/MPAIL.",
        },
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(args_cli.output))
    print(f"Saved {len(kept)} transitions from {success_episodes} successful episodes -> {args_cli.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
