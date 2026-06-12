#!/usr/bin/env python3
"""Play stored OmniReset raw teacher-demo action episodes in the Isaac viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Watch stored raw OmniReset demo actions.")
parser.add_argument(
    "--demo",
    type=Path,
    default=Path("mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_success_rollouts_raw.pt"),
)
parser.add_argument("--num-episodes", type=int, default=3)
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--sleep", type=float, default=0.03)
parser.add_argument("--print-every", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default=None)
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
    payload = torch.load(str(args_cli.demo), map_location="cpu", weights_only=False)
    data = payload["data"]
    metadata = payload.get("metadata", {})
    actions = data["actions"].float()
    episode_id = data["episode_id"].long()
    saved_success = data.get("success")
    saved_xyz = data.get("xyz_distance")
    unique_eps = torch.unique(episode_id, sorted=True)
    selected = unique_eps[args_cli.start_episode : args_cli.start_episode + args_cli.num_episodes]
    if selected.numel() == 0:
        raise ValueError(f"No episodes selected from {args_cli.demo}; available={len(unique_eps)}")

    reset_type = args_cli.reset_type or metadata.get("reset_type")
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested reset type, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = [reset_type]
        params["probs"] = [1.0]
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print(
        f"[INFO] Playing stored demo actions: {args_cli.demo} "
        f"episodes={len(unique_eps)} selected={selected.detach().cpu().tolist()}",
        flush=True,
    )
    print(f"[INFO] Metadata: {metadata}", flush=True)

    try:
        for ep in selected.detach().cpu().tolist():
            idx = (episode_id == int(ep)).nonzero(as_tuple=False).flatten()
            ep_actions = actions[idx].to(device=device)
            ep_success = saved_success[idx] if torch.is_tensor(saved_success) else None
            ep_xyz = saved_xyz[idx] if torch.is_tensor(saved_xyz) else None
            obs, _ = env.reset()
            total_reward = torch.zeros(1, device=device)
            print(f"[DEMO_EP_START] episode_id={int(ep)} steps={len(idx)}", flush=True)

            for step, action in enumerate(ep_actions):
                if not simulation_app.is_running():
                    return
                obs, rewards, terms, truncs, _infos = env.step(action.view(1, -1))
                total_reward += rewards.to(device)
                dones = terms | truncs
                progress = _progress_term(env)
                live_success = bool(progress.success.detach().cpu().flatten()[0])
                live_xyz = float(progress.xyz_distance.detach().cpu().flatten()[0])
                saved_s = None if ep_success is None else bool(ep_success[step].item())
                saved_d = None if ep_xyz is None else float(ep_xyz[step].item())
                if step % args_cli.print_every == 0 or bool(dones.any()) or saved_s:
                    print(
                        f"episode_id={int(ep)} step={step:03d}/{len(idx)} "
                        f"reward={rewards.mean().item():.4f} total={total_reward.mean().item():.4f} "
                        f"live_success={live_success} live_xyz={live_xyz:.4f} "
                        f"saved_success={saved_s} saved_xyz={saved_d if saved_d is None else round(saved_d, 4)} "
                        f"action_norm={action.norm().item():.4f} done={bool(dones.any())}",
                        flush=True,
                    )
                if bool(dones.any()):
                    print(f"[DEMO_EP_DONE] episode_id={int(ep)} env_done_at_step={step}", flush=True)
                    break
                if args_cli.sleep > 0:
                    time.sleep(args_cli.sleep)
            print(f"[DEMO_EP_END] episode_id={int(ep)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
