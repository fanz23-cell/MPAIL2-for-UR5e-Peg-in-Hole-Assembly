#!/usr/bin/env python3
"""Play mpail3 OmniReset checkpoints with task-command success diagnostics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPAIL3_ROOT = ROOT / "mpail-research"
sys.path.insert(0, str(MPAIL3_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--demo-path", type=Path, default=ROOT / "mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_mpail.pt")
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--max-steps-per-episode", type=int, default=180)
parser.add_argument("--print-every", type=int, default=5)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--reset-types", nargs="+", default=["ObjectAnywhereEEAnywhere"])
parser.add_argument("--raw-action-limit", nargs=7, type=float, default=[40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0])
parser.add_argument("--seed", type=int, default=44)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402

from mpail2.envs.isaac.omnireset_wrapper import OmniResetMPAILWrapper  # noqa: E402
from mpail2.learner import MPAIL2Learner  # noqa: E402
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig  # noqa: E402
from mpail2.train.utils.demos import load_demonstrations  # noqa: E402
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env  # noqa: E402


def _progress(env):
    try:
        return env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    except Exception:
        return None


def _float_metric(infos, key: str) -> float | None:
    info_log = infos.get("log", {}) if isinstance(infos, dict) else {}
    value = info_log.get(key)
    if value is None:
        return None
    return float(torch.as_tensor(value).float().mean().item())


def _build_mpail3(env, checkpoint: Path, demo_path: Path, device: str):
    learner_cfg = sync_learner_dims_from_env(OmniResetStateLearnerConfig(), env)
    learner_cfg.planner_cfg.temperature = 2.0
    learner_cfg.policy_learner_cfg.use_intrinsic_bonus = False
    demonstrations = load_demonstrations(str(demo_path), device=device)
    learner = MPAIL2Learner(demonstrations, 1, learner_cfg, device=device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = learner.planner.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"mpail3 checkpoint did not load cleanly: missing={len(missing)} unexpected={len(unexpected)}\n"
            f"missing[:8]={list(missing)[:8]}\nunexpected[:8]={list(unexpected)[:8]}"
        )
    learner.eval()
    print(f"[INFO] Strictly loaded mpail3 checkpoint iter={ckpt.get('iter', '?')}: {checkpoint}", flush=True)
    return learner


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = OmniResetMPAILWrapper(env, raw_action_limit=args_cli.raw_action_limit, compact_obs_dim=215)
    actor = _build_mpail3(env, args_cli.checkpoint, args_cli.demo_path, device)
    title = "S192 MPAIL3"

    print(f"[PLAY] {title}: {args_cli.num_episodes} sequential episodes, num_envs=1", flush=True)
    obs, _ = env.reset()
    ep = 1
    ep_step = 0
    ep_return = torch.zeros(1, device=device)
    prog = _progress(env)
    try:
        while ep <= args_cli.num_episodes and simulation_app.is_running():
            with torch.inference_mode():
                actions = actor.act(obs)
            obs, rewards, terms, truncs, infos = env.step(actions.to(env.unwrapped.device))
            env.unwrapped.sim.render()
            ep_return += rewards.to(device)
            ep_step += 1
            done = bool((terms | truncs).any()) or ep_step >= args_cli.max_steps_per_episode
            if ep_step % args_cli.print_every == 0 or done:
                extra = ""
                if prog is not None:
                    success = float(prog.success.float().mean().item())
                    xyz = float(prog.xyz_distance.mean().item())
                    extra += f" progress_success={success:.3f} xyz={xyz:.4f}"
                end_pos = _float_metric(infos, "Metrics/task_command/end_of_episode_pos_align_error")
                avg_pos = _float_metric(infos, "Metrics/task_command/average_pos_align_error")
                task_succ = _float_metric(infos, "Metrics/task_command/end_of_episode_success_rate")
                if end_pos is not None:
                    extra += f" task_end_pos={end_pos:.4f} geo25={float(end_pos <= 0.025):.0f} geo50={float(end_pos <= 0.050):.0f}"
                if avg_pos is not None:
                    extra += f" task_avg_pos={avg_pos:.4f}"
                if task_succ is not None:
                    extra += f" task_succ={task_succ:.3f}"
                print(
                    f"[{title}] ep={ep}/{args_cli.num_episodes} step={ep_step:03d} "
                    f"ret={ep_return.mean().item():.3f} action_norm={actions.norm(dim=-1).mean().item():.3f}{extra}",
                    flush=True,
                )
            if done:
                print(f"[EPISODE_DONE] {title} ep={ep} steps={ep_step} return={ep_return.item():.3f}", flush=True)
                ep += 1
                if ep <= args_cli.num_episodes:
                    obs, _ = env.reset()
                    ep_step = 0
                    ep_return.zero_()
            if args_cli.sleep > 0:
                time.sleep(args_cli.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
