#!/usr/bin/env python3
"""Play a sequence of MPAIL2 OmniReset checkpoints in one Isaac Sim session.

This is for screen recording training progress without using env.render(), whose
RGB annotator path is currently unreliable on this Isaac Sim install.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
parser.add_argument("--demo-path", type=Path, required=True)
parser.add_argument("--steps-per-checkpoint", type=int, default=120)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--compact-obs-dim", type=int, choices=(13, 19, 22, 215), default=22)
parser.add_argument("--raw-action-limit", nargs=7, type=float, default=None)
parser.add_argument("--force-gripper-close", action="store_true")
parser.add_argument("--gripper-close-value", type=float, default=-1.0)
parser.add_argument("--reset-types", nargs="+", default=None)
parser.add_argument(
    "--disable-domain-randomization",
    action="store_true",
    help="Disable OmniReset physical domain randomization while keeping deterministic controller setup.",
)
parser.add_argument("--planner-timesteps", type=int, default=None)
parser.add_argument("--loss-horizon", type=int, default=None)
parser.add_argument("--ready-file", type=Path, default=None)
parser.add_argument("--start-file", type=Path, default=None)
parser.add_argument("--done-file", type=Path, default=None)
parser.add_argument("--exit-file", type=Path, default=None)
parser.add_argument("--print-every", type=int, default=40)
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

from mpail2.envs.isaac.omnireset_clean_eval import apply_omnireset_clean_eval_overrides  # noqa: E402
from mpail2.envs.isaac.omnireset_wrapper import (  # noqa: E402
    DEFAULT_RAW_ACTION_LIMIT,
    OmniResetMPAILWrapper,
    compact_omnireset_demonstrations,
)
from mpail2.learner import MPAIL2Learner  # noqa: E402
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig  # noqa: E402
from mpail2.train.utils.demos import load_demonstrations  # noqa: E402
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env  # noqa: E402


def _load_action_scale(ckpt: dict, device: str) -> torch.Tensor | None:
    if "action_scale" not in ckpt:
        return None
    return torch.as_tensor(ckpt["action_scale"], dtype=torch.float32, device=device).view(1, -1)


def _wait_for_start_signal() -> None:
    if args_cli.ready_file is not None:
        args_cli.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.ready_file.write_text("ready\n")
        print(f"[READY_FOR_RECORDING] {args_cli.ready_file}", flush=True)
    if args_cli.start_file is not None:
        print(f"[WAITING_FOR_RECORDING_START] {args_cli.start_file}", flush=True)
        while simulation_app.is_running() and not args_cli.start_file.exists():
            time.sleep(0.2)
        print("[RECORDING_STARTED]", flush=True)


def _signal_done_and_wait_for_exit() -> None:
    if args_cli.done_file is not None:
        args_cli.done_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.done_file.write_text("done\n")
        print(f"[DONE_FOR_RECORDING] {args_cli.done_file}", flush=True)
    if args_cli.exit_file is not None:
        print(f"[WAITING_FOR_RECORDING_EXIT] {args_cli.exit_file}", flush=True)
        while simulation_app.is_running() and not args_cli.exit_file.exists():
            time.sleep(0.2)
        print("[RECORDING_EXITED]", flush=True)


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    if args_cli.disable_domain_randomization:
        changes = apply_omnireset_clean_eval_overrides(env_cfg)
        print(
            "[INFO] OmniReset clean-eval override: "
            f"disabled physical randomization terms={changes['disabled'] or 'none'}; "
            f"fixed controller terms={changes['fixed'] or 'none'}; reset events are unchanged.",
            flush=True,
        )
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-types, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_action_limit = args_cli.raw_action_limit or list(DEFAULT_RAW_ACTION_LIMIT)
    env = OmniResetMPAILWrapper(
        env,
        raw_action_limit=raw_action_limit,
        compact_obs_dim=args_cli.compact_obs_dim,
        force_gripper_close=args_cli.force_gripper_close,
        gripper_close_value=args_cli.gripper_close_value,
    )
    print(
        "[INFO] Compact wrapper: "
        f"obs={args_cli.compact_obs_dim} raw_action_limit={raw_action_limit} "
        f"force_gripper_close={args_cli.force_gripper_close}",
        flush=True,
    )

    learner_cfg = sync_learner_dims_from_env(OmniResetStateLearnerConfig(), env)
    if args_cli.planner_timesteps is not None:
        learner_cfg.planner_cfg.sampling_cfg.num_timesteps = args_cli.planner_timesteps
    if args_cli.loss_horizon is not None:
        learner_cfg.loss_horizon = args_cli.loss_horizon
    demonstrations = load_demonstrations(str(args_cli.demo_path), device=device)
    if args_cli.compact_obs_dim != 215:
        demonstrations = compact_omnireset_demonstrations(
            demonstrations,
            include_ee_to_hole=args_cli.compact_obs_dim in (19, 22),
            include_ee_vel=args_cli.compact_obs_dim == 22,
        )
    learner = MPAIL2Learner(demonstrations, args_cli.num_envs, learner_cfg, device=device)
    learner.init_storage(
        num_envs=args_cli.num_envs,
        num_steps_per_env=env.unwrapped.max_episode_length,
        actor_obs_shape={"policy": (learner_cfg.planner_cfg.encoder_cfg.input_dim,)},
        critic_obs_shape=None,
        action_shape=[learner_cfg.planner_cfg.action_dim],
    )
    learner.eval()

    _wait_for_start_signal()

    try:
        for ckpt_iter in args_cli.checkpoints:
            ckpt_path = args_cli.checkpoint_dir / f"model_{ckpt_iter}.pt"
            if not ckpt_path.exists():
                print(f"[SKIP] missing {ckpt_path}", flush=True)
                continue
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            learner.planner.load_state_dict(ckpt["model_state_dict"])
            action_scale = _load_action_scale(ckpt, device)
            print(f"[CHECKPOINT_START] model_{ckpt_iter}.pt iter={ckpt.get('iter', '?')}", flush=True)

            obs, _ = env.reset()
            total_reward = torch.zeros(args_cli.num_envs, device=device)
            for step in range(args_cli.steps_per_checkpoint):
                if not simulation_app.is_running():
                    return
                with torch.inference_mode():
                    actions = learner.act(obs)
                env_actions = actions if action_scale is None else actions * action_scale
                obs, rewards, terms, truncs, _ = env.step(env_actions.to(env.unwrapped.device))
                total_reward += rewards.to(device)
                dones = terms | truncs
                if step % args_cli.print_every == 0 or dones.any():
                    print(
                        f"[CHECKPOINT_STEP] iter={ckpt_iter} step={step} "
                        f"reward={rewards.mean().item():.4f} done={bool(dones.any())}",
                        flush=True,
                    )
                if dones.any():
                    obs, _ = env.reset()
                    total_reward.zero_()
                if args_cli.sleep > 0:
                    time.sleep(args_cli.sleep)
            print(f"[CHECKPOINT_END] model_{ckpt_iter}.pt", flush=True)
        _signal_done_and_wait_for_exit()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
