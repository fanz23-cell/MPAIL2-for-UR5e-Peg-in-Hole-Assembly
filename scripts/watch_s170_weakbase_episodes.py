#!/usr/bin/env python3
"""Play sequential OmniReset episodes for the S170 weak-base comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("base", "mpail"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--base-checkpoint", type=Path, default=Path("logs/obs22_S55_MPAILModelWarmstart_skipPolicy_beta025_8env_30iter_20260529_011141/models/model_30.pt"))
parser.add_argument("--demo-path", type=Path, default=Path("mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt"))
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--num-episodes", type=int, default=5)
parser.add_argument("--max-steps-per-episode", type=int, default=180)
parser.add_argument("--print-every", type=int, default=20)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--reset-types", nargs="+", default=["ObjectAnywhereEEAnywhere"])
parser.add_argument("--mppi-blend", type=float, default=0.05)
parser.add_argument("--raw-action-limit", nargs=7, type=float, default=[40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--summary-json", type=Path, default=None)
parser.add_argument("--no-render", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402

from mpail2.envs.isaac.omnireset_wrapper import OmniResetMPAILWrapper, compact_omnireset_demonstrations  # noqa: E402
from mpail2.learner import MPAIL2Learner  # noqa: E402
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig  # noqa: E402
from mpail2.train.utils.demos import load_demonstrations  # noqa: E402
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env  # noqa: E402


def _prefixed_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _make_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(215, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Linear(256, 512),
        nn.LayerNorm(512),
    )


def _make_policy() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 98),
    )


def _load_demo_normalizer(path: Path, device: str):
    demo = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(demo, dict) and "demonstrations" in demo:
        obs = demo["demonstrations"]["policy"].reshape(-1, 215).float()
    elif isinstance(demo, dict) and "data" in demo:
        obs0 = demo["data"]["obs"]["policy"].float()
        obs1 = demo["data"]["next_obs"]["policy"].float()
        obs = torch.cat([obs0, obs1], dim=0).reshape(-1, 215)
    else:
        raise RuntimeError(f"Unsupported demo format: {path}")
    mean = obs.mean(dim=0).to(device)
    std = torch.clamp(obs.std(dim=0), min=0.03).to(device)
    return mean, std


def _build_base_policy(checkpoint: Path, demo_path: Path, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    encoder = _make_encoder().to(device).eval()
    policy = _make_policy().to(device).eval()
    encoder.load_state_dict(_prefixed_state(state, "encoder.coder."))
    policy.load_state_dict(_prefixed_state(state, "sampling.policy.model."))
    mean, std = _load_demo_normalizer(demo_path, device)
    print(f"[INFO] Loaded base policy checkpoint iter={ckpt.get('iter', '?')}: {checkpoint}", flush=True)
    return encoder, policy, mean, std


def _base_action(obs, encoder, policy, mean, std, device: str):
    with torch.inference_mode():
        x = obs["policy"].to(device)
        x = torch.clamp((x - mean) / (std + 1e-8), -10.0, 10.0)
        z = encoder(x)
        out = policy(z)
        mu, _ = torch.chunk(out, 2, dim=-1)
        return torch.tanh(mu.view(mu.shape[0], 7, 7)[:, 0])


def _build_mpail(env, checkpoint: Path, base_checkpoint: Path, demo_path: Path, device: str):
    base_cfg = OmniResetStateLearnerConfig()
    base_cfg.identity_encoder = True
    learner_cfg = sync_learner_dims_from_env(base_cfg, env)
    learner_cfg.raw_base_policy_checkpoint = str(base_checkpoint)
    learner_cfg.raw_base_policy_obs_key = "raw_policy"
    learner_cfg.raw_base_policy_obs_dim = 215
    learner_cfg.raw_base_policy_normalizer_demo_path = str(demo_path)
    learner_cfg.raw_base_policy_normalizer_std_floor = 0.03
    learner_cfg.mppi_residual_base_policy = True
    learner_cfg.mppi_policy_blend = float(args_cli.mppi_blend)
    learner_cfg.mppi_policy_delta_l2_max = 0.03
    learner_cfg.mppi_policy_delta_max = 0.05
    learner_cfg.mppi_best_candidate_action = False
    learner_cfg.mppi_action_library_anchor_policy_size = 0
    learner_cfg.mppi_action_library_size = 0
    learner_cfg.planner_cfg.value_return_type = "avg"
    learner_cfg.planner_cfg.terminal_value_scale = 0.0
    learner_cfg.planner_cfg.reward_center_for_planning = True
    learner_cfg.planner_cfg.reward_clip_for_planning = 5.0
    learner_cfg.planner_cfg.sampling_cfg.num_rollouts = 512
    learner_cfg.planner_cfg.sampling_cfg.policy_proportion = 0.0
    learner_cfg.planner_cfg.sampling_cfg.max_std = 0.025
    learner_cfg.planner_cfg.sampling_cfg.min_std = 0.0025
    learner_cfg.planner_cfg.num_elites = 64
    learner_cfg.planner_cfg.sampling_cfg.num_timesteps = 3
    learner_cfg.planner_cfg.dynamics_cfg.residual = True

    demonstrations = load_demonstrations(str(demo_path), device=device)
    demonstrations = compact_omnireset_demonstrations(demonstrations, include_ee_to_hole="planner_v2")
    learner = MPAIL2Learner(demonstrations, 1, learner_cfg, device=device)
    print(f"[INFO] Loading S170 MPAIL checkpoint from CPU: {checkpoint}", flush=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current_state = learner.planner.state_dict()
    model_state = ckpt["model_state_dict"]
    compatible_state = {
        key: value
        for key, value in model_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    skipped = sorted(set(model_state) - set(compatible_state))
    current_state.update(compatible_state)
    print(
        f"[INFO] Applying S170 compatible planner state_dict: loaded={len(compatible_state)} "
        f"skipped={len(skipped)}",
        flush=True,
    )
    if skipped:
        print(
            "[INFO] S170 skipped keys: "
            + ", ".join(skipped[:12])
            + (" ..." if len(skipped) > 12 else ""),
            flush=True,
        )
    learner.planner.load_state_dict(current_state)
    learner.eval()
    print(
        f"[INFO] Loaded S170 MPAIL checkpoint iter={ckpt.get('iter', '?')}: {checkpoint} "
        f"with weak base {base_checkpoint} blend={args_cli.mppi_blend}",
        flush=True,
    )
    return learner


def _progress(env):
    try:
        return env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    except Exception:
        return None


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
    compact_dim = 215 if args_cli.mode == "base" else 60
    env = OmniResetMPAILWrapper(env, raw_action_limit=args_cli.raw_action_limit, compact_obs_dim=compact_dim)

    if args_cli.mode == "base":
        encoder, policy, mean, std = _build_base_policy(args_cli.checkpoint, args_cli.demo_path, device)
        actor = None
        title = "WEAK BASE"
    else:
        actor = _build_mpail(env, args_cli.checkpoint, args_cli.base_checkpoint, args_cli.demo_path, device)
        encoder = policy = mean = std = None
        title = "S170 MPAIL model_100"

    print(f"[PLAY] {title}: {args_cli.num_episodes} sequential episodes, num_envs=1", flush=True)
    obs, _ = env.reset()
    ep = 1
    ep_step = 0
    ep_return = torch.zeros(1, device=device)
    ep_success_seen = False
    ep_first_success_step = None
    ep_success_steps = 0
    ep_min_xyz = float("inf")
    ep_final_xyz = float("nan")
    episode_rows = []
    prog = _progress(env)
    try:
        while ep <= args_cli.num_episodes and simulation_app.is_running():
            if args_cli.mode == "base":
                actions = _base_action(obs, encoder, policy, mean, std, device)
            else:
                with torch.inference_mode():
                    actions = actor.act(obs)
            obs, rewards, terms, truncs, infos = env.step(actions.to(env.unwrapped.device))
            if not args_cli.no_render:
                env.unwrapped.sim.render()
            ep_return += rewards.to(device)
            ep_step += 1
            if prog is not None:
                step_success = bool(prog.success.detach().bool().any().item())
                step_xyz = float(prog.xyz_distance.detach().float().mean().item())
                ep_success_seen = ep_success_seen or step_success
                ep_success_steps += int(step_success)
                if step_success and ep_first_success_step is None:
                    ep_first_success_step = ep_step
                ep_min_xyz = min(ep_min_xyz, step_xyz)
                ep_final_xyz = step_xyz
            done = bool((terms | truncs).any()) or ep_step >= args_cli.max_steps_per_episode
            if ep_step % args_cli.print_every == 0 or done:
                if prog is not None:
                    success = float(prog.success.float().mean().item())
                    xyz = float(prog.xyz_distance.mean().item())
                    extra = f" success={success:.3f} xyz={xyz:.4f}"
                else:
                    extra = ""
                info_log = infos.get("log", {}) if isinstance(infos, dict) else {}
                end_pos = info_log.get("Metrics/task_command/end_of_episode_pos_align_error")
                end_succ = info_log.get("Metrics/task_command/end_of_episode_success_rate")
                avg_pos = info_log.get("Metrics/task_command/average_pos_align_error")
                if end_pos is not None:
                    end_pos_f = float(torch.as_tensor(end_pos).float().mean().item())
                    extra += f" task_end_pos={end_pos_f:.4f} geo25={float(end_pos_f <= 0.025):.0f} geo50={float(end_pos_f <= 0.050):.0f}"
                if avg_pos is not None:
                    avg_pos_f = float(torch.as_tensor(avg_pos).float().mean().item())
                    extra += f" task_avg_pos={avg_pos_f:.4f}"
                if end_succ is not None:
                    end_succ_f = float(torch.as_tensor(end_succ).float().mean().item())
                    extra += f" task_succ={end_succ_f:.3f}"
                print(
                    f"[{title}] ep={ep}/{args_cli.num_episodes} step={ep_step:03d} "
                    f"ret={ep_return.mean().item():.3f} action_norm={actions.norm(dim=-1).mean().item():.3f}{extra}",
                    flush=True,
                )
            if done:
                row = {
                    "episode": ep,
                    "steps": ep_step,
                    "return": float(ep_return.item()),
                    "success_any": bool(ep_success_seen),
                    "first_success_step": ep_first_success_step,
                    "success_steps": int(ep_success_steps),
                    "min_xyz": float(ep_min_xyz),
                    "final_xyz": float(ep_final_xyz),
                }
                episode_rows.append(row)
                print(
                    f"[EPISODE_DONE] {title} ep={ep} steps={ep_step} return={ep_return.item():.3f} "
                    f"success_any={float(ep_success_seen):.0f} first_success={ep_first_success_step} "
                    f"success_steps={ep_success_steps} runner_step_mean={ep_success_steps / max(ep_step, 1):.6f} "
                    f"min_xyz={ep_min_xyz:.4f} final_xyz={ep_final_xyz:.4f}",
                    flush=True,
                )
                ep += 1
                if ep <= args_cli.num_episodes:
                    obs, _ = env.reset()
                    ep_step = 0
                    ep_return.zero_()
                    ep_success_seen = False
                    ep_first_success_step = None
                    ep_success_steps = 0
                    ep_min_xyz = float("inf")
                    ep_final_xyz = float("nan")
            if args_cli.sleep > 0:
                time.sleep(args_cli.sleep)
    finally:
        if episode_rows:
            success_rate = sum(row["success_any"] for row in episode_rows) / len(episode_rows)
            min_xyz_mean = sum(row["min_xyz"] for row in episode_rows) / len(episode_rows)
            final_xyz_mean = sum(row["final_xyz"] for row in episode_rows) / len(episode_rows)
            summary = {
                "checkpoint": str(args_cli.checkpoint),
                "base_checkpoint": str(args_cli.base_checkpoint),
                "mode": args_cli.mode,
                "mppi_blend": args_cli.mppi_blend,
                "seed": args_cli.seed,
                "num_episodes": len(episode_rows),
                "success_rate": success_rate,
                "success_episodes": int(sum(row["success_any"] for row in episode_rows)),
                "min_xyz_mean": min_xyz_mean,
                "final_xyz_mean": final_xyz_mean,
                "episodes": episode_rows,
            }
            print(
                f"[EVAL_SUMMARY] episodes={summary['num_episodes']} "
                f"success_rate={success_rate:.3f} success_episodes={summary['success_episodes']} "
                f"min_xyz_mean={min_xyz_mean:.4f} final_xyz_mean={final_xyz_mean:.4f}",
                flush=True,
            )
            if args_cli.summary_json is not None:
                args_cli.summary_json.parent.mkdir(parents=True, exist_ok=True)
                args_cli.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                print(f"[EVAL_SUMMARY_JSON] {args_cli.summary_json}", flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
