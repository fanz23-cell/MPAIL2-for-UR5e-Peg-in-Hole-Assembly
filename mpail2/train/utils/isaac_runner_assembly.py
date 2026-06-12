"""Runtime assembly helpers for Isaac image training."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import mpail2.configs.defs as defs
from mpail2.configs.cfgs import MPAIL2RunnerCfg
from mpail2.train.utils.demos import resolve_demo_path
from mpail2.train.utils.runtime import make_runner_log_cfg, maybe_init_wandb


def assemble_isaac_runner(
    cfg: Any,
    learner_cfg: defs.LearnerConfig,
    *,
    demo_env_var: str,
    default_demo_rel: str,
    default_num_iterations: int,
    default_seed: int = 0,
) -> MPAIL2RunnerCfg:
    r = cfg.runner
    log = cfg.log

    demo = resolve_demo_path(
        r.path_to_demonstrations,
        demo_env_var,
        default_demo_rel,
    )

    logger = r.logger if r.logger is not None else ("wandb" if not log.no_wandb else None)
    num_iter = r.num_learning_iterations if r.num_learning_iterations is not None else default_num_iterations
    seed = r.seed if r.seed is not None else default_seed

    runner_log = make_runner_log_cfg(log, logger=logger)
    vis_rollouts = r.vis_rollouts if r.vis_rollouts is not None else log.video

    runner_cfg = MPAIL2RunnerCfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=num_iter,
        path_to_demonstrations=demo,
        resume_from_checkpoint=r.resume_from_checkpoint,
        resume_partial_ignore_mismatch=r.resume_partial_ignore_mismatch,
        action_scale=r.action_scale,
        teacher_policy_path=r.teacher_policy_path,
        teacher_bootstrap_steps=r.teacher_bootstrap_steps,
        teacher_blend_steps=r.teacher_blend_steps,
        policy_control_after_online_teacher_bc=r.policy_control_after_online_teacher_bc,
        teacher_diag_always=r.teacher_diag_always,
        teacher_dagger_collect=r.teacher_dagger_collect,
        teacher_dagger_buffer_size=r.teacher_dagger_buffer_size,
        teacher_dagger_export_path=r.teacher_dagger_export_path,
        teacher_dagger_label_mode=r.teacher_dagger_label_mode,
        teacher_intervention_cos_threshold=r.teacher_intervention_cos_threshold,
        teacher_residual_beta=r.teacher_residual_beta,
        teacher_residual_final_beta=r.teacher_residual_final_beta,
        teacher_oracle_mppi_gate=r.teacher_oracle_mppi_gate,
        teacher_oracle_mppi_pool_select=r.teacher_oracle_mppi_pool_select,
        teacher_oracle_mppi_margin=r.teacher_oracle_mppi_margin,
        failure_strat_export_path=r.failure_strat_export_path,
        failure_strat_label=r.failure_strat_label,
        render_during_training=r.render_during_training,
        render_interval_steps=r.render_interval_steps,
        seed=seed,
        logger=logger,
        log_cfg=runner_log,
        vis_rollouts=vis_rollouts,
    )

    model_save_path = f"{log.run_log_dir}/{log.model_save_dirname}"
    os.makedirs(log.run_log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    maybe_init_wandb(
        log_cfg=log,
        run_name=log.run_name,
        run_dir=log.run_log_dir,
        config_obj=asdict(runner_cfg),
    )

    return runner_cfg
