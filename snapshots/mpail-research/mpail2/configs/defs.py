from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass, field, MISSING
import mpail2.configs.cfgs as cfgs

'''
Default configurations for MPAIL.

The following settings MUST be set for each task:
1) encoder_cfg
2) action_dim
3) latent_dim

'''

#### SHARED CONSTANTS ####
OPT = "adam"
LR = 3e-4
HORIZON = 7
OPT_ITERS = 5
GAMMA = 0.99
LAM = 0.95
ENSEMBLE_SIZE = 5

MODEL_KWARGS = {
    "hidden_dims": [512, 512],
    "activation": "silu",
    "use_layer_norm": True,
    "use_weight_norm": True,
}
CNN_MODEL_KWARGS = {
    "out_channels": [32, 32, 32, 32],
    "kernel_sizes": [3, 3, 3, 3],
    "strides": [2, 2, 2, 1],
    "activation": "silu",
    "use_weight_norm": True,
}
OPT_PARAMS = {"lr": LR}

# Add these settings:
# 1) ACTION_DIM
# 2) LATENT_DIM
# 3) Encoder Configs

##############################
### Planner Configurations ###
##############################

#
# Reward and Value
#

@dataclass(kw_only=True)
class RewardConfig(cfgs.RewardCfg):
    state_dim: int = None
    ensemble_size: int = ENSEMBLE_SIZE
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        # "use_layer_norm": False,
        "disable_output_bias": True,
    })

@dataclass(kw_only=True)
class EnsembleValueConfig(cfgs.EnsembleValueCfg):
    state_dim: int = None
    action_dim: int = None
    ensemble_size: int = ENSEMBLE_SIZE
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
    })

#
# Encoders
#

@dataclass(kw_only=True)
class MLPCoderConfig(cfgs.MLPCoderCfg):
    ensemble_size: int = 1
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        ## OVERRIDES ##
        "hidden_dims": [256, 256],
        "override_last_layer_norm": True,
    })

@dataclass(kw_only=True)
class CNNCoderConfig(cfgs.CNNCoderCfg):
    ensemble_size: int = 1
    model_kwargs: dict = field(default_factory=lambda: {
        **CNN_MODEL_KWARGS,
    })

@dataclass(kw_only=True)
class MultiCoderConfig(cfgs.MultiCoderCfg):
    ensemble_size: int = 1

    @dataclass(kw_only=True)
    class ProprioCoderConfig(cfgs.MLPCoderCfg):
        ensemble_size: int = 1
        model_kwargs:dict = field(default_factory=lambda: {
            **MODEL_KWARGS,
            ## OVERRIDES ##
            "hidden_dims": [256],
            # "use_layer_norm": False,
            "override_last_layer_norm": True,
            # "override_last_layer_activation": True,
        })

    @dataclass(kw_only=True)
    class MultiCNNCoderConfig(cfgs.CNNCoderCfg):
        ensemble_size: int = 1
        model_kwargs:dict = field(default_factory=lambda: {
            **CNN_MODEL_KWARGS,
            ## OVERRIDES ##
            "override_last_layer_norm": True,
            # "override_last_layer_activation": True,
        })

    model_kwargs:dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        ## OVERRIDES ##
        "override_last_layer_norm": True,
    })

# FOR VISUALIZATION PURPOSES ONLY
@dataclass(kw_only=True)
class MLPDecoderConfig(cfgs.MLPDecoderCfg):
    '''Configuration for MLP Decoder'''

    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        ## OVERRIDES ##
        # "use_layer_norm": False,
    })

#
# Dynamics
#

@dataclass(kw_only=True)
class DynamicsConfig(cfgs.DynamicsCfg):

    x_dim: int = None
    '''agent state dimension'''

    action_dim: int = None
    '''action dimension'''

    latent_dim: int = None
    '''dimension of the latent space'''

    ensemble_size: int = ENSEMBLE_SIZE

    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        "override_last_layer_norm": True,
    })

#
# Sampling/Policy
#

@dataclass(kw_only=True)
class PolicySamplingConfig(cfgs.PolicySamplingCfg):

    action_dim: int = None
    state_dim: int = None

    action_lims: list[Tuple[float, float]] | Tuple[float, float] = (-1., 1.)
    num_rollouts: int = 512
    num_timesteps: int = HORIZON
    policy_proportion: float = 0.05
    min_std: float = 0.05
    max_std: float = 2.0
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
    })

#
# Planner
#

@dataclass(kw_only=True)
class PlannerConfig(cfgs.PlannerCfg):

    # THESE MUST BE SET FOR EACH TASK
    encoder_cfg: cfgs.CoderCfg = MISSING
    action_dim: int = MISSING

    random_act: bool = True
    latent_dim: int = 512
    num_elites: int = 64
    temperature: float = 1.0
    opt_iters: int = OPT_ITERS

    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    value_cfg: EnsembleValueConfig = field(default_factory=EnsembleValueConfig)
    sampling_cfg: PolicySamplingConfig = field(default_factory=PolicySamplingConfig)
    dynamics_cfg: DynamicsConfig = field(default_factory=DynamicsConfig)

    seed: int = 42
    u_per_command: int = 1
    step_size: float = 1.0
    debug: bool = False

    planning_eval_type: Literal["min", "avg", "sample"] = "sample"
    planning_dyn_type: Literal["avg", "sample"] = "sample"

    use_intrinsic_planning: bool = False
    add_rew_bonus: bool = True
    add_dyn_bonus: bool = False
    add_val_bonus: bool = True
    # intrinsic_lambda_dyn_init: float = 0.1
    # intrinsic_lambda_rew_init: float = 0.1
    # intrinsic_lambda_val_init: float = 0.1
    intrinsic_lambda_dyn_init: float = 1.0
    intrinsic_lambda_rew_init: float = 1.0
    intrinsic_lambda_val_init: float = 1.0
    intrinsic_norm_eps: float = 1e-6

##############################
### Learner Configurations ###
##############################

@dataclass(kw_only=True)
class ValueLearnerConfig(cfgs.ValueLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    gamma: float = GAMMA

    lam: float = LAM

    # max_grad_norm: float = 5.0
    max_grad_norm: float = None

    polyak_tau: float = 0.01

@dataclass(kw_only=True)
class RewardLearnerConfig(cfgs.RewardLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    gp_coeff: float = 100.0

    gp_target_gradient: float = 1.0

@dataclass(kw_only=True)
class PolicyLearnerConfig(cfgs.PolicyLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    # max_grad_norm: float = 1.0
    max_grad_norm: float = None

    target_entropy_per_action_dim: float = -0.5

    temp_lr: float = LR

    lam: float = LAM

    # For information bonus
    use_intrinsic_bonus: bool = True
    intrinsic_polyak_tau: float = 0.01
    intrinsic_lambda_lr: float = LR

@dataclass(kw_only=True)
class DynamicsLearnerConfig(cfgs.DynamicsLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    enc_lr_scale: float = 0.01

    rho: float = 1.0

    recon_coeff: float = 1.0 # Exact coefficient doesn't matter if not using recon loss

    sigreg_coeff: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

@dataclass(kw_only=True)
class LearnerConfig(cfgs.MPAIL2LearnerCfg):

    # Policy
    planner_cfg: cfgs.PlannerCfg = MISSING

    # Training parameters
    replay_ratio: float = 1.0

    # Replay buffer settings
    replay_size: int = 100_000
    replay_batch_size: int = 256

    # Loss horizon for trajectory-based updates
    loss_horizon: int = HORIZON  # Match num_timesteps in sampling_cfg

    # Dynamics - ENABLED for training
    dynamics_learner_cfg: cfgs.DynamicsLearnerCfg = field(default_factory=DynamicsLearnerConfig)

    # Reward (adversarial training)
    reward_learner_cfg: cfgs.RewardLearnerCfg = field(default_factory=RewardLearnerConfig)

    # Value
    value_learner_cfg: cfgs.ValueLearnerCfg = field(default_factory=ValueLearnerConfig)

    # Policy Learner
    policy_learner_cfg: cfgs.PolicyLearnerCfg = field(default_factory=PolicyLearnerConfig)

    iqlearn_pretrain_path: str | None = None
    iqlearn_pretrain_steps: int = 0
    iqlearn_pretrain_batch_size: int = 512
    iqlearn_alpha: float = 1.0
    iqlearn_init_temp: float | None = None
    iqlearn_learn_temp: bool = False
    iqlearn_policy_batch_ratio: float = 1.0
    iqlearn_regularize_all: bool = True
    iqlearn_log_every: int = 1000
    iqlearn_dynamics_coeff: float = 1.0
    iqlearn_pretrain_deterministic_policy: bool = False
    iqlearn_pretrain_bc_coeff: float = 0.0
    iqlearn_reward_distill_steps: int = 0
    iqlearn_reward_distill_batch_size: int = 512
    iqlearn_reward_clip: float | None = None
    demo_success_replay_path: str | None = None
    demo_success_replay_fraction: float = 0.0
    protect_demo_success_replay: bool = True
    skip_reward_updates: bool = False
    demo_action_raw_limit: list[float] | None = None
