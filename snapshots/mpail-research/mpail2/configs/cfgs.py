"""All configurations for MPAIL."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field, MISSING
from typing import Literal, Callable, Any, Optional, Type, List, Tuple, TYPE_CHECKING

from mpail2.layers import mlp_factory, cnn_factory, identity_factory

# Import actual classes for default class_type values
from mpail2.encoder import Coder, Decoder, CNNCoder, MultiCoder
from mpail2.dynamics import Dynamics
from mpail2.sampling import PolicySampling
from mpail2.reward import Reward
from mpail2.value import EnsembleValue
from mpail2.learner import MPAIL2Learner

if TYPE_CHECKING:
    from ..utils.rollout_vis import RolloutsVisualization

###############################################################################
# ENCODER CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class CoderCfg:

    input_dim: int = None
    '''input dimension'''

    output_dim: int = None
    '''output dimension'''

    obs_key: str = None

    ensemble_size: int = None

    class_type: Type['Coder'] = Coder
    '''class type for the encoder/decoder model'''


@dataclass(kw_only=True)
class IdentityCoderCfg(CoderCfg):
    '''Configuration for Identity Encoder/Decoder that returns input as is'''

    model_factory: callable = identity_factory
    '''model factory that creates an identity model'''

    model_kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class MLPCoderCfg(CoderCfg):

    model_factory: callable = mlp_factory

    model_kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class MLPDecoderCfg(MLPCoderCfg):
    '''Configuration for MLP Decoder'''

    model_kwargs: dict = field(default_factory=dict)

    class_type: Type['Decoder'] = Decoder


@dataclass(kw_only=True)
class CNNCoderCfg(CoderCfg):

    H: int = None
    W: int = None
    C: int = None
    '''height and width of the input image'''

    model_kwargs: dict = field(default_factory=dict)

    model_factory: callable = cnn_factory

    class_type: Type = CNNCoder


@dataclass(kw_only=True)
class MultiCoderCfg(MLPCoderCfg):
    '''Configuration for MultiEncoder that handles both images and state'''

    output_dim: int = None

    coder_list: List[CoderCfg] = None
    '''Dictionary mapping observation keys to their respective Encoder configurations'''

    model_kwargs: dict = field(default_factory=dict)

    model_factory: callable = mlp_factory

    class_type: Type['MultiCoder'] = MultiCoder

    def __post_init__(self):
        # Propagate output_dim to individual coders if not set
        if self.coder_list is not None and self.output_dim is not None:
            for coder_cfg in self.coder_list:
                if coder_cfg.output_dim is None:
                    coder_cfg.output_dim = self.output_dim


###############################################################################
# DYNAMICS CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class DynamicsCfg:
    """Configuration for latent dynamics models (operates on encoded states)."""

    class_type: Type['Dynamics'] = Dynamics
    '''class type for the dynamics model'''

    latent_dim: int = None
    '''latent space dimension (input/output dimension)'''

    action_dim: int = None
    '''action dimension'''

    ensemble_size: int = None

    model_kwargs: dict = None

    model_factory: callable = mlp_factory

###############################################################################
# SAMPLING CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class SamplingCfg:
    """Base configuration for sampling strategies."""

    action_dim: int = None
    """ Control dimension """

    noise: list[float] | float = None
    """ Noise """

    action_lims: List[Tuple[float, float]] | Tuple[float, float] = None
    """ Max control values """

    num_rollouts: int = None
    """ Number of rollouts """

    num_timesteps: int = None
    """ Number of timesteps. Equal to Horizon minus 1 """


@dataclass(kw_only=True)
class PolicySamplingCfg(SamplingCfg):
    """Configuration for PolicySampling - extends base SamplingCfg"""

    action_dim: int = None

    state_dim: int = None

    class_type: Type['PolicySampling'] = PolicySampling

    model_factory: callable = mlp_factory

    model_kwargs: dict = None

    policy_proportion: float = None
    '''Fraction of controls from policy vs noise'''

    min_std: float = None
    '''Minimum standard deviation for policy action distribution'''

    max_std: float = None
    '''Maximum standard deviation for policy action distribution'''

    noise: None = None
    '''NOT USED. NO NEED TO SPECIFY. DEFINED FOR BASE CLASS'''


###############################################################################
# REWARD CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class RewardCfg:

    class_type: Type['Reward'] = Reward

    state_dim: int = None

    ensemble_size: int = None

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory


@dataclass(kw_only=True)
class EnsembleValueCfg:
    """Configuration for ensemble value function with multiple Q-networks"""

    class_type: Type['EnsembleValue'] = EnsembleValue

    state_dim: int = None

    action_dim: int = None

    num_q: int = None  # Number of Q-networks in ensemble (default from TD-MPC2)

    ensemble_size: int = None

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory

    def __post_init__(self):
        if self.ensemble_size is None and self.num_q is not None:
            self.ensemble_size = self.num_q
        if self.num_q is None and self.ensemble_size is not None:
            self.num_q = self.ensemble_size

###############################################################################
# LEARNER CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class ValueLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    gamma: float = None
    '''Discount factor'''

    lam: float = None
    '''Lambda return parameter'''

    max_grad_norm: float = None
    '''Clips the gradient norm of the value function parameters to this value'''

    polyak_tau: float = None
    '''Polyak averaging coefficient: target = (1-tau) * target + tau * source'''


@dataclass(kw_only=True)
class RewardLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    gp_coeff: float = None
    '''Coefficient for gradient penalty term in WGAN loss'''

    gp_target_gradient: float = None
    '''Target gradient norm for gradient penalty'''

@dataclass(kw_only=True)
class DynamicsLearnerCfg:

    opt: str = None
    '''optimizer to use for training the dynamics model. Can be "adam", "sgd", etc.'''

    opt_params: dict = None
    '''Optimizer parameters for the dynamics model'''

    rho: float = None
    '''Temporal decay in JEP loss computation.'''

    recon_coeff: float = None
    '''Coefficient for the reconstruction loss term in the overall loss function. Only matters
    if reconstruction loss is used in the final loss. Otherwise decoder graph is detached from dynamics.'''

    enc_lr_scale: float = None
    '''Scaling factor for the encoder learning rate relative to the dynamics model learning rate.'''

    sigreg_coeff: float = None
    '''Coefficient for SIGReg auxiliary regularization loss. Disabled when <= 0.'''

    sigreg_knots: int = None
    '''Number of integration knots used by SIGReg.'''

    sigreg_num_proj: int = None
    '''Number of random projection vectors used by SIGReg.'''

@dataclass(kw_only=True)
class PolicyLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    max_grad_norm: float = None
    '''Clips the gradient norm of the policy parameters to this value'''

    target_entropy_per_action_dim: float = None
    '''Target entropy for automatic tuning. Common choice: -dim(action_space), e.g., -2 for 2D actions'''

    temp_lr: float = None
    '''Learning rate for temperature optimization'''

    lam: float = None
    '''Lambda parameter for Model-based vs Model-free trade-off'''

    use_intrinsic_bonus: bool = None
    '''Whether to add intrinsic exploration bonus during policy updates'''

    intrinsic_lambda_lr: float = None
    '''Learning rate for intrinsic lambda auto-tuning'''

    intrinsic_polyak_tau: float = None
    '''Polyak averaging coefficient for target policy used in lambda tuning'''


@dataclass(kw_only=True)
class ObsNormalizerCfg:
    """Configuration for observation normalization"""

    normalization_type: Literal["none", "cam_only", "fixed"] = None
    '''Type of normalization: none, cam_only, or fixed (uses expert demo statistics)'''

    eps: float = 1e-8
    '''Small constant for numerical stability in fixed normalization'''

###############################################################################
# VISUALIZATION CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class ImageRolloutVisConfig:
    """Visualise MPPI rollouts projected into camera image space.

    Decoded proprioception hand-positions are projected to pixel coordinates
    using camera intrinsics/extrinsics, then overlaid on the live camera frame.

    Camera setup (call one of these after env creation):
      • Simulation (MuJoCo/MetaWorld): ``vis.setup_camera(mj_model, camera_name)``
      • Simulation (IsaacLab):         ``vis.setup_camera_from_isaac(scene, sensor_names)``
      • Real robot:                    ``vis.setup_camera_from_config()``
            → fill in the ``realworld_*`` fields below first.
    """

    class_type: Type = None  # set to ImageSpaceRolloutsVisualization in __post_init__

    vis_rollouts: bool = True
    vis_n_envs: int = 1
    vis_n_rollouts: int = 20
    # If True, draw vis_n_rollouts sampled particles from the elite set instead
    # of the single optimal trajectory.  Rollouts are drawn with opacity scaled
    # by rank (best = fully opaque, worst ≈ 25 % opacity).
    sampled_particles: bool = True

    # Obs-dict key for the camera image (produced by camera_obs_key(camera_name))
    camera_obs_key: str = "cam_corner"

    # MuJoCo camera name used to extract intrinsics via setup_camera()
    camera_name: str = "corner"

    image_h: int = 84
    image_w: int = 84

    # Flip flags – must match the MetaWorldMultiCameraRenderer flags for this camera
    flip_vertical: bool = True    # DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA["corner"]
    flip_horizontal: bool = True  # corner is in DEFAULT_FLIP_RGB_HORIZONTAL_AFTER_VERTICAL

    # Slice of decoded proprioception that contains hand XYZ (MT1 default: first 3 dims)
    hand_xyz_slice: tuple = (0, 3)

    cost_range: Optional[tuple] = None

    # Physical radius (metres) used to scale projected dot sizes by depth.
    # Dots are sized as fx * dot_physical_radius_m / z, so a larger value gives
    # bigger dots overall; tune this if dots are too small/large on your camera.
    dot_physical_radius_m: float = 0.025

    # -------------------------------------------------------------------------
    # Real-world camera parameters – fill these in for physical robot deployments
    # then call ``vis.setup_camera_from_config()`` instead of the sim setup calls.
    # -------------------------------------------------------------------------
    #
    # INTRINSICS
    # Obtain from cv2.calibrateCamera, the RealSense / ZED SDK, or your camera's
    # calibration file.  The intrinsic matrix is:
    #
    #   K = [[fx,  0, cx],
    #        [ 0, fy, cy],
    #        [ 0,  0,  1]]
    #
    realworld_fx: Optional[float] = None   # focal length x (pixels)
    realworld_fy: Optional[float] = None   # focal length y (pixels)
    realworld_cx: Optional[float] = None   # principal point x; defaults to image_w / 2
    realworld_cy: Optional[float] = None   # principal point y; defaults to image_h / 2
    #
    # EXTRINSICS – camera pose expressed in the robot base frame.
    # Obtain via hand-eye calibration (e.g. easy_handeye, ROS tf, or a calibration board).
    #
    realworld_cam_pos:  Optional[tuple] = None  # (x, y, z) in metres
    realworld_cam_quat: Optional[tuple] = None  # (w, x, y, z) unit quaternion
    #
    # CONVENTIONS
    # ros_convention=True  → camera uses +Z forward, Y down  (RealSense, ZED, most ROS drivers)
    # ros_convention=False → camera uses -Z forward, Y up    (MuJoCo / OpenCV convention)
    #
    realworld_ros_convention: bool = True
    realworld_obs_key: str = "camera"      # key in obs_dict for this camera's image
    # -------------------------------------------------------------------------

    def __post_init__(self):
        from ..utils.image_rollout_vis import ImageSpaceRolloutsVisualization
        if self.class_type is None:
            self.class_type = ImageSpaceRolloutsVisualization


@dataclass(kw_only=True)
class RolloutVisConfig:

    class_type: Type['RolloutsVisualization'] = None

    vis_rollouts: bool = None

    vis_n_envs: int = None

    vis_n_rollouts: int = None

    xlim: tuple = None

    ylim: tuple = None

    show_velocity: bool = None

    show_elevation: bool = None

    cost_range: tuple = None

    show_trajectory_trace: bool = None

    # State index configuration for visualization
    # Indices for current position (x0)
    pos_x_idx: int = 0  # ee_x for x-axis of plot
    pos_y_idx: int = 1  # ee_y for y-axis of plot
    pos_z_idx: int = 2  # ee_z for z-axis of plot

    # Indices for cube/object trajectory rollouts (right plot)
    cube_x_idx: int = 7  # obj_x
    cube_y_idx: int = 8  # obj_y
    cube_z_idx: int = 9  # obj_z

    # Indices for velocity
    vel_x_idx: int = 6
    vel_y_idx: int = 7

    # Index for yaw
    yaw_idx: int = 5

    def __post_init__(self):
        # Import here to avoid circular imports
        from ..utils.rollout_vis import RolloutsVisualization
        if self.class_type is None:
            self.class_type = RolloutsVisualization

        self.control_scatter_1_indices: tuple = (self.pos_x_idx, self.pos_z_idx)
        self.control_scatter_cube_left_indices: tuple = (self.cube_x_idx, self.cube_z_idx)
        self.control_scatter_2_indices: tuple = (self.cube_x_idx, self.cube_y_idx)
        self.control_scatter_3_indices: tuple = (self.pos_x_idx, self.pos_y_idx)
        self.rollout_x_idx: int = self.pos_x_idx
        self.rollout_y_idx: int = self.pos_y_idx
        self.rollout_z_idx: int = self.pos_z_idx
        self.cube_pos_x_idx: int = self.cube_x_idx
        self.cube_pos_y_idx: int = self.cube_y_idx

###############################################################################
# PLANNER CONFIGURATION
###############################################################################

@dataclass(kw_only=True)
class PlannerCfg:
    """Configuration for MPAIL planner."""

    obs_dim: Optional[int] = None
    '''Propagates observation dimension to lower level modules'''

    action_dim: int = None
    '''Propagates action dimension to lower level modules'''

    latent_dim: int = None
    '''Propagates latent dimension to lower level modules'''

    temperature: float = None
    '''Temperature of optimization step'''

    opt_iters: int = None
    '''Number of optimization iterations per action'''

    u_per_command: int = None
    '''Number of control commands per action'''

    planning_eval_type: Literal["avg", "sample"] = None
    '''Type of planning evaluation: avg (average of ensemble rollouts) or sample (sample from ensemble rollouts)'''

    planning_dyn_type: Literal["avg", "sample"] = None
    '''Type of planning dynamics: avg (average of ensemble rollouts) or sample (sample from ensemble rollouts)'''

    step_size: float = 1.0
    '''Blend factor for the newly optimized plan vs rolled previous plan (1.0 uses only the new plan).'''

    num_elites: int = None
    '''Number of elite samples for CEM-style optimization'''

    vis_cfg: RolloutVisConfig = None
    '''Visualization configuration. None disables visualization'''

    encoder_cfg: CoderCfg = None
    '''Configuration for the encoder (obs -> latent)'''

    decoder_cfg: CoderCfg = None
    '''Configuration for the decoder (latent -> obs), optional'''

    reward_cfg: RewardCfg = None
    '''Configuration for reward function'''

    value_cfg: EnsembleValueCfg = None
    '''Configuration for value function'''

    gamma: Optional[float] = None
    '''Discount factor for reward computation during planning'''

    random_act: bool = None
    '''If True, add random actions to the planning rollouts'''

    use_intrinsic_planning: bool = None
    '''If True, add dynamics-ensemble disagreement bonus to MPPI costs (same formula as policy intrinsic).'''

    intrinsic_lambda_dyn_init: float = None
    '''Initial value of the learnable Lagrangian dual `lambda_dyn` weighting the
    dynamics-ensemble disagreement bonus.'''

    intrinsic_lambda_rew_init: float = None
    '''Initial value of the learnable Lagrangian dual `lambda_rew` weighting the
    reward-ensemble disagreement bonus.'''

    intrinsic_lambda_val_init: float = None
    '''Initial value of the learnable Lagrangian dual `lambda_val` weighting the
    value-ensemble disagreement bonus.'''

    intrinsic_norm_eps: float = None
    '''Epsilon added to the summed intrinsic disagreement bonus for numerical
    stability.'''

    sampling_cfg: PolicySamplingCfg = None
    '''Configuration for sampling module'''

    dynamics_cfg: DynamicsCfg = None
    '''Configuration for dynamics model'''

    def __post_init__(self):
        # Set default encoder config if not provided
        if self.encoder_cfg is None:
            self.encoder_cfg = MLPCoderCfg()

        # If obs_dim, action_dim, latent_dim are provided, propagate to lower level configs
        if self.obs_dim is not None:
            self.encoder_cfg.input_dim = self.obs_dim  # Encoder input
            if self.decoder_cfg is not None:
                self.decoder_cfg.output_dim = self.obs_dim  # Decoder output
        if self.action_dim is not None:
            if self.value_cfg is not None:
                self.value_cfg.action_dim = self.action_dim  # Value
            self.dynamics_cfg.action_dim = self.action_dim  # Dynamics
            self.sampling_cfg.action_dim = self.action_dim  # Policy
        if self.latent_dim is not None:
            self.sampling_cfg.state_dim = self.latent_dim  # Policy
            if self.reward_cfg is not None:
                self.reward_cfg.state_dim = self.latent_dim  # Reward
            if self.value_cfg is not None:
                self.value_cfg.state_dim = self.latent_dim  # Value
            self.dynamics_cfg.latent_dim = self.latent_dim  # Dynamics
            self.encoder_cfg.output_dim = self.latent_dim  # Encoder output
            if hasattr(self.encoder_cfg, '__post_init__'):
                self.encoder_cfg.__post_init__()  # Run encoder propagation if needed
            if self.decoder_cfg is not None:
                self.decoder_cfg.input_dim = self.latent_dim  # Decoder input


###############################################################################
# TOP-LEVEL CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class MPAIL2LearnerCfg:

    class_type: Type['MPAIL2Learner'] = MPAIL2Learner
    '''Type of the MPAIL learner class'''

    replay_ratio: float = None
    '''Number of gradient updates per new environment step collected'''

    replay_size: int = None
    '''Size of the replay buffer'''

    replay_batch_size: int = None
    '''Batch size for replay buffer sampling'''

    loss_horizon: int = None
    '''Number of steps to consider for loss computation'''

    use_terminations: bool = None
    '''Whether to use terminal mask in value function updates.
    When True, the next state value is set to 0 for terminal transitions,
    implementing the standard RL (1 - done) * V(s') term.'''

    # Planner
    planner_cfg: PlannerCfg = None
    '''Configuration for MPPI module'''

    # Dynamics
    dynamics_learner_cfg: DynamicsLearnerCfg = None

    # Reward
    reward_learner_cfg: RewardLearnerCfg = None
    '''Configuration for reward learning algorithm (WGAN-style adversarial training)'''

    # Value
    value_learner_cfg: ValueLearnerCfg = None
    '''Configuration for value function'''

    # Policy
    policy_learner_cfg: PolicyLearnerCfg = None

    # Observation Normalizer
    obs_normalizer_cfg: ObsNormalizerCfg = None

    iqlearn_pretrain_path: Optional[str] = None
    '''Optional expert transition dataset used for IQ-Learn inverse-soft-Q warmstart.'''

    iqlearn_pretrain_steps: int = 0
    '''Number of IQ-Learn Q/policy/dynamics updates before the first environment rollout.'''

    iqlearn_pretrain_batch_size: int = 512
    '''Batch size for IQ-Learn pretraining.'''

    iqlearn_alpha: float = 1.0
    '''Chi-square regularization alpha for IQ-Learn.'''

    iqlearn_init_temp: Optional[float] = None
    '''Optional entropy temperature used to initialize IQ-Learn pretraining.'''

    iqlearn_learn_temp: bool = False
    '''If True, update entropy temperature during IQ-Learn pretraining.'''

    iqlearn_policy_batch_ratio: float = 1.0
    '''Policy/generated batch size as a multiple of the expert batch size.'''

    iqlearn_regularize_all: bool = True
    '''If True, apply IQ-Learn chi-square regularization to mixed expert+policy samples.'''

    iqlearn_log_every: int = 1000
    '''Print IQ-Learn pretraining diagnostics every this many steps.'''

    iqlearn_dynamics_coeff: float = 1.0
    '''Weight for expert-transition latent dynamics pretraining during IQ-Learn warmstart.'''

    iqlearn_pretrain_deterministic_policy: bool = False
    '''If True, use policy mean and no entropy/log-prob term during offline IQ-Learn actor updates.'''

    iqlearn_pretrain_bc_coeff: float = 0.0
    '''Optional behavior-cloning anchor on expert actions during IQ-Learn pretraining.'''

    iqlearn_reward_distill_steps: int = 0
    '''Optional steps to distill recovered IQ reward Q-gamma*V(next) into the MPAIL reward model.'''

    iqlearn_reward_distill_batch_size: int = 512
    '''Batch size for IQ reward distillation.'''

    iqlearn_reward_clip: Optional[float] = None
    '''Optional symmetric clip for recovered IQ rewards before reward-model distillation.'''

    demo_success_replay_path: Optional[str] = None
    '''Optional raw expert dataset inserted into replay as protected demo-success experience.'''

    demo_success_replay_fraction: float = 0.0
    '''Fraction of replay trajectory slots to seed with protected demo-success experience.'''

    protect_demo_success_replay: bool = True
    '''If True, FIFO replay replacement preserves demo-success replay trajectories when possible.'''

    skip_reward_updates: bool = False
    '''If True, skip adversarial reward updates while still training dynamics/value/policy.'''

    demo_action_raw_limit: Optional[list[float]] = None
    '''Raw action limits used to normalize demonstration actions into [-1, 1].'''


@dataclass(kw_only=True)
class MPAIL2RunnerCfg:

    path_to_demonstrations: str = None
    '''Number of steps per environment'''

    resume_from_checkpoint: str = None
    '''Optional checkpoint used to initialize the planner before training'''

    resume_partial_ignore_mismatch: bool = False
    '''If True, warm-start only checkpoint tensors whose names and shapes match the current model.'''

    action_scale: list[float] = None
    '''Optional scale applied to learner actions before env.step.'''

    teacher_policy_path: str = None
    '''Optional TorchScript teacher policy used for online bootstrap rollouts.'''

    teacher_bootstrap_steps: int = 0
    '''Number of environment steps to execute with the online teacher policy.'''

    teacher_blend_steps: int = 0
    '''Number of environment steps to linearly blend from teacher actions to learner actions.'''

    policy_control_after_online_teacher_bc: bool = False
    '''If True, use teacher control only for the first rollout, then let the learner act.'''

    teacher_diag_always: bool = False
    '''If True, compute teacher-action diagnostics even when the teacher is not controlling.'''

    teacher_dagger_collect: bool = False
    '''If True, add teacher labels from policy-visited states to online BC batches.'''

    teacher_dagger_buffer_size: int = 0
    '''Maximum aggregated DAgger labels to keep. 0 disables persistence across rollouts.'''

    teacher_dagger_export_path: Optional[str] = None
    '''Optional path to export the aggregated DAgger obs/action buffer as a BC dataset.'''

    teacher_dagger_label_mode: Literal["teacher", "executed"] = "teacher"
    '''Whether DAgger BC targets are teacher actions or actually executed actions.'''

    teacher_intervention_cos_threshold: Optional[float] = None
    '''If set, execute teacher action when learner/teacher cosine falls below this threshold.'''

    teacher_residual_beta: float = 0.0
    '''Blend factor for teacher residual correction on policy-control steps.'''

    teacher_residual_final_beta: Optional[float] = None
    '''Optional final residual beta, linearly annealed over learning iterations.'''

    teacher_oracle_mppi_gate: bool = False
    '''Diagnostic only: execute MPPI/blended correction only when it is closer to teacher than the base policy.'''

    teacher_oracle_mppi_pool_select: bool = False
    '''Diagnostic only: gate using the best teacher-matching action from the full MPPI candidate pool.'''

    teacher_oracle_mppi_margin: float = 0.0
    '''Normalized-action L2 margin required for diagnostic teacher-oracle MPPI gate.'''

    failure_strat_export_path: Optional[str] = None
    '''Optional JSONL path for per-episode OmniReset failure stratification diagnostics.'''

    failure_strat_label: str = "policy"
    '''Label stored in failure-stratification JSONL records, e.g. base or teacher.'''

    learner_cfg: MPAIL2LearnerCfg = None
    '''Configuration for MPAIL learner'''

    num_learning_iterations: int = None
    '''Number of learning iterations'''

    logger: Optional[Literal["wandb"]] = None
    '''Logger type'''

    @dataclass(kw_only=True)
    class LogCfg:

        logger: Optional[Literal["wandb"]] = None
        '''Logger type'''

        checkpoint_every: int = None
        '''Frequency of saving model checkpoints during training (in learning iterations)'''

        no_wandb: bool = False
        '''Whether to disable Weights & Biases logging even if logger is set to "wandb"'''

        log_dir: str = None
        '''Directory to save logs and checkpoints'''

        video_interval: int = None
        '''Frequency of logging rollout videos during training (in learning iterations)'''

    log_cfg: LogCfg = None

    seed: int = None
    '''Random seed'''

    vis_rollouts: bool = False
    '''Whether to visualize rollouts during training'''
