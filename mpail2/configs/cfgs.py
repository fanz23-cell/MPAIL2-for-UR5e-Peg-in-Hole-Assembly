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

    model_kwargs: dict = None

    model_factory: callable = mlp_factory

    residual: bool = False
    '''If True, dynamics predicts a latent delta added to the current latent.'''

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

    action_dim: int = None

    use_action: bool = False

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory


@dataclass(kw_only=True)
class EnsembleValueCfg:
    """Configuration for ensemble value function with multiple Q-networks"""

    class_type: Type['EnsembleValue'] = EnsembleValue

    state_dim: int = None

    action_dim: int = None

    num_q: int = None  # Number of Q-networks in ensemble (default from TD-MPC2)

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory

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


@dataclass(kw_only=True)
class PolicyLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    max_grad_norm: float = None
    '''Clips the gradient norm of the policy parameters to this value'''

    target_entropy: float = None
    '''Target entropy for automatic tuning. Common choice: -dim(action_space), e.g., -2 for 2D actions'''

    alpha_lr: float = None
    '''Learning rate for entropy coefficient (alpha) optimization'''

    lam: float = None
    '''Lambda parameter for Model-based vs Model-free trade-off'''


@dataclass(kw_only=True)
class ObsNormalizerCfg:
    """Configuration for observation normalization"""

    normalization_type: Literal["none", "cam_only", "fixed"] = None
    '''Type of normalization: none, cam_only, or fixed (uses expert demo statistics)'''

    eps: float = 1e-8
    '''Small constant for numerical stability in fixed normalization'''

    std_floor: float = 0.0
    '''Minimum standard deviation used by fixed normalization. 0 keeps old behavior.'''

    skip_low_std_dims: bool = False
    '''If True, fixed normalization passes through dimensions whose demo std is below std_floor.'''

###############################################################################
# VISUALIZATION CONFIGURATIONS
###############################################################################

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

    value_return_type: str = "avg"
    '''Ensemble value aggregation used for the terminal value in MPPI planning.'''

    terminal_value_scale: float = 1.0
    '''Multiplier applied to the terminal value used by MPPI planning.'''

    reward_center_for_planning: bool = False
    '''If True, subtract a learner-provided reward center when MPPI evaluates rollouts.'''

    reward_clip_for_planning: Optional[float] = None
    '''Optional symmetric clip applied to centered rollout rewards before MPPI weighting.'''

    base_plan_delta_penalty: float = 0.0
    '''Penalty subtracted from MPPI candidate scores for moving away from the provided base plan.'''

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
            if self.reward_cfg is not None:
                self.reward_cfg.action_dim = self.action_dim  # Reward
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

    reward_center_for_planning: bool = False
    '''If True, subtract the current reward center for value targets and MPPI planning.'''

    reward_center_mode: Literal["midpoint", "demo_mean"] = "midpoint"
    '''How to estimate the reward center from reward learner batches.'''

    reward_clip_for_planning: Optional[float] = None
    '''Optional symmetric clip applied to centered rewards used by value and planning.'''

    demo_action_bc_path: Optional[str] = None
    '''Optional raw expert dataset used to regularize the policy sampler toward expert actions.'''

    demo_action_bc_coeff: float = 0.0
    '''Weight for optional behavior-cloning loss on the policy sampler first action.'''

    demo_action_bc_pretrain_steps: int = 0
    '''Optional BC-only policy updates run before the first environment rollout.'''

    demo_action_bc_pretrain_batch_size: int = 256
    '''Batch size for optional BC-only policy pretraining.'''

    iql_pretrain_path: Optional[str] = None
    '''Optional expert transition dataset used for IQL-style warmstart of the same MPAIL models.'''

    iql_pretrain_steps: int = 0
    '''Number of IQL-style demo pretraining updates before the first environment rollout.'''

    iql_pretrain_batch_size: int = 512
    '''Batch size for IQL-style demo pretraining.'''

    iql_expectile: float = 0.7
    '''Expectile used when fitting the policy-action value proxy from demo-action Q values.'''

    iql_beta: float = 3.0
    '''Inverse temperature for advantage-weighted demo-action policy pretraining.'''

    iql_adv_clip: float = 100.0
    '''Maximum advantage weight used by IQL-style policy pretraining.'''

    iql_demo_reward: float = 1.0
    '''Constant reward assigned to expert demo transitions during IQL-style Q pretraining.'''

    iql_dynamics_coeff: float = 1.0
    '''Weight for same-transition latent dynamics pretraining inside the IQL warmstart.'''

    iqlearn_pretrain_path: Optional[str] = None
    '''Optional expert transition dataset used for IQ-Learn inverse-soft-Q warmstart.'''

    iqlearn_pretrain_steps: int = 0
    '''Number of IQ-Learn Q/policy updates before the first environment rollout.'''

    iqlearn_pretrain_batch_size: int = 512
    '''Batch size for IQ-Learn pretraining.'''

    iqlearn_alpha: float = 1.0
    '''Chi-square regularization alpha for IQ-Learn.'''

    iqlearn_init_temp: Optional[float] = None
    '''Optional SAC temperature used only to initialize IQ-Learn pretraining.'''

    iqlearn_learn_temp: bool = False
    '''If True, update SAC temperature during IQ-Learn pretraining.'''

    iqlearn_policy_batch_ratio: float = 1.0
    '''Policy/generated batch size as a multiple of the expert batch size in IQ-Learn pretraining.'''

    iqlearn_regularize_all: bool = True
    '''If True, apply IQ-Learn chi-square regularization to mixed expert+policy samples.'''

    iqlearn_log_every: int = 1000
    '''Print IQ-Learn pretraining diagnostics every this many steps.'''

    iqlearn_dynamics_coeff: float = 1.0
    '''Weight for expert-transition latent dynamics pretraining during IQ-Learn warmstart.'''

    iqlearn_pretrain_deterministic_policy: bool = False
    '''If True, IQ-Learn pretraining uses the policy mean with no entropy/log-prob term, so policy std is not trained offline.'''

    iqlearn_pretrain_bc_coeff: float = 0.0
    '''Optional behavior-cloning anchor on expert actions during IQ-Learn pretraining.'''

    iqlearn_reward_distill_steps: int = 0
    '''Optional steps to distill recovered IQ reward Q-gamma*V(next) into the MPAIL reward model.'''

    iqlearn_reward_distill_batch_size: int = 512
    '''Batch size for IQ reward distillation.'''

    iqlearn_reward_clip: Optional[float] = None
    '''Optional symmetric clip for recovered IQ rewards before reward-model distillation.'''

    iqlearn_online_steps: int = 0
    '''Number of online IQ-Learn value/policy updates per learner update, using expert demos plus replay.'''

    iqlearn_online_batch_size: int = 512
    '''Batch size for online IQ-Learn expert and replay samples.'''

    iqlearn_online_policy_batch_ratio: float = 1.0
    '''Current-policy action batch size as a multiple of the online IQ-Learn batch size.'''

    iqlearn_online_regularize_all: bool = True
    '''If True, apply online IQ-Learn chi-square regularization to expert+replay+policy samples.'''

    iqlearn_online_update_policy: bool = True
    '''If True, apply the IQ-Learn actor update online in addition to MPAIL policy updates.'''

    iqlearn_online_reward_distill_steps: int = 0
    '''Number of online updates that distill recovered IQ reward into the MPAIL reward model.'''

    iqlearn_online_reward_distill_batch_size: int = 512
    '''Batch size for online IQ reward distillation from expert demos plus replay.'''

    demo_success_replay_path: Optional[str] = None
    '''Optional raw expert dataset inserted into replay as protected demo-success experience.'''

    demo_success_replay_fraction: float = 0.0
    '''Fraction of replay trajectory slots to seed with protected demo-success experience.'''

    protect_demo_success_replay: bool = True
    '''If True, FIFO replay replacement preserves demo-success replay trajectories when possible.'''

    skip_updates: bool = False
    '''If True, process rollouts into replay but skip all learner gradient updates.'''

    skip_policy_updates: bool = False
    '''If True, skip MPAIL policy-gradient updates while still training dynamics/reward/value.'''

    skip_reward_updates: bool = False
    '''If True, skip discriminator/reward updates while still training dynamics/value/policy as configured.'''

    mppi_policy_blend: float = 1.0
    '''When using MPPI actions, blend from direct policy action to MPPI action: 0=policy, 1=MPPI.'''

    mppi_residual_base_policy: bool = False
    '''If True, initialize MPPI sampling from the protected base policy plan instead of the previous MPPI plan.'''

    mppi_action_library_size: int = 0
    '''Number of nearest-neighbor demo action plans injected into each MPPI candidate set.'''

    mppi_action_library_anchor_policy_size: int = 0
    '''Number of protected base-policy plans prepended to the MPPI action library candidate set.'''

    mppi_best_candidate_action: bool = False
    '''If True, execute the best-scoring sampled candidate instead of the MPPI weighted-average action.'''

    mppi_best_candidate_policy_margin: float = 0.0
    '''Required rollout-return advantage over the policy anchor before accepting a best candidate.'''

    mppi_best_candidate_score_mode: Literal["sum", "first"] = "sum"
    '''Score used to choose the best candidate: full imagined return or first-step reward only.'''

    mppi_best_candidate_delta_penalty: float = 0.0
    '''Penalty applied to candidate score per unit normalized L2 distance from the protected policy action.'''

    mppi_best_candidate_policy_cos_min: Optional[float] = None
    '''If set, best-candidate selection ignores non-anchor candidates whose first action is below this policy cosine.'''

    mppi_policy_delta_max: Optional[float] = None
    '''Optional per-env max absolute normalized action delta allowed before falling back to policy action.'''

    mppi_policy_delta_l2_max: Optional[float] = None
    '''Optional L2 norm budget for normalized MPPI-policy correction before applying blend.'''

    mppi_reward_gate_margin: Optional[float] = None
    '''If set, accept MPPI correction only when planner reward exceeds policy reward by this margin.'''

    mppi_policy_cos_min: Optional[float] = None
    '''If set, accept MPPI correction only when MPPI-policy action cosine is above this threshold.'''

    mppi_progress_gate_margin: Optional[float] = None
    '''If set, accept correction only when predicted peg-in-hole distance is no worse than policy by this margin.'''

    online_mppi_outcome_rank_coeff: float = 0.0
    '''Reward-ranking coefficient from realized MPPI correction outcomes in the real environment.'''

    online_mppi_outcome_rank_steps: int = 1
    '''Reward optimizer steps per rollout for realized MPPI outcome ranking.'''

    online_mppi_outcome_rank_batch_size: int = 256
    '''Batch size for realized MPPI outcome ranking.'''

    online_mppi_outcome_rank_buffer_size: int = 8192
    '''Maximum number of realized MPPI outcome samples to keep.'''

    online_mppi_outcome_rank_margin: float = 0.02
    '''Reward margin for realized outcome ranking.'''

    online_mppi_outcome_good_margin: float = 0.0
    '''Actual distance improvement required to label a correction as good.'''

    online_mppi_outcome_bad_margin: float = 0.0
    '''Actual distance regression required to label a correction as bad.'''

    online_mppi_outcome_horizon: int = 1
    '''Real environment steps used to label whether an MPPI correction helped closed-loop progress.'''

    online_mppi_outcome_good_weight: float = 1.0
    '''Weight for ranking good corrections above policy.'''

    online_mppi_outcome_bad_weight: float = 1.0
    '''Weight for ranking bad corrections below policy.'''

    online_mppi_candidate_rank_coeff: float = 0.0
    '''Reward-ranking coefficient for teacher-labeled MPPI candidate-pool good/bad actions.'''

    online_mppi_candidate_rank_steps: int = 1
    '''Reward optimizer steps per rollout for candidate-pool ranking.'''

    online_mppi_candidate_rank_batch_size: int = 256
    '''Batch size for candidate-pool ranking.'''

    online_mppi_candidate_rank_buffer_size: int = 8192
    '''Maximum number of candidate-pool ranking samples to keep.'''

    online_mppi_candidate_rank_margin: float = 0.02
    '''Reward margin for candidate-pool good/bad actions relative to policy.'''

    online_mppi_candidate_good_weight: float = 1.0
    '''Weight for ranking teacher-better candidates above policy.'''

    online_mppi_candidate_bad_weight: float = 1.0
    '''Weight for ranking teacher-worse candidates below policy.'''

    mppi_elite_adversarial_reward_coeff: float = 0.0
    '''Extra WGAN generator weight for MPPI elite imagined transitions.'''

    mppi_elite_adversarial_topk: int = 16
    '''Number of highest-return MPPI elites per env to cache as adversarial reward negatives.'''

    mppi_elite_adversarial_buffer_size: int = 32768
    '''Maximum number of flattened MPPI elite imagined transitions kept for reward updates.'''

    mppi_elite_adversarial_batch_size: int = 256
    '''Batch size sampled from MPPI elite adversarial buffer during reward updates.'''

    mppi_candidate_selector_enabled: bool = False
    '''If True, create and train a local same-state candidate selector head.'''

    mppi_candidate_selector_apply: bool = False
    '''If True, use the candidate selector to score MPPI best-candidate actions.'''

    mppi_candidate_selector_hidden_dim: int = 256
    '''Hidden dimension for the local candidate selector.'''

    mppi_candidate_selector_lr: float = 1e-4
    '''Learning rate for the local candidate selector.'''

    mppi_candidate_selector_margin: float = 0.02
    '''Pairwise margin for local candidate selector ranking.'''

    mpail_policy_loss_coeff: float = 1.0
    '''Scale for the model-based MPAIL policy objective before adding BC regularization.'''

    policy_update_start_iter: int = 0
    '''Skip MPAIL policy-gradient updates before this learner iteration.'''

    policy_trust_region_coeff: float = 0.0
    '''MSE penalty keeping updated policy mean near the pre-update mean on replay states.'''

    online_teacher_bc_steps: int = 0
    '''Optional supervised policy updates on online teacher obs/action pairs collected in the rollout.'''

    online_teacher_bc_batch_size: int = 256
    '''Batch size for optional online teacher BC updates.'''

    residual_adapter_enabled: bool = False
    '''If True, train a small residual adapter on online action labels.'''

    residual_adapter_apply: bool = False
    '''If True, direct policy actions are corrected by the learned residual adapter.'''

    residual_adapter_steps: int = 0
    '''Number of residual-adapter updates per online label batch.'''

    residual_adapter_batch_size: int = 256
    '''Batch size for residual-adapter updates.'''

    residual_adapter_lr: float = 1e-4
    '''Optimizer learning rate for the learned residual adapter.'''

    residual_adapter_hidden_dim: int = 256
    '''Hidden dimension for the learned residual adapter MLP.'''

    residual_adapter_scale: float = 1.0
    '''Multiplier applied to the learned residual before clamping the final action.'''

    residual_adapter_use_raw_obs: bool = False
    '''If True, train/apply the residual adapter from raw policy observations instead of planner latent state.'''

    mppi_residual_adapter_candidate_size: int = 0
    '''Number of residual-adapter plans injected into the MPPI candidate set.'''

    mppi_residual_adapter_candidate_max_scale: float = 1.0
    '''Largest residual scale used when injecting residual-adapter MPPI candidates.'''

    demo_action_bc_direct_steps: int = 0
    '''Use the pretrained BC policy mean directly for this many action calls before enabling MPPI.'''

    demo_action_direct_mode: Literal["policy", "knn"] = "policy"
    '''How to choose direct demo-action bootstrap actions: learned BC policy or nearest demo observation.'''

    raw_base_policy_checkpoint: Optional[str] = None
    '''Optional checkpoint whose raw-obs policy sampler is used as the protected base controller.'''

    raw_base_policy_obs_key: str = "raw_policy"
    '''Observation key used by the protected raw-obs base policy.'''

    raw_base_policy_obs_dim: int = 215
    '''Input dimension for the protected raw-obs base policy.'''

    raw_base_policy_normalizer_demo_path: Optional[str] = None
    '''Raw-observation demonstration file used to normalize the protected base controller input.'''

    raw_base_policy_normalizer_std_floor: float = 0.03
    '''Minimum std for protected base policy input normalization.'''

    demo_action_dyn_coeff: float = 0.0
    '''Weight for optional one-step dynamics loss on expert obs/action/next_obs transitions.'''

    demo_action_multistep_dyn_coeff: float = 0.0
    '''Weight for optional multi-step dynamics loss on consecutive expert action sequences.'''

    demo_action_multistep_dyn_horizon: int = 3
    '''Horizon for optional multi-step demo-action dynamics rollout loss.'''

    demo_action_dyn_pretrain_steps: int = 0
    '''Optional demo-action dynamics updates run before the first environment rollout.'''

    demo_action_dyn_pretrain_batch_size: int = 256
    '''Batch size for optional demo-action dynamics pretraining.'''

    demo_action_rank_coeff: float = 0.0
    '''Weight for optional ranking loss that makes expert actions predict demo next states better than zero/random actions.'''

    demo_action_rank_margin: float = 0.02
    '''Margin used by the optional demo-action dynamics ranking loss.'''

    demo_action_reward_rank_coeff: float = 0.0
    '''Weight for optional reward ranking loss that scores expert actions above policy/zero/random actions.'''

    demo_action_reward_rank_margin: float = 0.02
    '''Margin used by the optional demo-action reward ranking loss.'''

    demo_action_reward_rank_policy_weight: float = 1.0
    '''Extra multiplier for the expert-vs-policy term in the demo-action reward ranking loss.'''

    demo_action_reward_rank_mppi_weight: float = 0.0
    '''Extra multiplier for same-state expert/policy-vs-MPPI hard-negative reward ranking.'''

    demo_action_reward_rank_blended_weight: float = 0.0
    '''Extra multiplier for same-state expert/policy-vs-blended-MPPI reward ranking.'''

    demo_action_reward_rank_mppi_blend: Optional[float] = None
    '''Blend used when constructing the optional blended-MPPI hard negative. Defaults to mppi_policy_blend.'''

    online_mppi_reward_rank_coeff: float = 0.0
    '''Weight for reward ranking on policy-visited states with teacher/policy above MPPI hard negatives.'''

    online_mppi_reward_rank_margin: float = 0.02
    '''Margin used by online MPPI hard-negative reward ranking.'''

    online_mppi_reward_rank_mppi_weight: float = 1.0
    '''Multiplier for online teacher/policy-vs-MPPI ranking terms.'''

    online_mppi_reward_rank_policy_weight: float = 1.0
    '''Multiplier for online teacher-vs-policy ranking terms.'''

    online_mppi_reward_rank_blended_weight: float = 0.0
    '''Multiplier for online teacher/policy-vs-blended-MPPI ranking terms.'''

    online_mppi_next_state_rank_weight: float = 0.0
    '''Multiplier for same-action reward ranking that penalizes bad MPPI imagined next states.'''

    online_mppi_reward_rank_steps: int = 1
    '''Reward optimizer steps per rollout using the online MPPI hard-negative buffer.'''

    online_mppi_reward_rank_batch_size: int = 256
    '''Batch size for online MPPI hard-negative reward ranking.'''

    online_mppi_reward_rank_buffer_size: int = 8192
    '''Maximum policy-visited states/actions kept for online MPPI hard-negative reward ranking.'''

    dyn_agent_coeff: float = 1.0
    '''Weight for the standard replay/agent dynamics JEP loss.'''

    dyn_task_obs_coeff: float = 0.0
    '''Extra dynamics loss weight on task-critical planner observation slices.'''

    dyn_progress_coeff: float = 0.0
    '''Extra dynamics loss weight on predicted peg-in-hole distance progress.'''

    dyn_progress_sign_coeff: float = 0.0
    '''Extra dynamics loss weight on the sign of predicted peg-in-hole progress.'''

    dyn_progress_over_coeff: float = 0.0
    '''Extra dynamics loss weight penalizing optimistic over-prediction of peg-in-hole progress.'''

    dyn_counterfactual_progress_coeff: float = 0.0
    '''Extra dynamics loss weight keeping unverified random/zero counterfactual actions conservative.'''

    demo_action_raw_limit: Optional[list[float]] = None
    '''Raw action limits used to normalize expert actions into the learner action range.'''

    identity_encoder: bool = False
    '''If True, use the compact policy observation itself as the planner latent.'''

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

        video_enabled: bool = False
        '''Whether to record training videos.'''

        video_length: int = 0
        '''Maximum number of frames per training video. 0 records the full rollout iteration.'''

        video_resolution: tuple[int, int] = (640, 360)
        '''Training video resolution as (width, height).'''

        video_crf: int = 30
        '''Training video quality for mp4 encoding.'''

        video_fps: int = 10
        '''Frames per second for training video uploads.'''

    log_cfg: LogCfg = None

    seed: int = None
    '''Random seed'''

    vis_rollouts: bool = False
    '''Whether to visualize rollouts during training'''

    render_during_training: bool = False
    '''If True, actively render the simulator viewer during online rollout collection.'''

    render_interval_steps: int = 1
    '''Render every this many environment steps when render_during_training is enabled.'''
