"""Image-space rollout visualisation for visual manipulation tasks.

Projects the optimal decoded hand-position rollout into the pixel space of one
or more named cameras, overlaying a solid green line on the live camera frame.

Usage (Isaac):
    1. Build ``ImageRolloutVisConfig`` and pass it as ``vis_cfg`` in ``PlannerConfig``.
    2. Call ``vis.setup_camera_from_isaac(scene, sensor_names=("table_cam", "wrist_cam"))``
       after ``env.reset()``.  The wrist camera is treated as dynamic: its world pose
       is re-fetched every call to ``update()``.
    3. ``Planner.create_vis()`` calls ``update()`` each optimisation step; the runner
       captures ``vis.fig`` into an MP4 and logs it to wandb.

Usage (MetaWorld / MuJoCo):
    1. Build config, call ``vis.setup_camera(mj_model, camera_name)`` once.
"""

from __future__ import annotations

import math
import os
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

if TYPE_CHECKING:
    from mpail2.configs.cfgs import ImageRolloutVisConfig


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix (columns = frame axes in parent)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)



# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class ImageSpaceRolloutsVisualization:
    """Render the optimal MPPI rollout as a solid green line on camera images.

    Supports an arbitrary list of cameras set up via ``setup_camera()`` (MuJoCo)
    or ``setup_camera_from_isaac()`` (IsaacLab).  Cameras labelled ``wrist_*``
    are treated as dynamic: their world pose is re-fetched from the Isaac sensor
    at every call to ``update()``.

    Figure layout: ``vis_n_envs`` rows × ``n_cameras`` columns.
    """

    def __init__(self, cfg: "ImageRolloutVisConfig") -> None:
        self.cfg = cfg
        self.vis_n_envs: int = cfg.vis_n_envs
        self.vis_n_rollouts: int = cfg.vis_n_rollouts
        self.image_h: int = cfg.image_h
        self.image_w: int = cfg.image_w
        self.hand_xyz_slice = slice(*cfg.hand_xyz_slice)

        # List of camera dicts; populated by setup_camera* methods.
        self._cameras: list[dict] = []

        self.reset()

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------

    def _n_cameras(self) -> int:
        return max(1, len(self._cameras))

    def reset(self) -> None:
        """Recreate the matplotlib figure for the current camera list."""
        if hasattr(self, "fig"):
            plt.close(self.fig)
        n_envs = self.vis_n_envs
        n_cams = self._n_cameras()
        w_inch = max(3.0, self.image_w / 21.0) * n_cams
        h_inch = max(3.0, self.image_h / 21.0) * n_envs
        self.fig, axes_raw = plt.subplots(
            n_envs, n_cams, figsize=(w_inch, h_inch), dpi=120, squeeze=False
        )
        # Always expose as 2-D list: axes[env_i][cam_j]
        self.axes = [[axes_raw[ei][cj] for cj in range(n_cams)] for ei in range(n_envs)]

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------

    def setup_camera(self, model, camera_name: str) -> None:
        """Add a MuJoCo / MetaWorld camera from the simulation model.

        Must be called after the environment is created.  Replaces any
        previously registered cameras (MetaWorld uses a single camera).

        Args:
            model: ``mujoco.MjModel`` from the environment.
            camera_name: Camera name in the MuJoCo XML (e.g. ``"corner"``).
        """
        import mujoco

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(
                f"Camera {camera_name!r} not found in MuJoCo model. "
                "Check the XML camera definition."
            )

        fovy_rad = math.radians(float(model.cam_fovy[cam_id]))
        fy = (self.image_h / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy  # MuJoCo assumes square pixels
        K = np.array(
            [[fx, 0.0, self.image_w / 2.0], [0.0, fy, self.image_h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self._cameras = [{
            "obs_key":       self.cfg.camera_obs_key,
            "label":         camera_name,
            "K":             K,
            "cam_pos":       np.array(model.cam_xpos[cam_id], dtype=np.float64),
            "cam_rot":       np.array(model.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3),
            "ros_convention": False,   # MuJoCo: -Z forward, Y up
            "flip_v":        self.cfg.flip_vertical,
            "flip_h":        self.cfg.flip_horizontal,
            "dynamic":       False,
            "sensor":        None,
            "robot":         None,
        }]
        self.reset()
        print(
            f"[ImageRolloutVis] Camera '{camera_name}': "
            f"fx={fx:.2f}  fy={fy:.2f}  pos={self._cameras[0]['cam_pos'].round(3)}"
        )

    def setup_camera_from_isaac(
        self,
        scene,
        sensor_names: tuple[str, ...] = ("table_cam",),
        robot_name: str = "robot",
    ) -> None:
        """Add IsaacLab cameras from the live scene.

        Must be called after ``env.reset()`` so sensor data is populated.
        Cameras whose name contains ``"wrist"`` are dynamic: their world pose
        is re-fetched from the sensor at every call to ``update()``.

        Args:
            scene: ``env.unwrapped.scene`` (IsaacLab ``InteractiveScene``).
            sensor_names: Ordered tuple of sensor names to add as columns.
            robot_name: Name of the robot articulation in the scene.
        """
        if isinstance(sensor_names, str):
            sensor_names = (sensor_names,)

        robot = scene[robot_name]
        robot_pos_w = robot.data.root_pos_w[0].cpu().numpy()
        robot_rot_w = _quat_wxyz_to_rot(robot.data.root_quat_w[0].cpu().numpy())

        self._cameras = []
        for sensor_name in sensor_names:
            sensor = (
                scene.sensors[sensor_name]
                if hasattr(scene, "sensors") and sensor_name in scene.sensors
                else scene[sensor_name]
            )
            K = sensor.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)

            # Initial pose (world frame → robot frame)
            cam_pos_w = sensor.data.pos_w[0].cpu().numpy()
            cam_rot_w = _quat_wxyz_to_rot(sensor.data.quat_w_ros[0].cpu().numpy())
            cam_pos = robot_rot_w.T @ (cam_pos_w - robot_pos_w)
            cam_rot = robot_rot_w.T @ cam_rot_w

            is_dynamic = "wrist" in sensor_name.lower()
            self._cameras.append({
                "obs_key":        sensor_name,
                "label":          sensor_name,
                "K":              K,
                "cam_pos":        cam_pos,
                "cam_rot":        cam_rot,
                "ros_convention": True,    # Isaac: +Z forward, Y down
                "flip_v":         False,
                "flip_h":         False,
                "dynamic":        is_dynamic,
                "sensor":         sensor if is_dynamic else None,
                "robot":          robot if is_dynamic else None,
            })
            kind = "dynamic" if is_dynamic else "static"
            print(
                f"[ImageRolloutVis] Isaac '{sensor_name}' ({kind}): "
                f"fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}  "
                f"cam_pos_robot={cam_pos.round(3)}"
            )

        self.reset()

    def setup_camera_from_config(self) -> None:
        """Add a real-world camera defined via ``ImageRolloutVisConfig`` fields.

        Use this instead of ``setup_camera()`` / ``setup_camera_from_isaac()``
        when training on a physical robot with a calibrated camera.

        Required config fields (``ImageRolloutVisConfig``):
            realworld_fx, realworld_fy  – focal lengths in pixels
            realworld_cam_pos           – (x, y, z) in the robot base frame (metres)
            realworld_cam_quat          – (w, x, y, z) rotation quaternion

        Optional config fields:
            realworld_cx / cy           – principal point; defaults to image centre
            realworld_ros_convention    – True (RealSense / ZED) or False (OpenCV / MuJoCo)
            realworld_obs_key           – obs-dict key for this camera's image tensor
        """
        cfg = self.cfg
        missing = [f for f in ("realworld_fx", "realworld_fy",
                                "realworld_cam_pos", "realworld_cam_quat")
                   if getattr(cfg, f, None) is None]
        if missing:
            raise ValueError(
                f"[ImageRolloutVis] Set these fields in ImageRolloutVisConfig "
                f"before calling setup_camera_from_config(): {missing}"
            )

        fx = float(cfg.realworld_fx)
        fy = float(cfg.realworld_fy)
        cx = float(cfg.realworld_cx) if getattr(cfg, "realworld_cx", None) is not None else self.image_w / 2.0
        cy = float(cfg.realworld_cy) if getattr(cfg, "realworld_cy", None) is not None else self.image_h / 2.0
        K  = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        cam_pos = np.array(cfg.realworld_cam_pos,  dtype=np.float64)
        cam_rot = _quat_wxyz_to_rot(np.array(cfg.realworld_cam_quat, dtype=np.float64))
        obs_key = getattr(cfg, "realworld_obs_key", "camera")

        self._cameras = [{
            "obs_key":        obs_key,
            "label":          obs_key,
            "K":              K,
            "cam_pos":        cam_pos,
            "cam_rot":        cam_rot,
            "ros_convention": getattr(cfg, "realworld_ros_convention", True),
            "flip_v":         False,
            "flip_h":         False,
            "dynamic":        False,
            "sensor":         None,
            "robot":          None,
        }]
        self.reset()
        print(
            f"[ImageRolloutVis] Real-world camera '{obs_key}': "
            f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}  "
            f"cam_pos_robot={cam_pos.round(3)}"
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _get_cam_extrinsics(self, cam: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam_pos, cam_rot) in robot base frame.

        Dynamic cameras (wrist) re-fetch pose from the Isaac sensor each call.
        """
        if not cam["dynamic"]:
            return cam["cam_pos"], cam["cam_rot"]

        sensor = cam["sensor"]
        robot  = cam["robot"]
        cam_pos_w = sensor.data.pos_w[0].cpu().numpy()
        cam_rot_w = _quat_wxyz_to_rot(sensor.data.quat_w_ros[0].cpu().numpy())
        robot_pos_w = robot.data.root_pos_w[0].cpu().numpy()
        robot_rot_w = _quat_wxyz_to_rot(robot.data.root_quat_w[0].cpu().numpy())
        return robot_rot_w.T @ (cam_pos_w - robot_pos_w), robot_rot_w.T @ cam_rot_w

    def _project(
        self,
        xyz: np.ndarray,
        cam: dict,
        cam_pos: np.ndarray,
        cam_rot: np.ndarray,
    ) -> np.ndarray:
        """Project ``(..., 3)`` robot-frame XYZ to ``(..., 2)`` pixel coords.

        Out-of-view and behind-camera points are returned as ``np.nan`` so that
        ``ax.plot()`` automatically breaks the line at discontinuities.
        """
        delta = xyz - cam_pos                                           # (..., 3)
        p_cam = (cam_rot.T @ delta[..., None]).squeeze(-1)             # (..., 3)

        K = cam["K"]
        H, W = self.image_h, self.image_w

        if cam["ros_convention"]:
            # Isaac / ROS: +Z forward, Y down, X right
            z_eff = p_cam[..., 2]
            v_sign = 1.0
        else:
            # MuJoCo: -Z forward, Y up, X right
            z_eff = -p_cam[..., 2]
            v_sign = -1.0

        valid = z_eff > 1e-4
        denom = np.where(valid, z_eff, 1.0)

        u = np.where(valid, K[0, 0] * p_cam[..., 0] / denom + K[0, 2], np.nan)
        v = np.where(valid, v_sign * K[1, 1] * p_cam[..., 1] / denom + K[1, 2], np.nan)

        if cam["flip_v"]:
            v = np.where(valid, (H - 1.0) - v, np.nan)
        if cam["flip_h"]:
            u = np.where(valid, (W - 1.0) - u, np.nan)

        # Don't nan out-of-bounds points — let ax.set_xlim/ylim clip the line.
        # Only nan is used for behind-camera points (already handled above).
        return np.stack([u, v], axis=-1)

    # ------------------------------------------------------------------
    # Vis update
    # ------------------------------------------------------------------

    def _anchor_to_current_ee(
        self, hand_xyz: np.ndarray, obs_dict: dict, env_i: int
    ) -> np.ndarray:
        """Shift a decoded trajectory so its first step aligns with the actual EE.

        The decoder outputs absolute EE positions in robot frame, but early in
        training it outputs near-zero garbage.  Anchoring preserves the relative
        planned deltas while pinning the start to the real current EE position
        (read from ``obs_dict["proprioception"]``).  At convergence
        hand_xyz[0] ≈ current_ee so the correction vanishes.
        """
        propri = obs_dict.get("proprioception")
        if propri is None:
            return hand_xyz
        try:
            ee = propri[env_i] if propri.ndim > 1 else propri
            current_ee = ee[self.hand_xyz_slice]   # [3]
        except (IndexError, TypeError):
            return hand_xyz
        return current_ee[None, :] + (hand_xyz - hand_xyz[0:1, :])

    def update(
        self,
        obs_dict: dict,
        rollouts: np.ndarray,
        rollout_rewards: Optional[np.ndarray] = None,
        rollout_costs: Optional[np.ndarray] = None,
        frame_timestep: Optional[int] = None,
    ) -> None:
        """Render the optimal rollout on each camera image.

        Args:
            obs_dict:
                Dict of numpy arrays containing camera images.  Camera keys
                must match the ``obs_key`` of registered cameras.  Images may
                be in ``[0, 1]`` or ``[-0.5, 0.5]`` (CamOnlyObsNormalizer);
                the normaliser shift is undone automatically.
            rollouts:
                ``[vis_n_envs, n_rollouts, horizon, state_dim]`` decoded
                proprioception.  Hand XYZ occupies ``self.hand_xyz_slice``.
            rollout_rewards:
                ``[vis_n_envs, n_rollouts, horizon]`` per-step rewards.
                The rollout with the highest cumulative reward is drawn.
            frame_timestep:
                Optional step index shown in subplot titles.
        """
        if not self._cameras:
            return

        for env_i in range(self.vis_n_envs):
            for cam_j, cam in enumerate(self._cameras):
                ax = self.axes[env_i][cam_j]
                ax.cla()

                # ---- camera background ----
                cam_imgs = obs_dict.get(cam["obs_key"])
                if cam_imgs is not None:
                    img = cam_imgs[env_i] if cam_imgs.ndim == 4 else cam_imgs
                    if img.min() < -0.05:  # undo CamOnlyObsNormalizer [-0.5, 0.5] shift
                        img = img + 0.5
                    img = np.clip(img, 0.0, 1.0)
                    ax.imshow(img, origin="upper", extent=[0, self.image_w, self.image_h, 0])

                ax.axis("off")
                title = cam.get("label", cam["obs_key"])
                if frame_timestep is not None:
                    title += f" t={frame_timestep}"
                ax.set_title(title, fontsize=8)

                if env_i >= rollouts.shape[0]:
                    ax.set_xlim(0, self.image_w)
                    ax.set_ylim(self.image_h, 0)
                    continue

                cam_pos, cam_rot = self._get_cam_extrinsics(cam)

                if getattr(self.cfg, "sampled_particles", False):
                    # ---- sampled rollouts: depth-sized dots, reward-coloured ----
                    n_vis   = rollouts.shape[1]
                    H       = rollouts.shape[2]
                    opt_ax  = cam_rot[:, 2]  # optical axis in robot frame
                    # Opacity fades along the horizon, consistent with optimal trajectory
                    h_alpha = np.linspace(0.9, 0.15, H)

                    # Pre-compute global reward statistics for consistent colouring
                    has_rew = rollout_rewards is not None
                    if has_rew:
                        r_all   = rollout_rewards[env_i]            # [n_vis, H]
                        r_min, r_max = np.nanmin(r_all), np.nanmax(r_all)
                        r_rng   = max(r_max - r_min, 1e-8)
                        cum_rew = r_all.sum(axis=-1)                # [n_vis]
                        c_min, c_max = cum_rew.min(), cum_rew.max()
                        c_rng   = max(c_max - c_min, 1e-8)
                        plasma_cm  = plt.cm.plasma
                        cool_cm    = plt.cm.cool

                    # Draw worst-to-best so highest-reward dots land on top
                    for ri in range(n_vis - 1, -1, -1):
                        hand_xyz = rollouts[env_i, ri, :, self.hand_xyz_slice]  # [H,3]
                        hand_xyz = self._anchor_to_current_ee(hand_xyz, obs_dict, env_i)
                        pixels   = self._project(hand_xyz, cam, cam_pos, cam_rot)  # [H,2]
                        valid    = ~np.isnan(pixels[:, 0])  # [H]

                        # Depth per point (one dot-product per rollout axis, vectorised)
                        z_eff = (hand_xyz - cam_pos) @ opt_ax  # [H]
                        if not cam["ros_convention"]:
                            z_eff = -z_eff
                        # Perspective-correct sizing: a physical sphere of radius r at
                        # depth z subtends fx*r/z pixels → scale scatter area by fx.
                        fx = cam["K"][0, 0]
                        r  = getattr(self.cfg, "dot_physical_radius_m", 0.025)
                        dot_sz = np.clip(fx * r / np.maximum(z_eff, 0.05) * 3.0, 1.0, 360.0)

                        # ── Non-terminal steps 0 .. H-2 ──
                        nt_mask = valid[:H - 1]
                        if nt_mask.any():
                            nt_pix   = pixels[:H - 1][nt_mask]
                            nt_sz    = dot_sz[:H - 1][nt_mask]
                            nt_alpha = h_alpha[:H - 1][nt_mask]
                            if has_rew:
                                nt_norm = np.clip(
                                    (r_all[ri, :H - 1][nt_mask] - r_min) / r_rng,
                                    0.0, 1.0,
                                )
                                nt_rgba = plasma_cm(nt_norm)  # [V, 4]
                            else:
                                nt_rgba = np.tile([50/255, 205/255, 50/255, 0.0],
                                                  (nt_mask.sum(), 1))
                            nt_rgba[:, 3] = nt_alpha
                            ax.scatter(nt_pix[:, 0], nt_pix[:, 1], s=nt_sz,
                                       c=nt_rgba, edgecolors="none", zorder=10)

                        # ── Terminal step H-1: 2.5× larger, white ring, cool_r ──
                        if valid[H - 1]:
                            t_sz = float(dot_sz[H - 1]) * 2.5
                            t_a  = float(h_alpha[H - 1])
                            if has_rew:
                                # Colour by cumulative reward → encodes full traj quality
                                t_norm = float(np.clip(
                                    (cum_rew[ri] - c_min) / c_rng, 0.0, 1.0))
                                t_rgba = list(cool_cm(t_norm))
                                t_rgba[3] = t_a
                            else:
                                t_rgba = [50/255, 205/255, 50/255, t_a]
                            ax.scatter(
                                [pixels[H - 1, 0]], [pixels[H - 1, 1]],
                                s=[t_sz], c=[t_rgba],
                                edgecolors="white", linewidths=0.5, zorder=11,
                            )
                else:
                    # ---- find and draw the single optimal rollout ----
                    if rollout_rewards is not None:
                        best_idx = int(np.nanargmax(rollout_rewards[env_i].sum(-1)))
                    elif rollout_costs is not None:
                        best_idx = int(np.nanargmin(rollout_costs[env_i].sum(-1)))
                    else:
                        best_idx = 0  # first rollout (planner pre-sorts by reward)

                    hand_xyz = rollouts[env_i, best_idx, :, self.hand_xyz_slice]  # [H, 3]
                    hand_xyz = self._anchor_to_current_ee(hand_xyz, obs_dict, env_i)
                    pixels   = self._project(hand_xyz, cam, cam_pos, cam_rot)      # [H, 2]

                    H_len = pixels.shape[0]
                    if H_len > 1:
                        segs = np.stack([pixels[:-1], pixels[1:]], axis=1)
                        alphas = np.linspace(1.0, 0.1, len(segs))
                        # limegreen = #32CD32
                        colors = np.c_[
                            np.full(len(segs), 50 / 255),
                            np.full(len(segs), 205 / 255),
                            np.full(len(segs), 50 / 255),
                            alphas,
                        ]
                        ax.add_collection(LineCollection(
                            segs, colors=colors, linewidth=3.5, zorder=10,
                        ))

                ax.set_xlim(0, self.image_w)
                ax.set_ylim(self.image_h, 0)  # inverted (v=0 at top)

        self.fig.tight_layout(pad=0.5)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def save_frame(self, output_dir: str, step: Optional[int] = None) -> None:
        fname = "image_rollouts_vis.png" if step is None else f"step-{step}_image_rollouts_vis.png"
        path = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fig.savefig(path, bbox_inches="tight")

    def close(self) -> None:
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Video wrapper
# ---------------------------------------------------------------------------

class ImageRolloutsVideo:
    """Accumulates per-step figures into an MP4.

    Drop-in replacement for the standard ``RolloutsVideo`` so that image-space
    vis is logged to wandb the same way.
    """

    def __init__(self, vis: ImageSpaceRolloutsVisualization) -> None:
        self.vis = vis
        self.img_frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.vis.reset()
        self.img_frames = []

    def update_video(self) -> None:
        """Capture the current matplotlib figure into the frame buffer."""
        buf = BytesIO()
        self.vis.fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        self.img_frames.append(imageio.imread(buf))
        buf.close()

    def save_video(
        self, output_dir: str, episode_num: Optional[int] = None, frame_rate: int = 10
    ) -> str:
        fname = "image_rollouts_video.mp4"
        if episode_num is not None:
            fname = f"episode-{episode_num}_{fname}"
        path = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.img_frames:
            imageio.mimsave(path, self.img_frames, fps=frame_rate)
            print(f"[ImageRolloutVis] Video saved to {path}")
        return path
