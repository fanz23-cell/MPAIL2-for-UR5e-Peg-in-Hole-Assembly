# 仿真环境补丁

实验使用的 Isaac Lab / UWLab 本地 checkout 有两处任务相关修改，分别以可读 patch 随仓库保存。它们不是 MPAIL2 算法改动。

## Isaac Lab v2.3.2

isaaclab-terminal-observation.patch 只修改 manager_based_rl_env.py：在 reset 前暂存 done 环境的终止观测，并保存、恢复观测历史缓存。

应用到 Isaac Lab v2.3.2 checkout：

    git -C /path/to/IsaacLab checkout v2.3.2
    git -C /path/to/IsaacLab apply --check /path/to/this/repo/patches/isaaclab-terminal-observation.patch
    git -C /path/to/IsaacLab apply /path/to/this/repo/patches/isaaclab-terminal-observation.patch

本项目 runner 当前没有消费 extras["terminal_observation"]，所以该补丁本身不保证模型训练使用终止观测。复现实验前要先确认目标 runner 的 done-transition 处理。

## UWLab

uwlab-fixed-position-reset.patch 基于 UWLab 提交 36d98afe1166f546083fc6e3c5d5bee04b486d84。它新增物体位置固定、机械臂起点仍随机的任务配置。

应用到对应 checkout：

    git -C /path/to/UWLab checkout 36d98afe1166f546083fc6e3c5d5bee04b486d84
    git -C /path/to/UWLab apply --check /path/to/this/repo/patches/uwlab-fixed-position-reset.patch
    git -C /path/to/UWLab apply /path/to/this/repo/patches/uwlab-fixed-position-reset.patch

bootstrap 脚本会检出基础版本，但不会自动应用以上补丁。
