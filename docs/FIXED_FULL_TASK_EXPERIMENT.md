# 固定起始位置完整任务训练实验

## 概述

这是一个**独立的实验配置**，用于测试固定起始位置的完整任务训练。所有文件都是新创建的，不会影响现有的训练配置。

## 实验设置

- **任务类型**: 完整任务（物体在桌上 → 抓取 → 插入）
- **起始位置**: 固定（所有 expert demo 和训练 reset 都使用相同的物体/机器人初始位置）
- **Reset 类型**: `ObjectAnywhereEEAnywhere`（完整任务，物体在桌上，非抓取状态）
- **训练参数**: S1 基线
  - `gp_coeff=500`
  - `replay_ratio=0.0391` (100/(8*160) for 8 envs)
  - `reward_lr=1e-4`
  - `num_rollouts=512`
  - `num_elites=64`
  - `horizon=7`
- **训练长度**: 100 iterations
- **环境数**: 8 envs

## 新创建的文件

### 1. Demo 收集脚本
- `scripts/collect_fixed_full_task_rollouts.sh` - 收集固定起始位置的 expert rollouts
- `scripts/build_fixed_full_task_demos.sh` - 从 rollouts 构建 MPAIL demo

### 2. 训练脚本
- `scripts/run_fixed_full_task_S1_100iter.sh` - 启动固定起始位置完整任务训练

### 3. Demo 存储目录
- `mpail2/train/isaac_franka/demos/fixed_full_task/` - 新的独立 demo 目录
  - `fixed_full_rollouts_raw.pt` - 原始 rollouts
  - `fixed_full_bc_raw.pt` - BC 格式 demo
  - `fixed_full_mpail.pt` - MPAIL 训练用 demo

## 使用流程

### 步骤 1: 收集 Expert Demo

首先需要设置完整任务的 teacher policy 路径：

```bash
export POLICY_PATH=/path/to/your/full_task_teacher_policy.pt
bash scripts/collect_fixed_full_task_rollouts.sh
```

这会收集 50 个成功的 episode，所有 episode 使用相同的固定起始位置。

### 步骤 2: 构建 MPAIL Demo

```bash
bash scripts/build_fixed_full_task_demos.sh
```

这会从 rollouts 生成 MPAIL 训练用的 demo 文件。

### 步骤 3: 启动训练

```bash
bash scripts/run_fixed_full_task_S1_100iter.sh
```

默认使用 8 envs，100 iterations。也可以自定义：

```bash
bash scripts/run_fixed_full_task_S1_100iter.sh 16 100  # 16 envs, 100 iter
```

## 与现有训练的隔离

### 完全独立的文件
- ✅ 新的 demo 目录: `fixed_full_task/`（不影响 `official_route_a/` 或 `official_full_x2/`）
- ✅ 新的训练脚本: `run_fixed_full_task_S1_100iter.sh`（不修改现有脚本）
- ✅ 新的 log 目录: `logs/obs22_fixed_full_S1_*`（独立的 run_id）

### 不会修改的现有文件
- ❌ 不修改 `scripts/run_omnireset_obs22_*.sh`（现有训练脚本）
- ❌ 不修改 `mpail2/train/isaac_franka/demos/official_*/`（现有 demo）
- ❌ 不修改任何 Python 源代码（只使用现有的训练框架）

## 关键配置差异

### 与现有半任务训练的区别

| 配置项 | 现有半任务 (Route A) | 新实验 (固定完整任务) |
|--------|---------------------|---------------------|
| Reset 类型 | `ObjectAnywhereEEGrasped` | `ObjectAnywhereEEAnywhere` |
| 任务范围 | 插入段（物体已抓取） | 完整任务（抓取+插入） |
| 起始位置 | 随机 | 固定 |
| Demo 路径 | `official_route_a/` | `fixed_full_task/` |
| Obs dim | 22 | 22 |
| Gripper | Free | Free |

## 注意事项

1. **Teacher Policy**: 需要一个完整任务的 teacher policy（支持从桌上抓取到插入的完整流程）
2. **固定起始位置**: `ObjectAnywhereEEAnywhere` 配合 `seed=42` 确保每次 reset 位置一致
3. **Obs 维度**: 使用 obs22，可能需要验证是否包含足够的抓取信息（EE-to-object 距离等）
4. **训练长度**: 先跑 100 iter 验证，成功后可以延长到 500-1000 iter

## 下一步

训练完成后，可以：
1. 用 `scripts/watch_mpail_omnireset.py` 观察策略表现
2. 对比固定起始位置 vs 随机起始位置的学习效率
3. 如果效果好，可以逐步增加起始位置的随机性
