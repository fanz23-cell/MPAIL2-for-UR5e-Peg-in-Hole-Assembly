# 完整任务 vs 后半段任务 切换指南

## 两种任务的区别

| 项目 | 后半段任务（Half Task） | 完整任务（Full Task） |
|------|------------------------|----------------------|
| 任务范围 | 物体已在手中 → 插入孔 | 物体在桌上 → 抓取 → 插入孔 |
| Reset 类型 | `ObjectAnywhereEEGrasped` | `ObjectAnywhereEEAnywhere` |
| Expert Demo | `official_route_a/route_a_mpail.pt` | `official_full_x2/full_object_anywhere_mpail.pt` |
| Demo 大小 | 905 transitions，26 episodes | 3002 transitions，55 episodes |
| 问题 | agent 频繁掉方块，demo 只覆盖抓住状态，obs 分布不匹配 → 判别器饱和 | 覆盖完整任务，obs 分布对齐 |

---

## 启动训练

### 完整任务（推荐）

```bash
bash scripts/run_full_task_obs22_gp500_lr1e6_500iter.sh
```

关键参数：
- `omnireset_reset_types=[ObjectAnywhereEEAnywhere]`
- `DEMO_PATH=mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt`
- `omnireset_disable_abnormal_termination=False`
- `+learner.use_terminations=True`
- `omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5]`
- `gp_coeff=500`, `lr=1e-6`, `r1_coeff=0.005`

### 后半段任务

```bash
bash scripts/run_s8_s6f_R1reg_300iter.sh
```

关键参数：
- `omnireset_reset_types=[ObjectAnywhereEEGrasped]`
- `DEMO_PATH=mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt`
- `omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5]`
- `gp_coeff=500`, `lr=1e-6`, `r1_coeff=0.005`

---

## 查看 IsaacLab 效果（watch 命令）

**注意：必须在有 GUI 的终端里跑，不能通过 Claude 远程执行。**

### 完整任务

```bash
run_dir="logs/<你的run_id>"
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/watch_mpail_omnireset_sequence.py \
  --checkpoint-dir $run_dir/models \
  --checkpoints 50 100 200 \
  --demo-path mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
  --reset-types ObjectAnywhereEEAnywhere \
  --compact-obs-dim 22 \
  --raw-action-limit 8 8 8 5 5 1.25 0.5
```

### 后半段任务

```bash
run_dir="logs/<你的run_id>"
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/watch_mpail_omnireset_sequence.py \
  --checkpoint-dir $run_dir/models \
  --checkpoints 50 100 200 \
  --demo-path mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt \
  --reset-types ObjectAnywhereEEGrasped \
  --compact-obs-dim 22 \
  --raw-action-limit 8 8 8 5 5 1.25 0.5
```

`--checkpoints` 后面跟想看的 iter 编号，可以写多个，会依次播放。

---

## 监控训练 log

```bash
# 找最新的 run
run_dir=$(ls -td logs/obs22_FullTask_* | head -1)   # 完整任务
run_dir=$(ls -td logs/obs22_S8_* | head -1)          # 后半段任务

# 实时看 log
tail -f $run_dir/train.log | grep -E "iter|disc_acc|disc_demo|disc_gen|succ"
```

健康的判别器指标（100iter 内）：
- `disc_demo` 在 1~3 之间缓慢上升
- `disc_gen` 在 -1~-3 之间缓慢下降
- gap（disc_demo - disc_gen）< 8 为健康，> 12 说明开始饱和

---

## 注意事项

1. **Tyler 的 terminated/truncated 分离**：完整任务必须保留 `omnireset_disable_abnormal_termination=False` 和 `+learner.use_terminations=True`，这两个参数启用了代码里已实现的 terminated/truncated 语义分离（terminal_observation、abnormal transition 不过滤）。后半段任务脚本里没有这两个参数，不要随意加。

2. **watch 脚本依赖 `omnireset_clean_eval.py`**：这个文件曾被误删，已从 git 恢复（`mpail2/envs/isaac/omnireset_clean_eval.py`）。如果再次出现 `ModuleNotFoundError: No module named 'mpail2.envs.isaac.omnireset_clean_eval'`，用以下命令恢复：
   ```bash
   git show 8eb3ed4:mpail2/envs/isaac/omnireset_clean_eval.py > mpail2/envs/isaac/omnireset_clean_eval.py
   ```

3. **watch 命令参数格式**：用 `--checkpoint-dir` + `--checkpoints <iter编号>`，不是 `--checkpoint <文件路径>`。checkpoint 文件在 `models/` 目录下，不是 `checkpoints/`。

4. **R1 regularization**：`r1_coeff=0.005` 对两种任务都启用，可以通过 `+learner.reward_learner_cfg.r1_coeff=0.0` 关闭。
