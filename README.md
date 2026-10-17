# MPAIL2 在 OmniReset UR5e 插孔装配任务中的适配与实验

本项目把公开的 MPAIL2 算法接入 OmniReset 的 UR5e 插销装配仿真任务，用一系列工程适配和对照实验，观察基于模型的模仿学习与在线规划在接触丰富、重置条件多变的任务中如何工作、在哪里失效。

**项目定位：**这是对既有 MPAIL2 方法的任务适配和实验评估，不是提出新的 MPAIL 算法，也没有被纳入 MPAIL2 原论文。仓库记录了环境接入、示范数据处理、策略预训练、MPPI 控制对照和失败分析；目前结果尚未证明完整装配任务获得稳定提升。

## 最新实验视频

<a href="docs/media/omnireset-s191-checkpoints-0-5-10.mp4">S191 检查点 0、5、10 的短片（约 80 秒）</a> ·
<a href="docs/media/omnireset-s191-live-training.mp4">S191 仿真视口录屏（约 10 分 26 秒，已裁掉桌面画面）</a>

短片和录屏记录的是训练过程，供查看机器人行为变化；它们本身不代表任务成功率或稳定性结论。配置和统计口径见[实验记录](docs/EXPERIMENTS.md)。

## 任务是什么

环境是 Isaac Sim / Isaac Lab 中的 UR5e 与 Robotiq 2F-85 插孔任务。实验分成两种起始条件：

- **Route A（已抓取插入）：**物体已经在夹爪中，策略主要处理对准和插入。
- **完整任务：**物体和机械臂从桌面初始状态开始，依次完成接近、抓取和插入。

实验曾使用 13、19、22 维紧凑状态，以及后期完整任务路线的 215 维观测。不同观测、重置和动作配置属于不同实验条件，不能把它们的结果直接合并。这里的结论只涉及仿真；没有真实机械臂部署或跨场景泛化评估。

## 算法在这个项目里如何工作

MPAIL2 的核心路线不要求人工编写插孔任务奖励。示范和机器人在线交互用于学习潜在状态表示、动力学、奖励和价值函数；环境自带的任务 reward 主要用于记录诊断指标。后期也试过带额外结果标签的诊断变体，应与核心学习路线分开看。

从观测到动作的流程是：

1. 编码器把环境观测映射到潜在状态。
2. 动力学模型根据候选动作预测未来潜在状态。
3. 策略提供动作序列的先验或候选；MPPI 再围绕这些提议采样和评估多条动作序列。
4. 规划器用学到的奖励和价值评估预测轨迹，按 MPPI 权重更新候选分布，选择当前要执行的动作。
5. 环境返回新观测，系统执行一小段动作后继续重规划。

因此，**策略会提供候选动作，但最终控制由模型预测规划和 MPPI 选择共同决定**。策略不是直接把动作原样送进环境。

示范在不同实验里承担不同作用：状态转移用于学习模仿奖励；较晚的 IQ-Learn 预训练和行为克隆（BC）路线还使用了专家动作标签。它们不是同一套“纯观察”设置。

## 我做了什么

我的工作主要是把现有算法变成一个能在目标仿真任务里运行、能做对照并能检查失败原因的实验流程：

- **接通任务和算法接口：**适配 OmniReset 的观测、7 维笛卡尔控制动作与夹爪符号、动作尺度、reset / termination 信号，并分别检查已抓取插入和完整抓取插入路线。
- **整理专家数据：**使用 OmniReset 导出的 teacher 示范，建立轨迹读取、状态—动作—后继状态转换和筛选流程；teacher 是数据来源，不是我训练出来的新专家。
- **搭建预训练和在线实验：**比较行为克隆、IQ-Learn 预训练、示范回放保护、基础策略、MPPI 规划和两者混合控制。
- **迭代排查训练行为：**检查 replay FIFO 与采样比例、观测归一化、动作缩放、终止语义、规划时域、奖励与价值目标、候选动作选择等因素，并保留配置、日志和视频以便回看。
- **按研究实验方式组织对照：**使用固定配置、控制方式对照和组件消融。当前证据仍是探索性实验，缺少多随机种子和统一独立测试集，不能据此声称统计显著提升。

## 实验如何迭代

| 阶段 | 问题与尝试 | 得到的认识 |
|---|---|---|
| 环境接入与 Route A | 检查 reset、夹爪方向、动作幅值和 13 / 19 / 22 维状态；先把已抓取插入流程跑通 | 环境接口和观测定义会显著影响策略；紧凑状态是否足够需要实验判断 |
| 基础策略与 MPPI 对照 | 比较纯基础策略、MPPI 规划以及不同混合比例；检查动作修正幅值 | 在已记录的短对照里没有看到稳定的 MPPI 增益，部分配置对最终动作影响很小 |
| 完整抓取插入 | 转向 215 维完整任务观测，整理完整任务示范，尝试 IQ-Learn + BC 预训练和受保护的示范回放 | 预训练和示范回放让实验流程更完整，但后续训练记录仍没有成功信号 |
| 后期模型与规划消融 | 在 32 个并行环境中比较奖励目标、规划设置、ensemble 和 weight normalization 等选项 | 这些实验帮助缩小排查范围；不能将某一个尝试描述成已证实的算法改进 |

## 结果怎么读

下面的 Route A 对照统计的是多个并行环境和控制步上的**成功状态占比**，不是“成功回合数 / 总回合数”。括号中的标准差是 20 个评估迭代之间的波动，不是多随机种子置信区间。

| 实验 | 控制方式 | 平均成功状态占比 | 迭代间标准差 |
|---|---|---:|---:|
| S164 | 基础策略 | 52.26% | 6.64 个百分点 |
| S163，运行 1 | 基础策略 + MPPI，blend 0.05 | 52.09% | 5.36 个百分点 |
| S163，运行 2 | 基础策略 + MPPI，blend 0.05 | 53.07% | 5.19 个百分点 |

这组短对照没有显示可靠的 MPPI 提升。S163 的实际动作修正平均幅度约为 0.0015，因此规划器对执行动作的影响也很有限。

在后期完整任务路线中，S189b、S190、S192b、S192c、S193a 和 S193b 的已保存训练记录都没有观察到 success 信号。S192c 计划运行 500 次迭代，日志只保存到第 209 行（迭代编号 0–208）。这些是训练期间的日志指标，不是独立测试集上的回合成功率；详细运行数和指标定义见[结果表与实验记录](docs/EXPERIMENTS.md)。

**当前结论：**我完成了 MPAIL2 与 OmniReset 插孔装配任务的适配和多轮实验，但现有证据不支持“在线学习或 MPPI 已稳定提升完整任务成功率”。把结果写成失败分析和方法适用性评估，比声称成功更准确。

## 代码与复现材料

- [根目录的 MPAIL2 任务适配](mpail2/envs/isaac/omnireset_wrapper.py)
- [Route A 示范数据构建入口](scripts/build_omnireset_route_a_demo.py)
- [完整任务示范数据处理入口](scripts/build_omnireset_success_demo.py)
- [S189b 完整任务训练配置](scripts/run_s189b_tyler_iqlearnPretrain_rewardTarget_fixedStdBC_demoReplay050_32env_100iter_wandb.sh)
- [S190 完整任务训练配置](scripts/run_s190_tyler_iqlearnPretrain_rewardTarget_fixedStdBC_demoReplay050_32env_300iter_wandb.sh)
- [S192c 后期消融配置](scripts/run_s192c_mpail3_s189b_equiv_tylerIQlearn_demoReplay_32env_500iter_wandb.sh)
- [根目录与后期实验源码版本说明](docs/SOURCE_VERSIONS.md)
- [第三方环境补丁说明](patches/README.md)

完整仿真需要 Isaac Sim / Isaac Lab、匹配的 UWLab 任务代码和本地专家示范文件。模型权重、示范数据、训练日志没有全部放进 Git。S192 / S193 使用单独的后期 MPAIL2 源码快照；历史实验还依赖 Isaac Lab / UWLab 补丁。按[源码与依赖说明](docs/SOURCE_VERSIONS.md)准备版本和数据后，再运行对应脚本。S191–S193 后期训练脚本支持通过 MPAIL_SOURCE_ROOT、OMNIRESET_TASK_DEMO_PATH、OMNIRESET_BC_DEMO_PATH 和 OMNIRESET_TEACHER_POLICY_PATH 指定源码与数据位置；W&B 默认离线记录。仓库没有声称已经完成新机器上的端到端复现。

## 上游来源

- [MPAIL2 原始源码](https://github.com/UWRobotLearning/mpail2)
- [MPAIL2 论文 v1](https://arxiv.org/abs/2602.24121v1)
- [MPAIL2 项目主页](https://uwrobotlearning.github.io/mpail2/)

原算法及论文成果归原作者。本项目是独立的仿真任务适配和实验记录，不代表原论文作者对这些 OmniReset 结果的认可。

    @article{han2026planning,
      title   = {Planning from Observation and Interaction},
      author  = {Han, Tyler and Shen, Siyang and Baijal, Rohan and Ravichandran, Harine and Nemekhbold, Bat and Huang, Kevin and Jung, Sanghun and Boots, Byron},
      journal = {arXiv preprint arXiv:2602.24121},
      year    = {2026}
    }
