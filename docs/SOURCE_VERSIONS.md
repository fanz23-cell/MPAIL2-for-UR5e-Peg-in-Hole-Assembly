# 源码版本与实验对应关系

本仓库包含两套不同阶段的 MPAIL2 源码。后期运行脚本切换过源码目录，因此复现实验时要使用匹配版本。

| 内容 | 仓库内路径 | 来源与基线 | 用于哪些记录 |
|---|---|---|---|
| 主任务适配 | mpail2/ | 本仓库 main 的整理基线 90a604f，代码位于根目录包 | Route A、较早的观测和 MPPI 对照，以及 S189b / S190 |
| 后期源码快照 | snapshots/mpail-research/ | UWRobotLearning/mpail-research 基线 e6f45392feedfc20d781566ac59d36fd33204323，叠加本地 OmniReset 实验工作树中的修改 | S191 的 MPAIL3 训练脚本、S192b / S192c、S193a / S193b |
| Isaac Lab | 本地环境 checkout | v2.3.2，基线提交 37ddf626871758333d6ed89cf64ad702aef127d0 | OmniReset 仿真运行 |
| UWLab 任务 | 本地环境 checkout | 基线提交 36d98afe1166f546083fc6e3c5d5bee04b486d84 | OmniReset UR5e / Robotiq 2F-85 任务与固定位置 reset |

mpail-research/ 原来是本地独立 Git 仓库，且被根目录 .gitignore 排除。为了避免提交 gitlink 或个人 .git 元数据，本仓库把需要的 Python、配置和文档导出成普通目录 snapshots/mpail-research/。该快照包含基线之上的本地改动；基线 hash 用来说明来源，不能单独还原这些本地改动。

“S192 / S193”是本项目实验脚本的运行编号和本地分支命名，不是 MPAIL3 论文方法或正式算法版本。S192 / S193 脚本通过切换到后期源码目录启动模型，不能用主目录的 mpail2/ 替代。

## 环境版本注意事项

根目录的 bootstrap 脚本选择 Python 3.11；后期源码快照的 pyproject.toml 仍声明 Python >=3.10,<3.11。历史运行脚本从源码目录直接启动，并不等价于在当前环境重新安装该快照。需要复现后期运行时，应先核对 Isaac Lab、PyTorch、Python 和 OmniReset 扩展版本；本仓库没有宣称已在干净机器上验证兼容性。

S192c 依赖本地专家数据和 teacher policy。数据与大模型权重没有提交到 Git，实验运行前需准备相同格式的数据文件。公开版训练脚本支持用环境变量指定数据路径，不依赖实验机上的绝对路径。

## Isaac Lab 与 runner 的 terminal observation

patches/isaaclab-terminal-observation.patch 保存实验机在 Isaac Lab v2.3.2 上的改动：环境 reset 前计算并保存 done 环境的 terminal observation，同时暂存并恢复观测历史，避免临时观测污染 history buffer。补丁把 terminal observation 写入 extras。

需要注意：当前仓库的训练 runner 没有读取这个 extras["terminal_observation"] 字段。因此，补丁能提供该信息，但不能据此声称当前 MPAIL runner 已经用 terminal observation 训练跨 reset 的动力学；具体终止转移语义仍需按运行脚本和 runner 源码核对。

## UWLab fixed-position reset

patches/uwlab-fixed-position-reset.patch 在 UWLab 任务配置中增加 ObjectFixedPositionEEAnywhere reset：插销位置固定在原始范围中点，机械臂起始姿态仍在原范围内随机采样。它用于固定物体位置的任务/数据实验，不代表完整任务的随机化泛化结果。
