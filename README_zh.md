<p align="center">
  <img src="assets/inference-autopilot-logo.svg" alt="Inference Autopilot" width="900">
</p>

<h1 align="center">Inference Autopilot</h1>

<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-Switch-7dd3fc?style=for-the-badge"></a>
  <a href="README_zh.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-当前-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <strong>针对你的硬件和真实工作负载，寻找更优的 SGLang 部署配置。</strong>
</p>

<p align="center">
  <a href="https://github.com/Jacki1223/inference-autopilot/releases/latest"><img alt="最新版本" src="https://img.shields.io/github/v/release/Jacki1223/inference-autopilot?color=4f46e5&label=release"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-2563eb">
  <a href="https://github.com/Jacki1223/inference-autopilot/actions/workflows/sglang-parameter-compat.yml"><img alt="SGLang 参数兼容性" src="https://github.com/Jacki1223/inference-autopilot/actions/workflows/sglang-parameter-compat.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-1e3a8a"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#结果与产物">结果</a> ·
  <a href="https://github.com/Jacki1223/inference-autopilot/releases/latest">下载</a>
</p>

Inference Autopilot（`inferopt`）是面向 [SGLang](https://github.com/sgl-project/sglang) 的单机（暂时）推理优化 CLI。你只需提供模型、GPU 服务器、代表性工作负载、可选的延迟 SLO 和实验预算，它便会验证部署可行性、测试相关候选配置、诊断性能瓶颈，并输出由实测数据支撑、可复现的启动命令。

工具给出的结论有明确边界：它表示在记录的模型、SGLang 版本、硬件、工作负载和实验预算内找到的最佳配置，而不是对全局最优的宣称。

## 你将获得什么

- 在昂贵 GPU 实验开始前完成部署可行性检查。
- 对 SGLang 基线和有潜力候选配置进行实测比较。
- 输出仅使用当前 SGLang 安装版本所支持参数的推荐启动命令。
- 生成 Markdown 报告，解释瓶颈、已测试改动、被拒绝候选、SLO 结果和最终决策。
- 保存可用于复现和审计的结构化实验产物。

如果没有候选配置取得可靠收益，保留基线同样是有效结果。

## 工作原理

1. **理解部署环境与全部参数**——检查 GPU 显存与拓扑、模型结构与权重大小、Checkpoint 精度、SGLang 能力、Cookbook 证据、工作负载形状、前缀复用、部署模式和 SLO；同时读取当前安装版本的 `ServerArgs` 与帮助信息，为每个可见参数建立 Parameter Capability Registry，记录类型和值域、作用机制、源码/Cookbook 证据、适用条件、依赖、冲突和风险。在消耗 GPU 时间前，先排除无法部署的并行方式和不兼容的功能组合。
2. **建立可信基线**——使用解析后的配置启动 SGLang，完成服务预热和实际接入容量探测，再按照并发数或运行时容量生成足够的稳态请求。记录吞吐、端到端延迟、TTFT、TPOT/ITL、错误率、显存余量和 SLO 结果。
3. **Profiling 与瓶颈诊断**——采集有边界、仅覆盖稳态推理区间的 Nsight Systems Trace，并结合 SGLang 日志、调度统计、缓存与队列遥测、CUDA Graph 覆盖率以及模型和工作负载证据。原始 profiler 现象会被归约为可以安全触发优化规则的标准瓶颈类别。
4. **按机制进行搜索**——根据当前 SGLang 参数协议、兼容的 Cookbook 配置、模型原生能力、硬件约束、工作负载特征和实测瓶颈构建 Candidate Registry。成熟参数使用版本化规则和动态值集；当前 SGLang 中尚未被规则覆盖、但语义明确且值域有界的安全参数，也经过同一套场景、依赖/冲突和风险门控进入实测，而不是为某个参数写特例。工具先覆盖不同优化机制，再只对正收益或不确定区域细化取值，并测试可能叠加的组合，而不是执行盲目的参数笛卡尔积。
5. **确认真实赢家**——候选配置首先与最强的直接父配置比较，再通过重复测量窗口、SLO 门槛、稳定性检查、置信区间和 Bayesian Sequential 证据与基线确认。没有额外价值的参数、噪声收益、性能回退和违反 SLO 的配置都会从最终命令中移除。
6. **报告并复用证据**——输出最简与完全可复现的启动命令、结构化 JSON、Benchmark 和 profiler 产物、候选拒绝原因、结论边界以及下一步优化方向。严格匹配的历史实验可以作为后续搜索的弱先验，但最终决策仍以当前实测结果为准。

规则系统不保存固定的“最佳配置”，而是根据当前部署证据选择相关优化机制并生成场景化候选值。Candidate Registry 会记录每个候选被选择、淘汰或测试的原因；只有通过兼容性检查、真实 Benchmark、用户声明的 SLO 和统计门槛的配置才会进入最终命令。

## 环境要求

- Python 3.9 或更高版本。
- 可由所选 Python 解释器运行的本地 SGLang checkout 或安装环境。
- 已经在服务器本地准备好的、与 SGLang 兼容的模型。
- 自动调优和 profiling 当前需要 NVIDIA GPU。
- `PATH` 中可用的 [Nsight Systems](https://developer.nvidia.com/nsight-systems)（`nsys`）。

AMD 硬件信息收集和规划已经支持，但 AMD 自动 profiling 和调优尚未实现。Nsight Compute 是可选项，仅在明确请求更深层 Kernel 分析时需要。

## 安装

直接从 GitHub 安装，不改变现有 SGLang、CUDA、PyTorch 或模型运行环境：

```bash
python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  "git+https://github.com/Jacki1223/inference-autopilot.git"
```

如需替换旧版本，请增加 `--force-reinstall`。如果从源码 checkout 进行开发，可运行 `python3 -m pip install .`。

## 快速开始

通过交互方式创建任务：

```bash
inferopt init --output task.json
```

交互提示遵循以下规则：

- 方括号内是默认值，直接按 **Enter** 即可采用，不要输入方括号。例如提示中的 `[balanced]` 表示按回车选择 `balanced`。
- 没有默认值的提示必须填写，除非提示中明确说明允许留空。
- 对 `yes/no` 问题输入 `yes` 或 `no`；直接回车采用方括号中的默认选项。
- 列表应遵循提示中的格式。GPU 编号使用不带空格的逗号分隔形式，例如 `0,1,2`；并发点同时支持逗号或空格分隔。
- 所有路径均指运行 InferOpt 的 GPU 服务器本地路径。

启动 GPU 实验前，先检查环境并生成实验计划：

```bash
inferopt doctor --task task.json --output doctor.json
inferopt plan --task task.json --output plan.json
```

确认计划后，运行实验并生成报告：

```bash
inferopt run --task task.json --yes --output final.json
inferopt report --result final.json --output report.md
```

`doctor` 和 `plan` 是只读操作，不会启动模型服务。`run --yes` 只会启动当前实验自己创建的进程，并实时显示容量测试、profiling、候选实验和最终确认的进度。

## 配置实验

交互式 `init` 会询问模型路径、SGLang 路径、GPU 选择、工作负载或数据集、优化目标、SLO 和实验预算。

### 部署模式

- `online_latency` 优化延迟以及满足 SLO 的安全服务容量。
- `offline_throughput` 最大化吞吐，也可以增加延迟或错误率约束。

离线吞吐默认按“每 GPU 吞吐”比较不同 TP/PP/DP 布局，因此多卡候选必须覆盖额外 GPU 成本。只有明确追求单服务最高吞吐时，才使用 `inferopt init --resource-scope per_service`；在线模式默认按单服务比较。

### 工作负载

InferOpt 支持固定 token 形状的合成请求、生成的共享前缀流量、本地自定义 JSONL 对话数据以及 ShareGPT 格式数据。真实数据始终保留在本地。应尽量使用接近生产环境的流量，因为推荐结果的代表性取决于测量工作负载是否真实。

### SLO

延迟约束可以统一选择 `p99` 或 `avg`，并设置端到端延迟、首 token 延迟（TTFT）和每输出 token 延迟（TPOT/ITL）。同时支持错误率和吞吐约束。如果只关注优化目标，可不设置延迟 SLO。

### 实验预算

- `fast` 默认24个trial，最多测试8个统一稳态候选，并为约两轮champion augmentation保留预算。
- `balanced` 默认40个trial，最多测试14个统一稳态候选，通常可执行三至四轮champion augmentation。
- `max` 默认96个trial，最多测试28个统一稳态候选，并扩大数值深挖与多轮组合深度。

所有模式使用相同的正确性、SLO 和最终确认门槛。模式只影响搜索宽度，不降低推荐结果所需的证据标准。你还可以显式限制 trial 数量、GPU-hours、总运行时间和并发使用的 GPU 数量。

确认阶段只固定预留最少的完整 Bayesian A/B block；结果模糊时才使用前面剩余的预算追加确认。如果正向或方向性 refinement 无法用满该层预算，空余槽会继续测试得分最高的 deferred 候选。

离线无SLO模式会先测得当前token形状下的实际容量。所有baseline、参数候选、相邻值、组合和最终确认窗口统一使用10个饱和容量波次，不再存在更短的粗筛/复测层；每次服务启动只产生一次可用于部署判断的准确测量。所有正收益候选都会保存并参与同层组合。Nsight仍保持独立的3波次有界诊断采样。在线SLO模式继续遵守延迟统计样本门槛，p99默认同样使用并发数的10个波次。

组合阶段采用多轮、预算驱动的champion augmentation，不再固定top-N。每一轮都以当前最强实测配置为champion，测试它与所有尚未包含的兼容正收益原子peer；通过parent增益门槛的赢家成为下一轮champion。循环只在没有增量收益或触及最终确认预留时停止。冲突或被支配的候选会记录原因，确定性的能力/显存失败会剪枝更激进的同类配置，避免再次启动模型浪费预算。释放的预算可继续覆盖缓存+调度、缓存+准入等尚未测试的champion增量边。

SGLang 0.5.18+ 的 Weight Cache 被视为执行层启动加速能力，而不是吞吐调参候选。工具会检查持久 daemon、speculative 冲突、TP 拓扑、并行 TP1 replica 和 KV/Mamba 容量固定条件；条件不安全时自动禁用并在报告中说明。某个必测 backend 启动失败时，剩余 refinement 预算会补测同机制的下一个 sibling；Bayesian 返回 `continue` 时会使用所有还能组成完整 AB pair 的剩余 trial。

自动化使用时，可以从 [`assets/task.autopilot.example.json`](assets/task.autopilot.example.json) 开始，也可以先通过 `init` 生成任务，再执行验证：

```bash
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

## 结果与产物

输出目录包含：

- `final.json`——机器可读的决策、指标、所选配置和部署命令。
- `report.md`——面向用户的分析结论和推荐报告。
- 解析后的任务以及 SGLang 启动参数。
- Benchmark 输出、服务日志、profiler 证据和候选拒绝原因。

运行产物可能包含模型路径、工作负载信息和生成文本，应保持输出目录私有。默认 `.gitignore` 会忽略生成的实验产物。

## 安全性与适用范围

Inference Autopilot 仅用于获得授权的单机实验。它不会在运行时安装软件包，不会修改驱动、CUDA、SGLang 源码、模型权重或 Kernel，不会自动部署到生产环境，也不会终止不属于当前实验的进程。改变数值精度的候选必须显式启用，并在部署前通过单独的质量评估。

多机搜索、生产发布编排、自动算子修改和完整多模态优化尚未实现。

## Agent 辅助使用

CLI 可以完全独立运行，不需要 Agent 或 Codex。支持 Skills 的环境可以使用 [`SKILL.md`](SKILL.md) 编排输入收集、计划审查、运行监控和结果解释；真正的性能决策仍以 CLI 和实验产物为准。

## 文档

- [`SKILL.md`](SKILL.md)——端到端操作流程。
- [`references/input-schema.md`](references/input-schema.md)——任务字段和指标定义。
- [`references/execution-schema.md`](references/execution-schema.md)——执行与产物协议。
- [`references/safety-policy.md`](references/safety-policy.md)——安全边界。
- [`references/sglang-adapter.md`](references/sglang-adapter.md)——SGLang 集成细节。

## 许可证

本项目使用 [Apache License 2.0](LICENSE)。
