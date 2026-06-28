# 系统说明文档

## 1. 项目做了什么

本项目实现了一个面向赛题“一种面向多智能体协作的低开销通信、状态传递与共享记忆机制”的可运行原型系统。系统不是单纯的工作流编排，而是围绕多 Agent 协作中的三个基础机制展开：

- 结构化通信：用紧凑协议替代自然语言长文本交接。
- 非文本状态传递：embedding 写入物理共享内存 IPC，工具执行结果写入状态交换层，Agent 间只传 `state_id`。
- 共享记忆复用：任务过程中的经验、证据和结论沉淀为可检索、可链接、可复用的记忆单元。

系统支持纯文本模式、结构化协议模式和机制消融实验，可生成 JSON 和 Markdown 报告，用于证明不同机制对通信开销、耗时、状态传递和记忆复用的贡献。

## 2. 系统模块和作用

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 多 Agent 运行时 | `multi_agent/runner.py` | 初始化 9 个具体 Agent、共享记忆、状态交换和实验上下文 |
| 编排器 | `multi_agent/orchestrator.py` | 执行 `planner -> researcher -> executor -> summarizer` 协作链路，支持顺序编排和可选 LangGraph |
| Agent 实现 | `multi_agent/agents.py` | 定义规划、检索、执行、总结、验证、路由、安全审查、调试、记忆归档等 Agent，并根据模式生成文本或结构化消息 |
| 结构化协议 | `multi_agent/protocol.py` | 定义 `agent-msg/v1`、握手、Agent Card、能力发现、协议映射、错误码和紧凑 wire format |
| 共享记忆 | `multi_agent/memory.py` | 使用 SQLite 存储记忆，支持关键词、标签、embedding 混合检索和动态记忆链接 |
| 状态交换 | `multi_agent/state_exchange.py` | 支持 `shared_memory` 和 `sqlite` 两种后端，embedding 可通过物理共享内存 IPC 传递，消息中只传状态引用 |
| 指标采集 | `multi_agent/metrics.py` | 统计消息数、字符数、token 估算、耗时、记忆命中、状态传递次数和规模 |
| 工具执行 | `multi_agent/tools.py` | 提供 CodeAct Python 动作空间，LLM 先生成安全工具计划，再编译为受限 Python 脚本在子进程中运行 |
| 任务集 | `multi_agent/tasks.py` | 提供三组关联连续任务，用于验证跨任务复用 |
| A/B 实验 | `experiments/run_ab.py` | 对相同任务运行 text 与 structured 两种模式 |
| 消融实验 | `experiments/ablation.py` | 对比 text、仅协议、协议+状态、协议+记忆、协议+记忆网络五种变体 |
| 演示报告 | `experiments/demo.py` | 生成演示 JSON 和 Markdown 报告 |
| 可视化面板 | `experiments/dashboard.py` | 提供浏览器页面查看单任务对比 |

## 3. 结构化通信机制

结构化模式使用 `agent-msg/v1` 协议。消息在传输时使用紧凑 JSON，核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `mid` | 消息 ID |
| `tid` | 任务 ID |
| `pid` | 父消息 ID |
| `f` / `t` | 发送方和接收方 |
| `s` | 协议版本 |
| `p.mt` | 消息类型，如 handshake、protocol_map、request、response、error |
| `p.a` | 动作类型 |
| `p.in` | 紧凑输入 |
| `p.out` | 紧凑输出 |
| `p.refs` | 记忆 ID 或状态 ID 引用 |
| `p.state` | 非文本状态元数据 |

结构化模式启动时，系统会先进行协议映射和 Agent 握手。每个 Agent 会暴露 Agent Card，包括能力、支持模式和可产出的 artifact 类型。协议协商开销和应用消息开销分开统计，避免把一次性启动成本混入任务通信成本。

## 4. 非文本状态传递机制

系统通过 `StateStore` 实现状态交换。默认配置为：

```bash
STATE_BACKEND=shared_memory
```

状态交换流程：

1. Executor 生成 embedding 和 CodeAct 工具执行结果。
2. embedding 向量写入 Python `multiprocessing.shared_memory` 创建的物理共享内存段。
3. `data/state.sqlite` 记录 `state_id`、`shm_name`、`dtype`、`shape`、大小和预览信息。
4. 后续 Agent 或评测模块可通过 `state_id` 读取状态。
5. 指标模块统计非文本状态传递次数、状态规模和状态引用 ID。

接收方可按如下方式零拷贝读取向量：

```python
from multiprocessing import shared_memory
import numpy as np

shm = shared_memory.SharedMemory(name=shm_name)
vector = np.ndarray(shape, dtype="float64", buffer=shm.buf)
```

这种设计避免把向量、中间对象全部转成自然语言长文本传递，也避免每次通过磁盘反序列化读取向量，符合赛题对“IPC、共享内存、减少文本编解码”和“中间表示直接交换”的要求。工具结果是结构化 JSON 对象，默认仍写入 SQLite 状态层，保证可追溯和可复现实验。

## 5. CodeAct 沙箱机制

系统默认使用：

```bash
CODEACT_SANDBOX=subprocess
```

CodeAct 代码会在独立 Python 子进程中执行，父进程通过 JSON 传入任务上下文和只读工具快照。正式路径中，Executor 先让 LLM 选择安全工具计划，例如是否读取 Markdown、检索记忆、计算指标、生成表格；系统再将计划编译为受限 Python 脚本。子进程侧只暴露受控工具函数：

- `read_file`
- `search_files`
- `load_json`
- `load_csv`
- `make_markdown_table`
- `compute_numeric_metrics`
- `summarize_records`
- `search_memory`

沙箱限制包括：

| 限制 | 作用 |
| --- | --- |
| `CODEACT_CPU_SECONDS` | 限制 CPU 时间 |
| `CODEACT_MEMORY_MB` | 限制进程地址空间 |
| `CODEACT_TIMEOUT_SECONDS` | 限制墙钟时间 |
| 安全 import 白名单 | 只允许 `math/json/statistics/re/collections/itertools` 等基础模块 |
| 工作区路径校验 | 工具只能读取工作区内部文件 |

相比原先纯 AST 白名单模式，子进程沙箱允许 `for`、`while` 和安全 import，能执行更复杂的 CodeAct 任务；同时通过进程隔离和资源限制降低风险。原来的保守模式仍可通过 `CODEACT_SANDBOX=inprocess` 启用。

## 6. 共享记忆机制

每条记忆包含：

| 字段 | 含义 |
| --- | --- |
| `memory_id` | 记忆 ID |
| `source_agent` | 来源 Agent |
| `created_at` | 创建时间 |
| `task_topic` | 任务主题 |
| `summary` | 摘要 |
| `tags` | 标签 |
| `embedding` | 语义向量 |
| `keywords` | 自动抽取关键词 |
| `links` | 相关历史记忆 ID |
| `access_count` | 被检索复用次数 |

检索使用混合打分：

```text
score = cosine(query_embedding, memory_embedding)
      + tag_overlap
      + keyword_overlap
      + text_overlap
      + link_score
```

记忆网络开关由 `MEMORY_GRAPH_ENABLED` 控制。关闭时只做基础共享记忆，开启时会生成关键词、链接和访问计数，用于验证动态记忆组织的效果。

正式评测默认启用两项在线复用优化，并保留 Agent 推理链路：

| 开关 | 作用 |
| --- | --- |
| `MEMORY_ARCHIVIST_ENABLED=0` | 将记忆归档 Agent 移出在线关键路径，避免把维护性整理成本计入每个任务时延 |
| `EMBEDDING_CACHE_ENABLED=1` | 对重复 query/summary embedding 做进程内缓存，减少真实 embedding 服务调用 |

该设计参考了四类已有思路：MemGPT 的分层/虚拟上下文管理把外部存储视作可调度记忆层；Generative Agents 的 memory stream 使用检索到的记忆参与后续规划和反应；A-MEM/Zettelkasten 风格记忆强调关键词、标签和链接组成的动态知识网络；HippoRAG 使用图式关联降低多跳检索成本。系统实现上对应为“写入结构化记忆、命中后以 memory refs 和压缩摘要进入真实 Agent 上下文、图链接辅助召回、离线整理维护”。

## 7. 实验设计

系统提供两类实验。

### 7.1 A/B 实验

`experiments.run_ab` 会对同一批任务分别运行：

- `text`：自然语言长文本交接，默认不启用非文本状态交换，作为纯文本协作基线。
- `structured`：结构化协议 + 状态引用 + 共享记忆。

输出：

```text
reports/results.json
```

### 7.2 消融实验

`experiments.ablation` 包含 5 个变体：

| 变体 | 目的 |
| --- | --- |
| `text_baseline` | 传统自然语言交接基线 |
| `structured_protocol` | 单独验证结构化协议 |
| `structured_state` | 验证非文本状态引用 |
| `structured_memory` | 验证共享记忆复用 |
| `structured_memory_graph` | 验证动态记忆链接 |

输出：

```text
reports/ablation_results.json
reports/ablation_report.txt
data/ablation/
```

## 8. 已覆盖的赛题要求

| 赛题要求 | 当前覆盖情况 |
| --- | --- |
| 不少于 3 个 Agent | 已实现 9 个具体 Agent，核心链路至少运行 planner、researcher、executor、summarizer，并可启用 router/verifier/security_reviewer/debugger/archivist |
| 覆盖规划、检索、执行、总结 | 已覆盖，并额外覆盖路由、验证、安全审查、调试和记忆归档 |
| 结构化通信协议 | 已实现 `agent-msg/v1` |
| 握手、能力发现、协议映射 | 已实现 handshake、Agent Card、protocol map |
| 纯文本与结构化模式对比 | 已实现 A/B 实验 |
| 非文本中间状态传递 | 已实现 StateStore 和 state refs |
| 结合 IPC、共享内存提升实现质量 | 已实现 `multiprocessing.shared_memory` 状态后端 |
| CodeAct 执行机制 | 已实现受限 Python 子进程沙箱 |
| 共享记忆存储、检索、复用 | 已实现 SharedMemory |
| 2 组以上连续任务 | 已实现 3 组连续任务 |
| 指标统计 | 已统计消息、字符、token、耗时、记忆命中、状态传递 |
| 不少于 10 轮连续任务 | 代码支持，正式提交需在目标环境运行 |
| openEuler 24.03-LTS-SP3 | 当前开发环境为 LTS-SP1，最终需在 SP3 复现 |

## 9. 还差多少

当前系统已经达到“可交付原型”水平，主要差距在正式评审材料和目标环境验证：

| 项目 | 状态 | 建议 |
| --- | --- | --- |
| SP3 复现 | 未完成 | 在 openEuler 24.03-LTS-SP3 上完整跑 smoke test、A/B、消融实验 |
| 真实 embedding 10 轮实验 | 未完成 | 使用 `EMBEDDING_BACKEND=ollama`、`bge-m3`、`EMBEDDING_TIMEOUT=60` 跑 10 轮 |
| token 统计 | 基本可用 | 如能安装 `tiktoken`，使用 `TOKEN_COUNT_METHOD=tiktoken` 补充真实 tokenizer 数据 |
| 系统技术亮点 | 已补强 | 已有 shared memory IPC 和 subprocess sandbox，可选继续补 Unix socket/mmap 对比 |
| 演示视频 | 未完成 | 建议录制命令行实验 + dashboard + SQLite 证据 |
| 报告数据稳定性 | 需要加强 | 至少跑 10 轮并报告均值、标准差和消融结果 |

如果按现在状态提交，属于“题目契合度高、工程完整、具备系统技术亮点”的水平；如果想继续冲奖，最值得补的是 SP3 实测、真实 embedding 10 轮消融、以及一组 shared memory 与 sqlite 状态后端的性能对比。
