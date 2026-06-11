# 面向多智能体协作的低开销通信、状态传递与共享记忆机制 —— 技术调研报告

> 赛道：应用创新（社区赛题，联系人：陈老师 chengong15@huawei.com / 李老师 liping136@huawei.com）
> 交付环境：openEuler 24.03-LTS-SP3（必须可编译、运行、测试）
> 报告日期：2026-06-11
> 调研方式：多源 web 检索 + 对抗式事实核验（37 条主张抽取，核验 25 条，**25 条全部确认，0 条被驳回**）

---

## 0. 结论速览（TL;DR）

| 维度 | 推荐选择 | 一句话理由 |
|------|---------|-----------|
| 运行时底座 | **图式运行时（LangGraph 类）+ 自研薄运行时做插桩** | 已有公开项目用 LangGraph + MCP 把工具做成可热插拔服务;评测要的消息/token/非文本计数,需要自研薄层来插桩 |
| 通信协议 | **自研薄结构化协议**(借鉴 MCP/A2A/ANP 概念,wire schema 自定义) | 协议生态(MCP/ACP/A2A/ANP)碎片化、场景化,不宜整体押注单一标准 |
| 结构化 vs 纯文本 | **必须真做 A/B**,不能假设结构化必胜 | PACT 证明结构化省 token、性能不降;但 NLT 反向证明自然语言工具调用准确率反而 +18.4pp。证据相互冲突,正是赛题要"A/B 对比"的意义 |
| 非文本状态传递 | **KV cache / embedding 复用**(KVCOMM 最贴合多 Agent 场景) | 真实可行、2–8x 加速;但**仅限同架构/同基座模型**,跨任意模型不可行 |
| 共享记忆 | 向量库 + 结构化记忆单元 schema | 向量库/记忆框架的 openEuler 兼容性**未被证据覆盖,需真机验证** |
| 工具执行 | **CodeAct(可执行 Python 作统一动作空间)** + 轻量沙箱 | 成功率 +20%、更少交互轮次;沙箱选型需在 openEuler 上实测 |

> ⚠️ **本次调研的范围缺口(必须自行真机验证,不能照搬结论)**:
> 1. 多框架横评(AutoGen/CrewAI/CAMEL/MetaGPT)——只有 LangGraph+MCP 有证据;
> 2. 向量库(Milvus/Qdrant/Chroma/FAISS)在 openEuler 24.03 的编译运行兼容性;
> 3. 记忆框架(MemGPT/Mem0/Letta)与记忆单元 schema 细节;
> 4. 沙箱(gVisor/Firecracker/WASM/nsjail/Docker)选型与 openEuler 支持;
> 5. IPC/共享内存/Socket/eBPF 的具体落地。
> 这些都未被本轮证据验证,**当作开放工程决策,在目标 OS 上直接测试**。

---

## 1. 运行时底座与框架选型

### 1.1 核心结论

- **推荐:图式 Agent 运行时 + MCP 作可热插拔的工具/通信层**。已有公开项目把 LangGraph 的 agent 连接到由自定义 MCP server 提供的远程工具,每个工具通过 SSE 或 STDIO 独立托管——正好就是赛题要的"模块化、可替换通信层"。MCP 规范确认 STDIO 与 HTTP/SSE/Streamable-HTTP 为标准传输,所以"独立托管、可替换的 server"是一等公民。*(confidence: high, 3-0)*
  - ⚠️ 这是单一 GitHub 项目佐证集成模式(博客级证据),但 MCP 传输由一手规范确认。
- **自研薄运行时的取舍**:仅在你需要给"消息条数 / token / 非文本传递次数"插桩做评测时才自建——而图式运行时本就能很好地暴露这些钩子。建议**底座用现成图式运行时,外面包一层自研薄运行时专门做插桩与 A/B 切换**。

### 1.2 框架横评(范围缺口,需自测)

赛题点名的 AutoGen/CrewAI/CAMEL/MetaGPT **未被本轮证据覆盖**。开放问题:哪个框架既能暴露"可替换通信层",又能注入评测钩子(计消息/token/非文本传递),且能在 openEuler 24.03-LTS-SP3 上干净编译运行?

> **建议:** 以 LangGraph(图式、状态显式、易插桩)为主候选先在 openEuler 上验证可装可跑,同时对比 AutoGen(对话式多 Agent)在评测注入上的便利性,再定夺。

---

## 2. 结构化通信协议设计

### 2.1 协议生态:碎片化,不宜整体押注

- arXiv:2505.02279 综述了四个协议:**MCP**(工具/资源访问)、**ACP**(Agent 间消息)、**A2A**(跨环境协作)、**ANP**(网络化 Agent 生态),各自面向不同部署场景的互操作性,整体是**碎片化、场景化**而非统一。*(confidence: high, 3-0)*
- **实践含义**:MCP 在工具访问上最成熟;能力发现/握手/协议映射可借鉴 A2A/ANP 的概念,但**wire schema 保持自研**——这恰好满足赛题"设计结构化通信协议 + 握手 + 能力发现 + 协议映射"的要求,且不被单一标准绑死。

### 2.2 高密度结构化消息:PACT 范式

- arXiv:2606.05304 分析了两种 MAS 拓扑下的五种 Agent 间通信策略,发现**没有单一固定策略普适最优**,提出 **PACT**:把消息重构为紧凑的"动作-状态记录"(action type / inputs / results),替代冗长自然语言。PACT 在不同架构下都一致改善"性能-成本"权衡——任务性能相当或更强,token 显著更少。*(confidence: high, 3-0;注:preprint 自报基准)*
- **直接对应赛题要求 (b)/(c)**:结构化模式应编码 **动作类型 / 参数 / 结果 / 能力描述**,A/B harness 对照纯文本基线度量 token 节省。

### 2.3 ⚠️ 反向证据:结构化不一定赢——必须真做 A/B

- arXiv:2510.14453(NLT)报告:**自然语言工具调用**相比 JSON/结构化函数调用,工具调用准确率**提升 18.4 个百分点**(16 模型 / 10 领域),输出方差降低约 **70%**(6400 次试验的析因设计)。*(confidence: medium, 3-0;单篇未经同行评审 preprint)*
- **这与"结构化必胜"假设相反**,正是赛题要求 (c)"可复现 A/B"的最强理由:
  - 结构化协议在 **token/开销密度** 上赢(PACT);
  - 纯语言调用可能在 **原始任务准确率/鲁棒性** 上赢(NLT),取决于模型和任务。
- **结论:把 PACT 和 NLT 都当作待你的 A/B 实验裁决的"假设",而非既定结论。** 这一点写进技术报告会显著加分(体现严谨性)。

---

## 3. 非文本中间状态传递(赛题要求 d 的关键)

### 3.1 核心约束:只在"同架构/同基座模型族"内可行

> ⚠️ **最重要的现实约束**:跨 LLM 的 KV/hidden-state 复用,只有在**模型共享同一架构/同一基座**时才稳健。任意跨模型的 embedding 或 hidden state 交换**不被证据支持,不应承诺。**

### 3.2 可落地的技术路线(按贴合度排序)

| 方法 | 论文 | 关键结果 | 适用性 |
|------|------|---------|--------|
| **KVCOMM**(最贴合多 Agent) | arXiv:2510.12872, NeurIPS 2025 | 免训练,在线 anchor pool 对齐不同前缀下重叠内容的 cache offset,解决"offset 方差";**>70% 复用率**,5-Agent 全连接下 **最高 7.8x** 加速,TTFT 从 ~430ms 降到 ~55ms | **3+ Agent 协作的首选非文本传递机制**,直接支撑指标 (g) |
| **DroidSpeak**(跨节点不同 LLM) | arXiv:2411.02820 | 首个跨节点复用 KV cache 的分布式系统,限**同架构**(同基座微调变体);复用 input embedding(E cache)+ KV cache,**最高 4x 吞吐、~3.1x prefill 加速**,质量损失可忽略;识别关键层、只选择性重算 | 复合/agentic 系统共享 context 前缀场景 |
| **Prompt Cache**(段级复用) | arXiv:2311.04934 | 复用复现文本段的 attention/KV 状态,无需改参数;TTFT 加速 **~8x(GPU)到 ~60x(CPU)**,长文档 QA 最显著 | 共享记忆 + 检索管线 |
| **CacheBlend**(RAG 非前缀融合) | arXiv:2405.16444, EuroSys 2025 最佳论文 | 解决 RAG 检索块非前缀导致前缀缓存命中率低的问题;复用任意块 KV + 选择性重算小 token 子集("cached knowledge fusion");命中率近 100%,TTFT 降 **2.2–3.3x**,吞吐升 **2.8–5x**,无质量损失、无额外存储 | 语义检索记忆喂给 Agent 的场景 |

### 3.3 务实的兜底设计

- 若团队部署是**单模型**:可用 KVCOMM/DroidSpeak 式 KV/embedding 复用,直接满足"非文本传递"。
- 若是**多模型**或 API 隐藏了 hidden state:兜底用 **embedding 向量经共享记忆传递**——既保留"非文本"精神,又工程可行。
- 注:KVCOMM/DroidSpeak/CacheBlend 的"无质量损失"均为作者在特定模型/数据集上的自评,未经独立复现。

---

## 4. 共享记忆系统

### 4.1 现状:范围缺口,需真机验证

赛题点名的向量库(Milvus/Qdrant/Chroma/FAISS)选型、在 openEuler 24.03-LTS-SP3 的编译运行兼容性,以及记忆框架(MemGPT/Mem0/Letta)与记忆单元 schema,**本轮证据均未覆盖**,作为开放工程决策处理。

### 4.2 落地建议(基于赛题硬性要求 + 通用工程经验)

- **记忆单元 schema**(赛题硬性要求,务必逐字段覆盖):记忆 ID、来源 Agent、创建时间、任务主题、摘要描述,加上标签、embedding 向量。
- **检索三通道**(赛题要求):关键词、标签、语义相似度;语义检索由向量库支撑。
- **向量库选型建议**:Chroma(轻量、嵌入式、易部署,适合赛题原型) → Qdrant(Rust、性能好、单机友好) → Milvus(功能全但部署重)。**在 openEuler 24.03 上逐一验证可装可跑后再定**(Milvus 在某些环境有兼容 issue,需实测)。
- **记忆复用命中率评估**(赛题指标 g):设计 2 组关联连续任务,统计第二组任务命中第一组沉淀记忆的比例、命中带来的 token/耗时节省。
- 可借鉴 MemGPT/Mem0/Letta 的记忆分层(core memory / archival memory)与"memory blocks"思路(需自行验证其在 openEuler 上可用性)。

---

## 5. 评测方法论(赛题占比高:通信效率 25 + 实验验证 15)

### 5.1 A/B 对比实验设计

- **两种模式**:纯文本协作模式 vs 结构化协议协作模式,**相同任务、相同模型、相同随机种子**,可复现。
- **核心指标**(赛题要求 g 逐项统计):
  - Agent 间消息次数
  - 文本通信 token / 字符开销
  - 非文本状态传递次数 + 数据规模
  - 单任务总耗时
  - 共享记忆命中率
  - 整体性能提升
- **统计严谨性**:多次试验 + 固定种子 + 配对比较(paired comparison),给出均值与方差,而非单次跑分。这能可信地裁决 PACT(结构化省 token)vs NLT(自然语言准确率高)的冲突。
- **关联任务设计**(要求 f):至少 2 组有关联的连续任务,验证结构化通信 + 非文本传递 + 记忆复用在**减少重复计算、降低协作开销、提升效率**上的实际效果;稳定执行不少于 10 轮。

### 5.2 加分点

把"PACT vs NLT 的证据冲突 + 我们用 A/B 裁决"写进技术报告,体现对前沿的把握和实验严谨性。

---

## 6. CodeAct 模式与轻量沙箱

### 6.1 CodeAct:已验证的推荐范式

- CodeAct(arXiv:2402.01030, ICML 2024)用**可执行 Python 作为统一动作空间**,取代 text/JSON 动作,让 Agent 能基于执行反馈修订动作、组合多个操作。在 17 个 LLM、API-Bank 与 M³ToolEval 上,**成功率最高 +20%**,且用更少交互轮次完成任务(把本需多次单独工具调用的工作打包)。*(confidence: high, 3-0)*
- **契合赛题要求 (j)**:适配工具执行 Agent 角色,天然搭配轻量沙箱;更少轮次也直接改善指标 (g) 的消息次数与时延。

### 6.2 沙箱选型(范围缺口,需 openEuler 实测)

gVisor/Firecracker/WASM/nsjail/Docker 的选型与 openEuler 支持**未被证据覆盖**。通用权衡参考:

| 沙箱 | 隔离强度 | 启动开销 | openEuler 适配性 | 适用 |
|------|---------|---------|-----------------|------|
| **nsjail** | 中(namespace+seccomp) | 极低 | 需验证 | 轻量、低延迟首选候选 |
| **gVisor** | 高(用户态内核) | 中 | 需验证 | 安全性高 |
| **Firecracker** | 高(microVM) | 中 | 需验证 | 强隔离但偏重 |
| **WASM**(如 Wasmtime) | 中高 | 极低 | 需验证 | 低延迟、可隔离,契合赛题"轻量沙箱"措辞 |
| **Docker** | 中 | 中 | 好(成熟) | 部署简单,保底 |

> **建议:** 原型先用 Docker(保底、openEuler 成熟)跑通 CodeAct 闭环,再向 nsjail/WASM(低延迟、可隔离,符合赛题"低延迟、可隔离"措辞)演进。

---

## 7. 待验证的开放问题(开工前/中逐一落实)

1. LangGraph vs AutoGen vs CrewAI vs CAMEL vs MetaGPT:哪个最好地暴露"可替换通信层 + 可注入评测钩子",且能在 openEuler 24.03-LTS-SP3 干净编译运行?
2. Milvus/Qdrant/Chroma/FAISS 哪个在 openEuler 24.03-LTS-SP3 上真正可装、性能可用?记忆单元 schema 与命中率评估法具体怎么定(借鉴 MemGPT/Mem0/Letta)?
3. 非文本传递:部署是单模型(可用 KVCOMM/DroidSpeak 式 KV/embedding 复用)还是多模型?若 API 隐藏 hidden state,兜底用什么(如经共享记忆传 embedding 向量)?
4. 沙箱:gVisor/Firecracker/WASM/nsjail/Docker 哪个在 openEuler 上给出合适的安全/开销权衡?A/B 评测如何做统计设计(试验次数、种子、配对比较)以可信裁决 PACT-vs-NLT 的结构化 vs 自然语言之争?

---

## 8. 建议的技术路线图(分阶段、保底优先)

| 阶段 | 目标 | 对应分值 |
|------|------|---------|
| **0. 环境验证** | openEuler 24.03 搭建;验证图式运行时、向量库、沙箱可装可跑 | 前置 |
| **1. 多 Agent 协作骨架(保底)** | ≥3 个 Agent(规划/检索/总结/执行覆盖 3 类),跑通多步骤任务,稳定 10+ 轮 | 系统完整性 20 |
| **2. 双模式 + 评测模块** | 纯文本模式 + 结构化协议模式 + 评测插桩(消息/token/耗时/命中率) | 通信效率 25 + 实验验证 15 |
| **3. 共享记忆** | 向量库 + 记忆单元 schema + 三通道检索 + 2 组关联任务复用 | 记忆复用 20 |
| **4. 非文本状态传递** | KVCOMM/embedding 复用(单模型族)或 embedding-经记忆兜底 | 状态传递创新 20 |
| **5. 加分创新** | CodeAct + 轻量沙箱、IPC/共享内存优化、eBPF 观测协作开销 | 加分 |

---

## 附录 A：核心参考来源(全部一手/已核验)

| 来源 | 类型 | 用途 |
|------|------|------|
| https://arxiv.org/html/2505.02279v1 | 一手(综述) | MCP/ACP/A2A/ANP 协议生态碎片化 |
| https://huggingface.co/papers/2606.05304 | 一手(preprint) | PACT 结构化"动作-状态记录",省 token、性能不降 |
| https://arxiv.org/html/2510.14453v1 | 一手(preprint) | NLT 反向证据:自然语言工具调用准确率 +18.4pp |
| https://arxiv.org/html/2510.12872v1 | 一手(NeurIPS 2025) | KVCOMM 多 Agent KV 复用 >70%、7.8x 加速 |
| https://arxiv.org/abs/2411.02820 | 一手 | DroidSpeak 跨节点 KV/E cache 复用(同架构) |
| https://arxiv.org/abs/2311.04934 | 一手 | Prompt Cache 段级 KV 复用 |
| https://arxiv.org/abs/2405.16444 | 一手(EuroSys 2025 最佳论文) | CacheBlend RAG 非前缀 KV 融合 |
| https://arxiv.org/abs/2402.01030 | 一手(ICML 2024) | CodeAct 可执行 Python 动作空间,成功率 +20% |
| https://github.com/junfanz1/MCP-MultiServer-Interoperable-Agent2Agent-LangGraph-AI-System | 博客级 | LangGraph + MCP 可热插拔工具层存在性证明 |

## 附录 B：调研可信度说明

- 本轮 25 条核验主张**全部确认(25/25),无被驳回**,但存在**范围缺口**:框架横评、向量库选型、记忆框架、沙箱选型、IPC/eBPF 细节均未被证据覆盖——作为开放工程决策,需在 openEuler 24.03-LTS-SP3 上直接验证。
- **证据强度**:PACT(2606.05304)与 NLT(2510.14453)均为近期 preprint,自报基准、未独立复现,且**结论方向相反**——务必当作你的 A/B 要裁决的"假设",而非既定结论。
- KVCOMM/DroidSpeak/CacheBlend 的"无质量损失"为作者在特定模型/数据集上的自评。
- **对要求 (d) 的关键约束**:跨 LLM 的 KV/hidden-state 复用仅在同架构/同基座族稳健;任意跨模型 embedding/hidden state 交换不被支持,**不要承诺**。
- **时效性**:MCP 传输在演进(Streamable HTTP 现优于纯 SSE);协议格局在 2025/2026 年中仍在快速变化。
