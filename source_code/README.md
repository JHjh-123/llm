# 多智能体低开销通信实验系统

这是一个用于验证“多智能体协作中的低开销通信、状态传递与共享记忆”的实验原型。

系统目前支持：

- 多 Agent 协作：`planner -> researcher -> executor -> summarizer`
- 两种通信模式对比：纯文本通信 `text` 与结构化通信 `structured`
- 共享记忆：SQLite 持久化，支持关键词、标签、embedding 混合检索和动态记忆链接
- 状态交换：SQLite 持久化，Agent 间只传 embedding/tool result 的 `state_id`
- 指标采集：消息数、字符数、近似 token、耗时、记忆命中、非文本状态传递
- 真实 LLM 后端：Ollama 或 OpenAI-compatible HTTP
- CodeAct 工具执行：受限 Python 工具动作空间
- 浏览器对比面板：直接查看 text 和 structured 的差距

## 交付文档

项目只保留 3 个 Markdown 文档：

```text
README.md
docs/deployment.md
docs/technical_description.md
```

- `README.md`：项目概览、能力列表、常用运行命令。
- `docs/deployment.md`：环境配置、部署、测试和正式实验命令。
- `docs/technical_description.md`：具体做了什么、每个模块的作用、赛题覆盖情况和剩余差距。

## 依赖安装

建议在 `source_code` 目录下操作：

```bash
cd /home/pxf/llm/source_code
python3 -m pip install -r requirements.txt
```

说明：

- 核心顺序执行器、Ollama/OpenAI-compatible 调用、SQLite 记忆、CodeAct 工具和 dashboard 都只依赖 Python 标准库。
- `requirements.txt` 当前安装的是可选的 `langgraph`，用于启用 `ORCHESTRATOR=langgraph`。
- 如果只使用默认 `sequential` 编排，不安装依赖也能运行核心系统。
- 如果安装 `tiktoken`，可以设置 `TOKEN_COUNT_METHOD=tiktoken` 使用真实 tokenizer；否则默认使用 `unicode_heuristic`。

## 后端配置

`LLM_BACKEND` 必须显式配置。系统不会再默认走 mock 或假模型。

### Ollama 后端

```bash
cd /home/pxf/llm/source_code
export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=http://192.168.110.22:11434
export LLM_MODEL=qwen3:8b
export OLLAMA_THINK=false
export OLLAMA_NUM_PREDICT=256
```

如果要使用真实 embedding：

```bash
export EMBEDDING_BACKEND=ollama
export OLLAMA_EMBED_MODEL=bge-m3
export EMBEDDING_TIMEOUT=10
```

### OpenAI-compatible 后端

```bash
cd /home/pxf/llm/source_code
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=http://your-server:8000/v1
export LLM_MODEL=your-model-name
export LLM_API_KEY=dummy
```

该后端只要求服务兼容：

```text
POST /v1/chat/completions
```

## 运行 A/B 实验

运行 10 轮默认任务集：

```bash
cd /home/pxf/llm/source_code
export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=http://192.168.110.22:11434
export LLM_MODEL=qwen3:8b
export OLLAMA_THINK=false
export OLLAMA_NUM_PREDICT=256
export EMBEDDING_BACKEND=ollama
export OLLAMA_EMBED_MODEL=bge-m3
export MEMORY_RESET=1
export STATE_RESET=1
export EXPERIMENT_ROUNDS=10
export TOKEN_COUNT_METHOD=unicode_heuristic
python3 -m experiments.run_ab
```

输出文件：

```text
reports/results.json
data/memory.sqlite
data/state.sqlite
```

## 运行机制消融实验

消融实验会依次运行：

```text
text_baseline
structured_protocol
structured_state
structured_memory
structured_memory_graph
```

用于分别验证结构化协议、非文本状态引用、共享记忆和动态记忆网络的边际贡献。

```bash
cd /home/pxf/llm/source_code
export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=http://192.168.110.22:11434
export LLM_MODEL=qwen3:8b
export OLLAMA_THINK=false
export OLLAMA_NUM_PREDICT=128
export EMBEDDING_BACKEND=hash
export ABLATION_ROUNDS=1
python3 -m experiments.ablation
```

正式提交建议改为：

```bash
export EMBEDDING_BACKEND=ollama
export OLLAMA_EMBED_MODEL=bge-m3
export EMBEDDING_TIMEOUT=60
export ABLATION_ROUNDS=10
python3 -m experiments.ablation
```

输出文件：

```text
reports/ablation_results.json
data/ablation/
```

## 生成演示报告

```bash
cd /home/pxf/llm/source_code
LLM_BACKEND=ollama \
OLLAMA_BASE_URL=http://192.168.110.22:11434 \
LLM_MODEL=qwen3:8b \
OLLAMA_THINK=false \
OLLAMA_NUM_PREDICT=192 \
EMBEDDING_BACKEND=ollama \
OLLAMA_EMBED_MODEL=bge-m3 \
EMBEDDING_TIMEOUT=60 \
ORCHESTRATOR=langgraph \
MEMORY_RESET=1 \
STATE_RESET=1 \
DEMO_ROUNDS=10 \
python3 -m experiments.demo
```

输出文件：

```text
reports/demo_results.json
reports/demo_report.txt
```

## 运行浏览器对比面板

前台运行方式：

```bash
cd /home/pxf/llm/source_code
LLM_BACKEND=ollama \
OLLAMA_BASE_URL=http://192.168.110.22:11434 \
LLM_MODEL=qwen3:8b \
OLLAMA_THINK=false \
OLLAMA_NUM_PREDICT=128 \
EMBEDDING_BACKEND=ollama \
OLLAMA_EMBED_MODEL=bge-m3 \
python3 -m experiments.dashboard
```

打开浏览器访问：

```text
http://127.0.0.1:8765
```

如果当前 shell 设置了代理，命令行访问本机接口时可以绕过代理：

```bash
curl --noproxy '*' http://127.0.0.1:8765/api/config
```

### 后台运行 dashboard

```bash
cd /home/pxf/llm/source_code
LLM_BACKEND=ollama \
OLLAMA_BASE_URL=http://192.168.110.22:11434 \
LLM_MODEL=qwen3:8b \
OLLAMA_THINK=false \
OLLAMA_NUM_PREDICT=128 \
EMBEDDING_BACKEND=ollama \
OLLAMA_EMBED_MODEL=bge-m3 \
nohup python3 -m experiments.dashboard > reports/dashboard.log 2>&1 &
echo $! > reports/dashboard.pid
```

查看服务日志：

```bash
tail -f reports/dashboard.log
```

### 停止 dashboard 服务

如果是前台运行，直接在运行服务的终端按：

```text
Ctrl+C
```

如果是按上面的后台方式运行：

```bash
cd /home/pxf/llm/source_code
kill "$(cat reports/dashboard.pid)"
rm -f reports/dashboard.pid
```

如果忘了 PID，也可以查找进程后停止：

```bash
pgrep -af "python3 -m experiments.dashboard"
kill <PID>
```

## dashboard 对比口径

面板支持两种口径：

```text
isolated
```

`text` 和 `structured` 使用两个独立的临时 memory 数据库，主要观察通信编码本身的开销差距。

```text
shared
```

`text` 先运行，`structured` 后运行，共用同一个临时 memory 数据库，主要观察记忆复用带来的系统效果。

dashboard 每次运行会把最近一次结果写入：

```text
reports/dashboard_last.json
```

## 当前如何提高通信效率

系统目前通过这些方式降低通信开销：

- `structured` 模式用 `a/in/out/refs/state` 这类短字段组织消息。
- `StateStore` 将 embedding 和 CodeAct 结果写入 `data/state.sqlite`，协议只传 `state_id`。
- `SharedMemory` 为新记忆生成 keywords/links/access_count，形成可复用的动态记忆网络。
- `refs` 只传记忆 ID 或状态引用，避免重复传完整上下文。
- `state` 承载 embedding 维度、工具状态 ID、状态大小等元数据。
- `_short()` 会压缩结构化 payload 中较长的输入与输出。
- 协议握手开销与应用消息开销分开统计，方便判断结构化通信是否被启动成本抵消。

## 最终交付检查

```bash
python3 -m experiments.test_protocol
python3 -m experiments.test_tools
MEMORY_RESET=1 STATE_RESET=1 EXPERIMENT_ROUNDS=10 python3 -m experiments.run_ab
ABLATION_ROUNDS=10 python3 -m experiments.ablation
DEMO_ROUNDS=10 python3 -m experiments.demo
```

提交时保留 `reports/results.json`、`reports/ablation_results.json`、`data/memory.sqlite` 和 `data/state.sqlite`，并在 openEuler 24.03-LTS-SP3 上重复验证。

需要注意：

- 当前 token 统计方法是 `char_approx_4`，即按字符数近似估算，不是真实 tokenizer。
- 当前非文本传递是 embedding、memory refs 和 tool result 级别，还不是真正的 KV-cache 或 hidden-state 复用。
- CodeAct 当前是 AST 白名单轻隔离，不是 Docker/nsjail/WASM 强沙箱。

## 可选组件

### LangGraph 编排

默认使用内置顺序编排器。安装 `requirements.txt` 后可以启用 LangGraph：

```bash
export ORCHESTRATOR=langgraph
python3 -m experiments.run_ab
```

如果没有安装 LangGraph，系统会自动回退到 `sequential`。

### 共享记忆

默认记忆文件：

```text
data/memory.sqlite
```

常用环境变量：

```bash
export MEMORY_PATH=data/memory.sqlite
export MEMORY_RESET=1
```

测试脚本默认使用 `/tmp` 下的临时数据库，不会再污染 `data/memory.sqlite`。

## 测试命令

```bash
cd /home/pxf/llm/source_code
python3 -m compileall multi_agent experiments
python3 -m experiments.test_tools
python3 -m experiments.test_protocol
```
