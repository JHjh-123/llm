# 部署与运行文档

## 1. 目标环境

最终交付要求在 openEuler 24.03-LTS-SP3 上编译、运行和测试。当前开发机可先使用 WSL 中的 openEuler 环境验证功能：

```bash
cat /etc/os-release
python3 --version
```

正式提交前请在 SP3 环境重新执行本文件中的命令，并保留输出截图或日志。

## 2. 进入工程

```bash
cd /home/pxf/llm/source_code
```

Windows 资源管理器路径：

```text
\\wsl.localhost\openEuler-24.03\home\pxf\llm\source_code
```

## 3. 安装依赖

核心系统只依赖 Python 标准库。可选依赖用于 LangGraph 编排和真实 tokenizer：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install tiktoken
```

如果不能安装 `tiktoken`，系统会自动回退到 `unicode_heuristic` token 估算。

## 4. 配置 LLM 与 embedding

Ollama 示例：

```bash
export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=http://192.168.110.22:11434
export LLM_MODEL=qwen3:8b
export OLLAMA_THINK=false
export OLLAMA_NUM_PREDICT=192

export EMBEDDING_BACKEND=ollama
export OLLAMA_EMBED_MODEL=bge-m3
export EMBEDDING_TIMEOUT=60

export STATE_BACKEND=shared_memory
export CODEACT_SANDBOX=subprocess
export CODEACT_CPU_SECONDS=2
export CODEACT_MEMORY_MB=512
export CODEACT_TIMEOUT_SECONDS=5
```

如果只做流程验证，可以使用 hash embedding：

```bash
export EMBEDDING_BACKEND=hash
```

OpenAI-compatible 示例：

```bash
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=http://your-server:8000/v1
export LLM_MODEL=your-model
export LLM_API_KEY=dummy
```

## 5. 运行 smoke test

```bash
python3 -m experiments.test_protocol
python3 -m experiments.test_tools
```

预期输出：

```text
protocol smoke test passed
```

工具测试会输出一段 JSON，`ok` 应为 `true`。

## 6. 运行正式 10 轮实验

```bash
export MEMORY_RESET=1
export STATE_RESET=1
export STATE_BACKEND=shared_memory
export CODEACT_SANDBOX=subprocess
export EXPERIMENT_ROUNDS=10
export TOKEN_COUNT_METHOD=unicode_heuristic
python3 -m experiments.run_ab
```

输出：

```text
reports/results.json
```

生成可读演示报告：

```bash
export DEMO_ROUNDS=10
python3 -m experiments.demo
```

输出：

```text
reports/demo_results.json
reports/demo_report.txt
```

## 7. 运行 dashboard

```bash
python3 -m experiments.dashboard
```

浏览器访问：

```text
http://127.0.0.1:8765
```

如果从 Windows 访问 WSL 服务失败，请确认 WSL 网络和防火墙设置，或在 WSL 内执行：

```bash
curl --noproxy '*' http://127.0.0.1:8765/api/config
```

## 8. 运行机制消融实验

```bash
export ABLATION_ROUNDS=10
export TOKEN_COUNT_METHOD=unicode_heuristic
export STATE_BACKEND=shared_memory
export CODEACT_SANDBOX=subprocess
python3 -m experiments.ablation
```

输出：

```text
reports/ablation_results.json
data/ablation/
```

该实验包含 5 个变体：

```text
text_baseline
structured_protocol
structured_state
structured_memory
structured_memory_graph
```

用于向评审展示结构化协议、状态引用、共享记忆和动态记忆网络的边际贡献。

## 9. 交付清单

建议提交：

```text
source_code/
  multi_agent/
  experiments/
  docs/
  reports/
  README.md
  requirements.txt
```

必须包含：

- 完整源码
- `README.md`
- `docs/deployment.md`
- `docs/technical_description.md`
- `reports/results.json`
- `reports/ablation_results.json`
- `data/memory.sqlite`
- `data/state.sqlite`
- 演示视频
- openEuler 24.03-LTS-SP3 运行截图或日志
