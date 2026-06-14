# Multi-Agent Low-Overhead Communication Prototype

This is a minimal runnable prototype for the technical research task:

- multiple agents collaborate on a task
- two communication modes are supported: plain text and structured messages
- a shared memory store is available
- metrics are collected for messages, approximate tokens, elapsed time, memory hits, and non-text state transfers
- the LLM backend is swappable: mock first, OpenAI-compatible HTTP later

## Quick Start

Run the mock A/B experiment:

```bash
cd /home/pxf/llm/source_code
MEMORY_RESET=1 EXPERIMENT_ROUNDS=1 python3 -m experiments.run_ab
```

The result will be written to:

```text
reports/results.json
```

## Later: Connect A Server LLM

For an Ollama server, set:

```bash
cd /home/pxf/llm/source_code
export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=http://192.168.110.22:11434
export LLM_MODEL=qwen3:8b
export OLLAMA_THINK=false
export OLLAMA_NUM_PREDICT=256
export MEMORY_RESET=1
export EXPERIMENT_ROUNDS=1
python3 -m experiments.run_ab
```

For an OpenAI-compatible server, set:

```bash
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=http://your-server:8000/v1
export LLM_MODEL=your-model-name
export LLM_API_KEY=dummy
export MEMORY_RESET=1
export EXPERIMENT_ROUNDS=1
python3 -m experiments.run_ab
```

The OpenAI-compatible path only needs a chat-completions compatible endpoint:

```text
POST /v1/chat/completions
```

## Optional Components

### LangGraph orchestration

The prototype runs with a built-in sequential orchestrator by default. If LangGraph is installed, enable it with:

```bash
export ORCHESTRATOR=langgraph
python3 -m experiments.run_ab
```

If LangGraph is not installed, the system automatically falls back to sequential orchestration.

### Persistent memory

Shared memory is stored in SQLite:

```text
data/memory.sqlite
```

Useful environment variables:

```bash
export MEMORY_PATH=data/memory.sqlite
export MEMORY_RESET=1
```

### Real embeddings

The default embedding provider is a deterministic local hash embedding, so the system runs without extra dependencies.

To use a real Ollama embedding model, first deploy/pull an embedding-capable model on the Ollama server, then run:

```bash
export EMBEDDING_BACKEND=ollama
export OLLAMA_EMBED_MODEL=nomic-embed-text
export EMBEDDING_TIMEOUT=10
```

Your current `qwen3:8b` model is a chat/thinking/tools model; it is not listed as embedding-capable by the Ollama tags response.

### CodeAct tool execution

`ExecutorAgent` now runs a restricted Python action through `CodeActExecutor`. The tool result is written into structured message state and counted as a non-text transfer.

The registered tool functions are:

```text
read_file(path, max_chars=4000)
search_files(pattern="*.md", max_results=20)
load_json(path)
load_csv(path, max_rows=100)
make_markdown_table(rows, columns=None)
compute_numeric_metrics(values)
summarize_records(rows)
search_memory(query, limit=3)
```

Run the tool smoke test:

```bash
python3 -m experiments.test_tools
```
