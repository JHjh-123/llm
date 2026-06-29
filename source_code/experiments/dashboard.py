from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from multi_agent.environment import collect_environment
from multi_agent.runner import ExperimentRunner
from multi_agent.tasks import DEFAULT_TASKS, TASK_GROUPS


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RESULTS_PATH = Path("reports/results.json")
ABLATION_PATH = Path("reports/ablation_results.json")


def main() -> None:
    host = os.getenv("DASHBOARD_HOST", DEFAULT_HOST)
    port = int(os.getenv("DASHBOARD_PORT", str(DEFAULT_PORT)))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MultiAgentDashboard/2.0"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(_html())
            return
        if self.path == "/api/config":
            self._send_json(
                {
                    "environment": collect_environment(),
                    "task_groups": TASK_GROUPS,
                    "default_tasks": DEFAULT_TASKS,
                    "modes": ["ab", "text", "structured"],
                    "round_options": [1, 10],
                }
            )
            return
        if self.path == "/api/tasks":
            self._send_json({"ok": True, "task_groups": TASK_GROUPS, "default_tasks": DEFAULT_TASKS})
            return
        if self.path in {"/api/latest", "/api/sample"}:
            self._send_json(_load_latest_results())
            return
        if self.path == "/api/ablation":
            self._send_json(_load_json_result(ABLATION_PATH))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            payload = self._read_json()
            task_group = str(payload.get("task_group") or "all")
            rounds = int(payload.get("rounds") or 1)
            mode = str(payload.get("mode") or "ab")
            result = run_experiment(task_group=task_group, rounds=rounds, mode=mode)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return
        self._send_json({"ok": True, "path": str(RESULTS_PATH), "result": result})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, body: Any, status: int = 200) -> None:
        encoded = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_experiment(task_group: str, rounds: int, mode: str) -> dict[str, Any]:
    if rounds not in {1, 10}:
        raise ValueError("rounds must be 1 or 10")
    if mode not in {"ab", "text", "structured"}:
        raise ValueError("mode must be ab, text, or structured")

    tasks = _select_tasks(task_group)
    run_id = uuid.uuid4().hex[:10]
    started = time.perf_counter()

    env = {
        "MEMORY_PATH": str(Path(tempfile.gettempdir()) / f"llm_dashboard_{run_id}_memory.sqlite"),
        "STATE_PATH": str(Path(tempfile.gettempdir()) / f"llm_dashboard_{run_id}_state.sqlite"),
        "MEMORY_RESET": "1",
        "STATE_RESET": "1",
    }
    with _temporary_env(env):
        runner = ExperimentRunner()
        if mode == "ab":
            result = runner.run_ab(tasks=tasks, rounds=rounds)
        else:
            runs: list[dict[str, Any]] = []
            for round_index in range(rounds):
                for task in tasks:
                    run = runner.run_task(task=task, mode=mode)
                    run["round"] = round_index + 1
                    runs.append(run)
            result = {
                "summary": _summarize_runs(runs),
                "runs": runs,
                "memory": runner.memory.to_dict(),
                "states": runner.state_store.to_dict(),
                "environment": collect_environment(),
            }

    result.update(
        {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "task_group": task_group,
            "tasks": tasks,
            "rounds": rounds,
            "requested_mode": mode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    )
    _add_comparison_summary(result)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _select_tasks(task_group: str) -> list[str]:
    if task_group == "all":
        return list(DEFAULT_TASKS)
    if task_group not in TASK_GROUPS:
        raise ValueError(f"unknown task_group: {task_group}")
    return list(TASK_GROUPS[task_group])


def _load_latest_results() -> dict[str, Any]:
    for path in (RESULTS_PATH, ABLATION_PATH, Path("reports/dashboard_last.json")):
        data = _load_json_result(path)
        if data.get("ok"):
            return data
    return {"ok": False, "error": "暂无实验结果，请先运行实验"}


def _load_json_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "error": f"{path} not found"}
    return {"ok": True, "path": str(path), "result": json.loads(path.read_text(encoding="utf-8"))}


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("text", "structured"):
        metrics = [run["metrics"] for run in runs if run.get("mode") == mode]
        summary[mode] = {
            "runs": len(metrics),
            "avg_messages": _avg(metrics, "message_count"),
            "avg_chars": _avg(metrics, "char_count"),
            "avg_approx_tokens": _avg(metrics, "approx_token_count"),
            "avg_elapsed_ms": _avg(metrics, "elapsed_ms"),
            "avg_memory_hits": _avg(metrics, "memory_hit_count"),
            "avg_non_text_transfers": _avg(metrics, "non_text_transfer_count"),
            "avg_non_text_transfer_size": _avg(metrics, "non_text_transfer_size"),
        }
    return summary


def _add_comparison_summary(result: dict[str, Any]) -> None:
    summary = result.setdefault("summary", _summarize_runs(result.get("runs", [])))
    text = summary.get("text", {})
    structured = summary.get("structured", {})
    summary["comparison"] = {
        "token_delta_pct": _pct_delta(structured.get("avg_approx_tokens"), text.get("avg_approx_tokens")),
        "char_delta_pct": _pct_delta(structured.get("avg_chars"), text.get("avg_chars")),
        "elapsed_delta_pct": _pct_delta(structured.get("avg_elapsed_ms"), text.get("avg_elapsed_ms")),
    }


def _avg(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0) or 0) for item in metrics]
    return round(sum(values) / len(values), 3) if values else 0.0


def _pct_delta(new: Any, old: Any) -> float | None:
    new_value = float(new or 0)
    old_value = float(old or 0)
    if old_value == 0:
        return None
    return round((new_value - old_value) / old_value * 100, 2)


@contextmanager
def _temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多智能体低开销通信实验面板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef3f8;
      --panel: #ffffff;
      --panel-soft: #f7fafc;
      --text: #172033;
      --muted: #607086;
      --line: #d9e2ec;
      --text-mode: #475569;
      --structured: #0f766e;
      --structured-soft: #e6fffb;
      --state: #7c3aed;
      --memory: #15803d;
      --bad: #b91c1c;
      --warn: #b45309;
      --blue: #2563eb;
      --shadow: 0 10px 28px rgba(15, 23, 42, .06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 20px 30px 16px;
      background: linear-gradient(90deg, #ffffff 0%, #f3fbfa 58%, #eff6ff 100%);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 26px; line-height: 1.25; }
    .subtitle { margin-top: 6px; color: var(--muted); font-size: 14px; }
    main { max-width: none; margin: 0; padding: 18px 30px 34px; }
    .control {
      display: grid;
      grid-template-columns: minmax(320px, 2fr) 190px 270px repeat(3, 138px);
      gap: 12px;
      align-items: end;
      margin-bottom: 10px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 750; }
    select, button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
      padding: 0 10px;
    }
    button { cursor: pointer; font-weight: 800; }
    button.primary { border-color: var(--structured); background: var(--structured); color: white; }
    button:disabled { cursor: wait; opacity: .62; }
    .status {
      min-height: 22px;
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
      padding-left: 4px;
    }
    .section {
      margin-bottom: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .section h2 {
      margin: 0;
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
      font-size: 16px;
    }
    .overview {
      display: grid;
      grid-template-columns: 1.2fr repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .overview-card {
      min-height: 88px;
      padding: 13px 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .overview-card .label { color: var(--muted); font-size: 12px; font-weight: 800; }
    .overview-card .value { margin-top: 8px; font-size: 24px; font-weight: 900; overflow-wrap: anywhere; }
    .overview-card .hint { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .flow {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 10px;
      padding: 14px;
    }
    .agent {
      min-height: 86px;
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 10px;
      background: #fff;
    }
    .agent .name { font-weight: 850; }
    .agent .state { margin-top: 10px; color: var(--muted); font-size: 12px; }
    .agent.running { border-color: var(--blue); box-shadow: inset 0 0 0 1px var(--blue); }
    .agent.done { border-color: #86efac; background: #f0fdf4; }
    .agent.failed { border-color: #fecaca; background: #fef2f2; }
    .compare {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .mode-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 14px;
      box-shadow: var(--shadow);
    }
    .mode-card h2 { display: flex; justify-content: space-between; align-items: baseline; }
    .mode-card.text h2 { color: var(--text-mode); }
    .mode-card.structured h2 { color: var(--structured); }
    .summary-text {
      padding: 12px 14px;
      min-height: 126px;
      color: #263244;
      line-height: 1.45;
      white-space: pre-wrap;
      border-bottom: 1px solid var(--line);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(128px, 1fr));
      gap: 8px;
      padding: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 10px;
      background: var(--panel-soft);
      min-height: 72px;
    }
    .metric .label { color: var(--muted); font-size: 12px; font-weight: 750; }
    .metric .value { margin-top: 6px; font-size: 20px; font-weight: 850; overflow-wrap: anywhere; }
    .refs { padding: 0 12px 12px; display: grid; gap: 8px; }
    .pill-line { display: flex; flex-wrap: wrap; gap: 6px; min-height: 28px; align-items: center; }
    .pill {
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: #eef2ff;
      color: #3730a3;
      overflow-wrap: anywhere;
    }
    .pill.memory { background: #dcfce7; color: var(--memory); }
    .pill.state { background: #f3e8ff; color: var(--state); }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr)) 190px 190px;
      gap: 10px;
      padding: 12px;
    }
    .chart, .small-card {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 12px;
      background: var(--panel-soft);
      min-height: 126px;
    }
    .chart-title, .small-card .label { color: var(--muted); font-size: 12px; font-weight: 800; margin-bottom: 12px; }
    .bar-row { display: grid; grid-template-columns: 74px 1fr 70px; gap: 8px; align-items: center; margin: 8px 0; font-size: 12px; }
    .bar-track { height: 14px; border-radius: 99px; background: #e5e7eb; overflow: hidden; }
    .bar-fill { height: 100%; min-width: 2px; border-radius: 99px; }
    .bar-fill.text { background: #64748b; }
    .bar-fill.structured { background: var(--structured); }
    .small-card .value { font-size: 28px; font-weight: 900; }
    .good { color: var(--memory); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .table-tools { padding: 10px 14px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 10px; align-items: end; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; min-width: 1180px; border-collapse: collapse; font-size: 13px; }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); background: var(--panel-soft); font-size: 12px; }
    details trace-view { display: block; }
    pre {
      margin: 8px 0 0;
      padding: 10px;
      border-radius: 6px;
      background: #111827;
      color: #e5e7eb;
      overflow: auto;
      max-height: 240px;
      font-size: 12px;
      line-height: 1.45;
    }
    .empty {
      padding: 18px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    @media (max-width: 1180px) {
      main { padding: 14px 12px 24px; }
      .control, .compare, .chart-grid, .overview { grid-template-columns: 1fr; }
      .flow { grid-template-columns: repeat(2, 1fr); }
      .metrics { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <h1>多智能体低开销通信实验面板</h1>
    <div class="subtitle">纯文本基线与结构化协议协作效果评估看板</div>
  </header>
  <main>
    <div class="control">
      <label>任务组
        <select id="taskGroup">
          <option value="all">全部默认任务</option>
          <option value="protocol_design">协议设计任务组 protocol_design</option>
          <option value="memory_reuse">共享记忆任务组 memory_reuse</option>
          <option value="tool_execution">CodeAct 工具执行任务组 tool_execution</option>
        </select>
      </label>
      <label>实验轮数
        <select id="rounds">
          <option value="1" selected>快速演示：1 轮</option>
          <option value="10">正式实验：10 轮</option>
        </select>
      </label>
      <label>运行模式
        <select id="mode">
          <option value="ab" selected>A/B 对比：text + structured</option>
          <option value="text">只运行 text</option>
          <option value="structured">只运行 structured</option>
        </select>
      </label>
      <button class="primary" id="runBtn">运行实验</button>
      <button id="loadBtn">加载最近结果</button>
      <button id="clearBtn">清空结果</button>
    </div>
    <div class="status" id="status"></div>

    <section class="section">
      <h2>Agent 流程</h2>
      <div class="flow" id="agentFlow"></div>
    </section>

    <div id="content" class="empty">暂无实验结果，请先运行实验或加载最近结果。</div>
  </main>

  <script>
    const agents = [
      {key: 'router', label: '路由 Router'},
      {key: 'planner', label: '规划 Planner'},
      {key: 'researcher', label: '检索 Researcher'},
      {key: 'executor', label: '执行 Executor'},
      {key: 'summarizer', label: '总结 Summarizer'},
      {key: 'verifier', label: '验证 Verifier'}
    ];
    const stateText = {waiting: '等待', running: '运行中', done: '完成', failed: '失败'};
    const taskGroup = document.querySelector('#taskGroup');
    const rounds = document.querySelector('#rounds');
    const mode = document.querySelector('#mode');
    const runBtn = document.querySelector('#runBtn');
    const loadBtn = document.querySelector('#loadBtn');
    const clearBtn = document.querySelector('#clearBtn');
    const statusBox = document.querySelector('#status');
    const flow = document.querySelector('#agentFlow');
    const content = document.querySelector('#content');
    let currentData = null;
    let progressTimer = null;

    function initFlow(states = {}) {
      flow.innerHTML = agents.map(agent => {
        const state = states[agent.key] || 'waiting';
        return `<div class="agent ${state}"><div class="name">${agent.label}</div><div class="state">${stateText[state] || state}</div></div>`;
      }).join('');
    }

    async function loadConfig() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      const env = cfg.environment || {};
      statusBox.textContent = `后端模型：${env.llm_backend || 'required'} / ${env.llm_model || 'required'}；状态后端：${env.state_backend || 'shared_memory'}；Token 统计：${env.token_count_method || ''}`;
    }

    async function runExperiment() {
      setBusy(true, '正在运行真实实验。1 轮适合现场演示，10 轮需要更久。');
      startProgress();
      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({task_group: taskGroup.value, rounds: Number(rounds.value), mode: mode.value})
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        currentData = normalizeResult(data.result, data.path);
        renderAll(currentData);
        markFlowFromRuns(currentData.runs);
        statusBox.textContent = `运行完成：${data.path}，run_id=${data.result.run_id || 'n/a'}`;
      } catch (err) {
        markFailed();
        content.className = 'empty';
        content.textContent = String(err.message || err);
        statusBox.textContent = '运行失败。';
      } finally {
        stopProgress();
        setBusy(false);
      }
    }

    async function loadLatest() {
      setBusy(true, '正在加载最近结果...');
      try {
        const res = await fetch('/api/latest');
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        currentData = normalizeResult(data.result, data.path);
        renderAll(currentData);
        markFlowFromRuns(currentData.runs);
        statusBox.textContent = `已加载：${data.path}`;
      } catch (err) {
        content.className = 'empty';
        content.textContent = String(err.message || err);
      } finally {
        setBusy(false);
      }
    }

    function clearResults() {
      currentData = null;
      content.className = 'empty';
      content.textContent = '暂无实验结果，请先运行实验或加载最近结果。';
      initFlow();
      statusBox.textContent = '已清空页面结果。';
    }

    function setBusy(busy, text) {
      runBtn.disabled = busy;
      loadBtn.disabled = busy;
      clearBtn.disabled = busy;
      if (text) statusBox.textContent = text;
    }

    function startProgress() {
      let index = 0;
      progressTimer = setInterval(() => {
        const states = {};
      agents.forEach((agent, i) => states[agent.key] = i < index ? 'done' : i === index ? 'running' : 'waiting');
        initFlow(states);
        index = (index + 1) % agents.length;
      }, 900);
    }

    function stopProgress() {
      if (progressTimer) clearInterval(progressTimer);
      progressTimer = null;
    }

    function markFailed() {
      const states = {};
      agents.forEach(agent => states[agent.key] = 'failed');
      initFlow(states);
    }

    function markFlowFromRuns(runs) {
      const trace = runs.flatMap(run => (run.metrics && run.metrics.message_trace) || []);
      const alias = {
        router: 'router',
        planner: 'planner',
        researcher: 'researcher',
        executor: 'executor',
        summarizer: 'summarizer',
        verifier: 'verifier',
        user: 'verifier'
      };
      const done = {};
      for (const item of trace) {
        if (item.from && alias[String(item.from).toLowerCase()]) done[alias[String(item.from).toLowerCase()]] = 'done';
        if (item.to && alias[String(item.to).toLowerCase()]) done[alias[String(item.to).toLowerCase()]] = 'done';
      }
      const states = {};
      agents.forEach(agent => states[agent.key] = done[agent.key] || 'done');
      initFlow(states);
    }

    function normalizeResult(result, path) {
      if (result.variants) return normalizeAblation(result, path);
      const runs = (result.runs || []).map(run => ({...run, display_mode: run.mode}));
      const summary = buildSummary(runs, result.summary);
      return {kind: 'run', path, raw: result, runs, summary};
    }

    function normalizeAblation(result, path) {
      const byName = Object.fromEntries((result.variants || []).map(v => [v.name, v]));
      const text = byName.text_baseline || {runs: [], summary: {}};
      const structured = byName.structured_memory_graph || byName.structured_protocol || {runs: [], summary: {}};
      const runs = [
        ...(text.runs || []).map(r => ({...r, mode: 'text', display_mode: 'text', variant: text.name || 'text_baseline'})),
        ...(structured.runs || []).map(r => ({...r, mode: 'structured', display_mode: 'structured', variant: structured.name || 'structured'}))
      ];
      const summary = buildSummary(runs);
      return {kind: 'ablation', path, raw: result, runs, summary};
    }

    function buildSummary(runs, existing) {
      const summary = existing || {};
      for (const name of ['text', 'structured']) {
        const metrics = runs.filter(r => r.mode === name).map(r => r.metrics || {});
        summary[name] = {
          ...(summary[name] || {}),
          runs: metrics.length,
          avg_messages: avg(metrics, 'message_count'),
          avg_chars: avg(metrics, 'char_count'),
          avg_approx_tokens: avg(metrics, 'approx_token_count'),
          avg_elapsed_ms: avg(metrics, 'elapsed_ms'),
          avg_memory_hits: avg(metrics, 'memory_hit_count'),
          avg_non_text_transfers: avg(metrics, 'non_text_transfer_count'),
          avg_non_text_transfer_size: avg(metrics, 'non_text_transfer_size')
        };
      }
      summary.comparison = {
        token_delta_pct: pct(summary.structured.avg_approx_tokens, summary.text.avg_approx_tokens),
        char_delta_pct: pct(summary.structured.avg_chars, summary.text.avg_chars),
        elapsed_delta_pct: pct(summary.structured.avg_elapsed_ms, summary.text.avg_elapsed_ms)
      };
      return summary;
    }

    function renderAll(data) {
      content.className = '';
      content.innerHTML = `
        ${overview(data)}
        <div class="compare">${modeColumn('纯文本协作模式', 'text', data)}${modeColumn('结构化协议协作模式', 'structured', data)}</div>
        <section class="section"><h2>指标图表：文本基线 vs 结构化协议</h2>${charts(data.summary)}</section>
        <section class="section"><h2>实验结果明细表</h2>${resultTable(data.runs)}</section>
      `;
      document.querySelector('#modeFilter').addEventListener('change', event => renderTableRows(data.runs, event.target.value));
      renderTableRows(data.runs, 'all');
    }

    function overview(data) {
      const s = data.summary || {};
      const c = s.comparison || {};
      const runCount = (data.runs || []).length;
      const taskCount = new Set((data.runs || []).map(run => run.task)).size;
      return `<div class="overview">
        <div class="overview-card">
          <div class="label">当前结果文件</div>
          <div class="value">${escapeHtml(data.path || '未加载')}</div>
          <div class="hint">${runCount} 条运行记录 / ${taskCount} 个任务</div>
        </div>
        <div class="overview-card">
          <div class="label">Token 变化率</div>
          <div class="value ${deltaClass(c.token_delta_pct)}">${fmtPct(c.token_delta_pct)}</div>
          <div class="hint">结构化相对纯文本</div>
        </div>
        <div class="overview-card">
          <div class="label">字符数变化率</div>
          <div class="value ${deltaClass(c.char_delta_pct)}">${fmtPct(c.char_delta_pct)}</div>
          <div class="hint">消息载荷大小</div>
        </div>
        <div class="overview-card">
          <div class="label">耗时变化率</div>
          <div class="value ${deltaClass(c.elapsed_delta_pct)}">${fmtPct(c.elapsed_delta_pct)}</div>
          <div class="hint">端到端执行时间</div>
        </div>
        <div class="overview-card">
          <div class="label">结构化证据</div>
          <div class="value">${fmt((s.structured || {}).avg_non_text_transfers)} 次</div>
          <div class="hint">平均状态传递</div>
        </div>
      </div>`;
    }

    function modeColumn(title, name, data) {
      const runs = data.runs.filter(r => r.mode === name);
      const latest = runs[runs.length - 1] || {};
      const m = data.summary[name] || {};
      const memoryIds = uniqueIds(runs, 'memory_hit_ids');
      const stateIds = uniqueIds(runs, 'state_ref_ids');
      return `<section class="mode-card ${name}">
        <h2>${title}<span>${m.runs || 0} 次运行</span></h2>
        <div class="summary-text">${escapeHtml(short(latest.final || '暂无输出摘要', 520))}</div>
        <div class="metrics">
          ${metric('消息数', m.avg_messages)}
          ${metric('字符数', m.avg_chars)}
          ${metric('近似 Token', m.avg_approx_tokens)}
          ${metric('总耗时 ms', m.avg_elapsed_ms)}
          ${metric('记忆命中', m.avg_memory_hits, 'good')}
          ${metric('状态传递', m.avg_non_text_transfers, name === 'structured' ? 'warn' : '')}
          ${metric('传递大小', m.avg_non_text_transfer_size)}
          ${metric('链路步数', traceCount(runs))}
        </div>
        ${name === 'structured' ? `<div class="refs">
          <div><strong>共享记忆引用 memory_id</strong><div class="pill-line">${pills(memoryIds, 'memory')}</div></div>
          <div><strong>非文本状态引用 state_id</strong><div class="pill-line">${pills(stateIds, 'state')}</div></div>
        </div>` : ''}
      </section>`;
    }

    function charts(summary) {
      const text = summary.text || {};
      const structured = summary.structured || {};
      return `<div class="chart-grid">
        ${barChart('平均 Token 对比', text.avg_approx_tokens, structured.avg_approx_tokens)}
        ${barChart('平均字符数对比', text.avg_chars, structured.avg_chars)}
        ${barChart('平均耗时 ms 对比', text.avg_elapsed_ms, structured.avg_elapsed_ms)}
        <div class="small-card"><div class="label">记忆命中</div><div class="value good">${fmt(structured.avg_memory_hits)}</div><div class="label">结构化平均</div></div>
        <div class="small-card"><div class="label">状态传递</div><div class="value warn">${fmt(structured.avg_non_text_transfers)}</div><div class="label">结构化平均</div></div>
      </div>
      <div class="metrics">
        ${metric('Token 变化率', fmtPct(summary.comparison.token_delta_pct), deltaClass(summary.comparison.token_delta_pct))}
        ${metric('字符数变化率', fmtPct(summary.comparison.char_delta_pct), deltaClass(summary.comparison.char_delta_pct))}
        ${metric('耗时变化率', fmtPct(summary.comparison.elapsed_delta_pct), deltaClass(summary.comparison.elapsed_delta_pct))}
      </div>`;
    }

    function resultTable(runs) {
      return `<div class="table-tools">
        <label>模式过滤<select id="modeFilter"><option value="all">全部</option><option value="text">纯文本 text</option><option value="structured">结构化 structured</option></select></label>
        <span>${runs.length} 条记录</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>任务</th><th>轮次</th><th>模式</th><th>消息数</th><th>Token</th><th>字符数</th><th>耗时 ms</th><th>记忆命中</th><th>状态传递</th><th>传递大小</th><th>State ID</th><th>Memory ID</th><th>消息链路</th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table></div>`;
    }

    function renderTableRows(runs, filter) {
      const tbody = document.querySelector('#rows');
      const filtered = runs.filter(run => filter === 'all' || run.mode === filter);
      tbody.innerHTML = filtered.map(run => {
        const m = run.metrics || {};
        const stateIds = (m.state_ref_ids || []).join(', ');
        const memoryIds = (m.memory_hit_ids || []).join(', ');
        return `<tr>
          <td>${escapeHtml(short(run.task || '', 120))}</td>
          <td>${fmt(run.round || 1)}</td>
          <td>${escapeHtml(modeLabel(run.mode || ''))}</td>
          <td>${fmt(m.message_count)}</td>
          <td>${fmt(m.approx_token_count)}</td>
          <td>${fmt(m.char_count)}</td>
          <td>${fmt(m.elapsed_ms)}</td>
          <td>${fmt(m.memory_hit_count)}</td>
          <td>${fmt(m.non_text_transfer_count)}</td>
          <td>${fmt(m.non_text_transfer_size)}</td>
          <td>${escapeHtml(short(stateIds, 180))}</td>
          <td>${escapeHtml(short(memoryIds, 180))}</td>
          <td><details><summary>展开链路</summary>${traceTable(m.message_trace || [])}</details></td>
        </tr>`;
      }).join('');
    }

    function traceTable(trace) {
      if (!trace.length) return '<pre>[]</pre>';
      return `<table><thead><tr><th>发送方</th><th>接收方</th><th>动作</th><th>引用 refs</th><th>状态 state</th><th>字符数</th><th>近似 Token</th></tr></thead>
        <tbody>${trace.map(item => `<tr>
          <td>${escapeHtml(item.from || '')}</td><td>${escapeHtml(item.to || '')}</td><td>${escapeHtml(item.action || '')}</td>
          <td>${escapeHtml(JSON.stringify(item.refs || []))}</td><td>${escapeHtml(JSON.stringify(item.state || {}))}</td>
          <td>${fmt(item.chars)}</td><td>${fmt(item.approx_tokens)}</td>
        </tr>`).join('')}</tbody></table>`;
    }

    function metric(label, value, klass = '') {
      return `<div class="metric"><div class="label">${escapeHtml(label)}</div><div class="value ${klass}">${escapeHtml(String(fmt(value)))}</div></div>`;
    }

    function barChart(title, textValue, structuredValue) {
      const max = Math.max(Number(textValue || 0), Number(structuredValue || 0), 1);
      return `<div class="chart"><div class="chart-title">${escapeHtml(title)}</div>
        ${barRow('纯文本', textValue, max, 'text')}
        ${barRow('结构化', structuredValue, max, 'structured')}
      </div>`;
    }

    function barRow(label, value, max, klass) {
      const width = Math.max(2, Math.round((Number(value || 0) / max) * 100));
      return `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill ${klass}" style="width:${width}%"></div></div><strong>${fmt(value)}</strong></div>`;
    }

    function pills(values, klass) {
      if (!values.length) return '<span class="pill">无</span>';
      return values.slice(0, 12).map(value => `<span class="pill ${klass}">${escapeHtml(value)}</span>`).join('');
    }

    function modeLabel(value) {
      if (value === 'text') return '纯文本 text';
      if (value === 'structured') return '结构化 structured';
      return value;
    }

    function uniqueIds(runs, key) {
      const ids = [];
      for (const run of runs) {
        for (const id of ((run.metrics || {})[key] || [])) if (!ids.includes(id)) ids.push(id);
      }
      return ids;
    }

    function traceCount(runs) {
      return runs.reduce((n, run) => n + (((run.metrics || {}).message_trace || []).length), 0);
    }

    function avg(metrics, key) {
      if (!metrics.length) return 0;
      return round(metrics.reduce((sum, item) => sum + Number(item[key] || 0), 0) / metrics.length);
    }

    function pct(newValue, oldValue) {
      const oldNum = Number(oldValue || 0);
      if (!oldNum) return null;
      return round((Number(newValue || 0) - oldNum) / oldNum * 100);
    }

    function deltaClass(value) {
      return value === null || value === undefined ? 'warn' : value <= 0 ? 'good' : 'bad';
    }

    function fmtPct(value) {
      return value === null || value === undefined ? 'n/a' : `${value}%`;
    }

    function fmt(value) {
      return value === null || value === undefined || value === '' ? '0' : value;
    }

    function round(value) {
      return Math.round(Number(value || 0) * 1000) / 1000;
    }

    function short(text, max = 300) {
      const value = String(text || '');
      return value.length > max ? `${value.slice(0, max)}...` : value;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    runBtn.addEventListener('click', runExperiment);
    loadBtn.addEventListener('click', loadLatest);
    clearBtn.addEventListener('click', clearResults);
    initFlow();
    loadConfig();
    loadLatest();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
