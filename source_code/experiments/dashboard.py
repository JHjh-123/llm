from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from multi_agent.environment import collect_environment
from multi_agent.runner import ExperimentRunner
from multi_agent.tasks import DEFAULT_TASKS, TASK_GROUPS


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main() -> None:
    host = os.getenv("DASHBOARD_HOST", DEFAULT_HOST)
    port = int(os.getenv("DASHBOARD_PORT", str(DEFAULT_PORT)))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MultiAgentDashboard/1.0"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(_html())
            return
        if self.path == "/api/config":
            self._send_json(
                {
                    "tasks": DEFAULT_TASKS,
                    "task_groups": TASK_GROUPS,
                    "environment": collect_environment(),
                    "supported_modes": ["isolated", "shared"],
                }
            )
            return
        if self.path == "/api/sample":
            self._send_json(_load_sample_results())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return

        try:
            payload = self._read_json()
            task = str(payload.get("task") or DEFAULT_TASKS[0])
            memory_mode = str(payload.get("memory_mode") or "isolated")
            result = run_comparison(task=task, memory_mode=memory_mode)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        self._send_json({"ok": True, "result": result})

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


def run_comparison(task: str, memory_mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:10]

    if memory_mode == "shared":
        memory_path = _tmp_memory_path(run_id, "shared")
        with _temporary_env({"MEMORY_PATH": str(memory_path), "MEMORY_RESET": "1"}):
            runner = ExperimentRunner()
            text_run = runner.run_task(task, "text")
            structured_run = runner.run_task(task, "structured")
    elif memory_mode == "isolated":
        text_run = _run_isolated(task, "text", run_id)
        structured_run = _run_isolated(task, "structured", run_id)
    else:
        raise ValueError("memory_mode must be isolated or shared")

    result = {
        "run_id": run_id,
        "task": task,
        "memory_mode": memory_mode,
        "environment": collect_environment(),
        "text": text_run,
        "structured": structured_run,
        "comparison": _compare(text_run["metrics"], structured_run["metrics"]),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }

    output_path = Path("reports/dashboard_last.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _run_isolated(task: str, mode: str, run_id: str) -> dict[str, Any]:
    memory_path = _tmp_memory_path(run_id, mode)
    with _temporary_env({"MEMORY_PATH": str(memory_path), "MEMORY_RESET": "1"}):
        runner = ExperimentRunner()
        return runner.run_task(task, mode)


def _tmp_memory_path(run_id: str, suffix: str) -> Path:
    return Path(tempfile.gettempdir()) / f"llm_dashboard_{run_id}_{suffix}.sqlite"


def _compare(text: dict[str, Any], structured: dict[str, Any]) -> dict[str, Any]:
    text_tokens = float(text.get("approx_token_count", 0) or 0)
    structured_tokens = float(structured.get("approx_token_count", 0) or 0)
    text_chars = float(text.get("char_count", 0) or 0)
    structured_chars = float(structured.get("char_count", 0) or 0)
    return {
        "application_token_delta_pct": _pct_delta(structured_tokens, text_tokens),
        "application_char_delta_pct": _pct_delta(structured_chars, text_chars),
        "elapsed_delta_pct": _pct_delta(
            float(structured.get("elapsed_ms", 0) or 0),
            float(text.get("elapsed_ms", 0) or 0),
        ),
        "protocol_token_overhead": structured.get("protocol_approx_token_count", 0),
        "structured_total_with_protocol_tokens": structured.get("approx_token_count", 0)
        + structured.get("protocol_approx_token_count", 0),
        "text_total_with_protocol_tokens": text.get("approx_token_count", 0)
        + text.get("protocol_approx_token_count", 0),
        "memory_hit_delta": structured.get("memory_hit_count", 0) - text.get("memory_hit_count", 0),
        "non_text_transfer_delta": structured.get("non_text_transfer_count", 0)
        - text.get("non_text_transfer_count", 0),
    }


def _pct_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _load_sample_results() -> dict[str, Any]:
    for path in (Path("reports/demo_results.json"), Path("reports/results.json")):
        if path.is_file():
            return {"ok": True, "path": str(path), "result": json.loads(path.read_text(encoding="utf-8"))}
    return {"ok": False, "error": "No sample report found. Run an experiment first."}


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
  <title>Multi-Agent Communication Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #9a3412;
      --good: #166534;
      --bad: #b91c1c;
      --code: #101828;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      padding: 18px 24px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 28px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 190px 130px 130px;
      gap: 10px;
      align-items: end;
      margin-bottom: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 650;
    }
    select, button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
      padding: 0 10px;
    }
    button {
      cursor: pointer;
      font-weight: 700;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    button.primary:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .status {
      min-height: 22px;
      margin: 2px 0 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 82px;
    }
    .metric .name {
      font-size: 12px;
      color: var(--muted);
      font-weight: 650;
      margin-bottom: 8px;
    }
    .metric .value {
      font-size: 22px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .metric .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 15px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfd;
    }
    .final {
      padding: 12px 14px;
      color: var(--code);
      white-space: pre-wrap;
      line-height: 1.45;
      border-top: 1px solid var(--line);
      max-height: 230px;
      overflow: auto;
      font-size: 13px;
    }
    details {
      border-top: 1px solid var(--line);
      padding: 8px 14px 12px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 700;
      font-size: 13px;
    }
    pre {
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 6px;
      background: #111827;
      color: #e5e7eb;
      overflow: auto;
      max-height: 300px;
      font-size: 12px;
      line-height: 1.45;
    }
    .empty {
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
    }
    @media (max-width: 980px) {
      .toolbar, .metrics, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Multi-Agent Communication Dashboard</h1>
  </header>
  <main>
    <div class="toolbar">
      <label>任务
        <select id="task"></select>
      </label>
      <label>记忆口径
        <select id="memoryMode">
          <option value="isolated">isolated</option>
          <option value="shared">shared</option>
        </select>
      </label>
      <button class="primary" id="run">Run A/B</button>
      <button id="sample">Load Sample</button>
    </div>
    <div class="status" id="status"></div>
    <div id="output" class="empty">等待运行。需要配置真实 LLM 后端，例如 LLM_BACKEND=ollama。</div>
  </main>
  <script>
    const taskSelect = document.querySelector('#task');
    const memoryMode = document.querySelector('#memoryMode');
    const runButton = document.querySelector('#run');
    const sampleButton = document.querySelector('#sample');
    const statusBox = document.querySelector('#status');
    const output = document.querySelector('#output');

    const fmt = (value) => value === null || value === undefined ? 'n/a' : value;
    const clsDelta = (value) => value === null || value === undefined ? 'warn' : value <= 0 ? 'good' : 'bad';

    async function init() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      cfg.tasks.forEach((task, index) => {
        const option = document.createElement('option');
        option.value = task;
        option.textContent = `${index + 1}. ${task}`;
        taskSelect.appendChild(option);
      });
      statusBox.textContent = `后端: ${cfg.environment.llm_backend} / ${cfg.environment.llm_model}; Embedding: ${cfg.environment.embedding_backend} / ${cfg.environment.embedding_model}`;
    }

    async function runComparison() {
      setBusy(true, '运行真实 A/B 中，模型响应可能需要几十秒。');
      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({task: taskSelect.value, memory_mode: memoryMode.value})
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        renderLive(data.result);
        statusBox.textContent = `完成。结果已写入 reports/dashboard_last.json`;
      } catch (err) {
        output.className = 'empty';
        output.textContent = String(err.message || err);
        statusBox.textContent = '运行失败。检查 LLM_BACKEND、模型名和服务地址。';
      } finally {
        setBusy(false);
      }
    }

    async function loadSample() {
      setBusy(true, '加载已有报告。');
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        renderSample(data.result, data.path);
        statusBox.textContent = `已加载 ${data.path}`;
      } catch (err) {
        output.className = 'empty';
        output.textContent = String(err.message || err);
      } finally {
        setBusy(false);
      }
    }

    function setBusy(busy, text) {
      runButton.disabled = busy;
      sampleButton.disabled = busy;
      if (text) statusBox.textContent = text;
    }

    function renderLive(result) {
      const t = result.text.metrics;
      const s = result.structured.metrics;
      const c = result.comparison;
      output.className = '';
      output.innerHTML = `
        ${metricStrip([
          ['应用 token 差值', `${fmt(c.application_token_delta_pct)}%`, 'structured vs text', clsDelta(c.application_token_delta_pct)],
          ['应用字符差值', `${fmt(c.application_char_delta_pct)}%`, 'wire chars', clsDelta(c.application_char_delta_pct)],
          ['结构化协议开销', fmt(c.protocol_token_overhead), 'protocol tokens', 'warn'],
          ['结构化总 token', fmt(c.structured_total_with_protocol_tokens), 'application + protocol', ''],
          ['记忆命中差值', fmt(c.memory_hit_delta), 'structured - text', c.memory_hit_delta >= 0 ? 'good' : 'bad'],
          ['非文本传输差值', fmt(c.non_text_transfer_delta), 'structured - text', 'warn']
        ])}
        <div class="grid">
          ${runPanel('Text', result.text)}
          ${runPanel('Structured', result.structured)}
        </div>
      `;
    }

    function renderSample(result, path) {
      const summary = result.summary || {};
      const t = summary.text || {};
      const s = summary.structured || {};
      output.className = '';
      output.innerHTML = `
        ${metricStrip([
          ['报告', path, 'sample file', ''],
          ['结构化 token 差值', `${fmt(summary.structured_token_delta_pct)}%`, 'avg application tokens', clsDelta(summary.structured_token_delta_pct)],
          ['Text runs', fmt(t.runs), 'samples', ''],
          ['Structured runs', fmt(s.runs), 'samples', ''],
          ['Text avg token', fmt(t.avg_approx_tokens), 'application', ''],
          ['Structured avg token', fmt(s.avg_approx_tokens), 'application', '']
        ])}
        <section>
          <h2>Summary JSON</h2>
          <pre>${escapeHtml(JSON.stringify(summary, null, 2))}</pre>
        </section>
      `;
    }

    function metricStrip(items) {
      return `<div class="metrics">${items.map(([name, value, sub, klass]) => `
        <div class="metric">
          <div class="name">${escapeHtml(name)}</div>
          <div class="value ${klass || ''}">${escapeHtml(String(value))}</div>
          <div class="sub">${escapeHtml(sub || '')}</div>
        </div>
      `).join('')}</div>`;
    }

    function runPanel(title, run) {
      const m = run.metrics;
      return `
        <section>
          <h2>${title}</h2>
          <table>
            <tbody>
              ${row('Messages', m.message_count)}
              ${row('Chars', m.char_count)}
              ${row('Approx tokens', m.approx_token_count)}
              ${row('Elapsed ms', m.elapsed_ms)}
              ${row('Memory hits', m.memory_hit_count)}
              ${row('Non-text transfers', m.non_text_transfer_count)}
              ${row('Protocol events', m.protocol_event_count)}
              ${row('Protocol tokens', m.protocol_approx_token_count)}
            </tbody>
          </table>
          <h2>Trace</h2>
          <table>
            <thead><tr><th>From</th><th>To</th><th>Action</th><th>Refs</th><th>State</th><th>Tokens</th></tr></thead>
            <tbody>${(m.message_trace || []).map(traceRow).join('')}</tbody>
          </table>
          <div class="final">${escapeHtml(run.final || '')}</div>
          <details>
            <summary>Non-text transfers</summary>
            <pre>${escapeHtml(JSON.stringify(m.non_text_transfer_trace || [], null, 2))}</pre>
          </details>
          <details>
            <summary>Protocol messages</summary>
            <pre>${escapeHtml(JSON.stringify(run.protocol || [], null, 2))}</pre>
          </details>
        </section>`;
    }

    function row(name, value) {
      return `<tr><th>${escapeHtml(name)}</th><td>${escapeHtml(String(fmt(value)))}</td></tr>`;
    }

    function traceRow(item) {
      const state = item.state || {};
      return `<tr>
        <td>${escapeHtml(item.from || '')}</td>
        <td>${escapeHtml(item.to || '')}</td>
        <td>${escapeHtml(item.action || '')}</td>
        <td>${escapeHtml(String((item.refs || []).length))}</td>
        <td>${escapeHtml(Object.keys(state).join(', '))}</td>
        <td>${escapeHtml(String(item.approx_tokens || ''))}</td>
      </tr>`;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    runButton.addEventListener('click', runComparison);
    sampleButton.addEventListener('click', loadSample);
    init();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
