#!/usr/bin/env bash
set -euo pipefail
nohup /bin/bash /home/pxf/llm/source_code/experiments/start_dashboard.sh \
  >/tmp/multi_agent_dashboard.log 2>/tmp/multi_agent_dashboard.err &
