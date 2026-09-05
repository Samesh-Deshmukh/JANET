#!/usr/bin/env bash
# Start/stop the models JANET keeps on the GPU, so you can have the card back.
#
#   tools/janet-gpu.sh status     what is running, and what is using the card
#   tools/janet-gpu.sh stop       free the VRAM (stops JANET too, if running)
#   tools/janet-gpu.sh start      bring the LLM back up and wait for it
#   tools/janet-gpu.sh restart    stop, then start
#
# WHY `kill` IS NOT ENOUGH, AND THIS SCRIPT EXISTS
# ------------------------------------------------
# llama-server holds ~9.7 GB of a 16 GB card, and it runs under a **user
# systemd unit** (~/.config/systemd/user/llama-qwen.service) with
# `Restart=always, RestartSec=5`. Kill the process and systemd restarts it five
# seconds later with a new PID — which looks exactly like "the VRAM never got
# freed". Measured: killing it dropped the card to 6.1 GB, and it was back at
# 15.5 GB before anyone noticed. The unit has to be stopped, not the process.
#
# It is a USER unit, so none of this needs sudo.
#
# WHAT IT WILL NOT TOUCH
# ----------------------
# Only JANET's own two things: the llama-qwen unit, and src/main.py under this
# repo's venv. `nvidia-smi` also lists your compositor, your streaming daemon
# and whatever game is open; a careless `pkill llama` or `pkill python` takes
# those with it. Every stop here is by unit name or by exact PID.
set -uo pipefail

UNIT="${JANET_LLM_UNIT:-llama-qwen.service}"
PORT="${JANET_LLAMA_PORT:-8081}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

janet_pids() { pgrep -f "$REPO/venv/bin/python.* main\.py" 2>/dev/null; }
gpu_line()   { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null; }

gpu_procs() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    while IFS=, read -r pid mem; do
        pid="${pid// /}"
        printf '    %-9s %-10s %s\n' "$pid" "${mem# }" "$(ps -p "$pid" -o comm= 2>/dev/null)"
    done
}

healthy() { curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

cmd_status() {
    echo "GPU: $(gpu_line)"
    echo "  everything using the card:"
    gpu_procs
    echo
    printf '  %-28s %s\n' "$UNIT" "$(systemctl --user is-active "$UNIT" 2>/dev/null)"
    local jp; jp="$(janet_pids | tr '\n' ' ')"
    printf '  %-28s %s\n' "JANET (src/main.py)" "${jp:-not running}"
    printf '  %-28s %s\n' "health on :$PORT" "$(healthy && echo ok || echo unreachable)"
    # The documented failure mode: two servers holding the same weights. Ollama
    # idling with no model loaded is harmless; Ollama with a model is 4 GB gone
    # and JANET's replies drop from 1-3s to 6-7s.
    if systemctl is-active --quiet ollama.service 2>/dev/null; then
        local loaded
        loaded="$(curl -s -m 3 localhost:11434/api/ps 2>/dev/null)"
        case "$loaded" in
            *'"models":[]'*) printf '  %-28s %s\n' "ollama" "running, no model loaded (harmless)" ;;
            "")              printf '  %-28s %s\n' "ollama" "running, not answering" ;;
            *)               printf '  %-28s %s\n' "ollama" "RUNNING WITH A MODEL — competing for VRAM" ;;
        esac
    fi
}

cmd_stop() {
    # JANET first: it talks to the LLM, so the other order leaves it briefly
    # answering from its canned offline fallback.
    local pids; pids="$(janet_pids | tr '\n' ' ')"
    if [ -n "${pids// /}" ]; then
        echo "  stopping JANET (pid $pids)"
        kill -TERM $pids 2>/dev/null; sleep 3
        kill -0 $pids 2>/dev/null && { kill -KILL $pids 2>/dev/null; sleep 1; }
    fi
    echo "  stopping $UNIT"
    systemctl --user stop "$UNIT" 2>/dev/null
    sleep 5   # Restart=always means give systemd a moment to NOT come back
    echo "  $UNIT is now $(systemctl --user is-active "$UNIT" 2>/dev/null)"
    echo "GPU: $(gpu_line)"
}

cmd_start() {
    echo "  starting $UNIT"
    systemctl --user start "$UNIT" 2>/dev/null || { echo "  could not start $UNIT"; return 1; }
    for _ in $(seq 1 60); do
        sleep 1
        healthy && { echo "  ready (weights loaded)"; echo "GPU: $(gpu_line)"; return 0; }
    done
    echo "  not healthy after 60s — journalctl --user -u $UNIT -n 40"
    return 1
}

case "${1:-status}" in
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    start)   cmd_start ;;
    restart) cmd_stop; cmd_start ;;
    *) echo "usage: $0 {status|start|stop|restart}"; exit 2 ;;
esac
