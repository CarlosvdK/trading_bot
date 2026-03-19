#!/bin/bash
# =============================================================
# Trading Bot — Full System Launcher
# Starts: FastAPI backend + Next.js dashboard + Daily scanner
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BOT_DIR/logs"
PID_DIR="$BOT_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Trading Bot System Launcher${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ── Helper functions ──────────────────────────────────────────

stop_service() {
    local name=$1
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  Stopping $name (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

start_service() {
    local name=$1
    local cmd=$2
    local logfile="$LOG_DIR/$name.log"
    local pidfile="$PID_DIR/$name.pid"

    stop_service "$name"

    echo -e "  Starting ${YELLOW}$name${NC}..."
    cd "$BOT_DIR"
    nohup bash -c "$cmd" >> "$logfile" 2>&1 &
    local pid=$!
    echo "$pid" > "$pidfile"
    echo -e "  ${GREEN}✓${NC} $name started (PID $pid, log: $logfile)"
}

check_service() {
    local name=$1
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $name (PID $pid) — running"
            return 0
        fi
    fi
    echo -e "  ${RED}●${NC} $name — stopped"
    return 1
}

# ── Commands ──────────────────────────────────────────────────

case "${1:-start}" in
    start)
        echo "Starting all services..."
        echo ""

        # 1. FastAPI backend
        start_service "api" "cd $BOT_DIR && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

        # 2. Next.js dashboard
        start_service "dashboard" "cd $BOT_DIR/dashboard && npm run dev -- --port 3000"

        # 3. Daily scanner (runs pipeline scan every 6 hours during market hours)
        start_service "scanner" "cd $BOT_DIR && python scripts/run_scanner.py"

        echo ""
        sleep 2

        echo "Service status:"
        check_service "api"
        check_service "dashboard"
        check_service "scanner"

        echo ""
        echo -e "${GREEN}Dashboard:${NC}  http://localhost:3000"
        echo -e "${GREEN}API:${NC}        http://localhost:8000"
        echo -e "${GREEN}API Docs:${NC}   http://localhost:8000/docs"
        echo ""
        echo "Logs: $LOG_DIR/"
        echo ""
        echo "To stop:   $0 stop"
        echo "To status: $0 status"
        ;;

    stop)
        echo "Stopping all services..."
        stop_service "scanner"
        stop_service "dashboard"
        stop_service "api"
        echo -e "${GREEN}All services stopped.${NC}"
        ;;

    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;

    status)
        echo "Service status:"
        check_service "api"
        check_service "dashboard"
        check_service "scanner"
        ;;

    logs)
        local svc="${2:-api}"
        tail -f "$LOG_DIR/$svc.log"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]}"
        echo ""
        echo "Services: api, dashboard, scanner"
        exit 1
        ;;
esac
