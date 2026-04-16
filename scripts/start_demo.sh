#!/bin/bash
# DocWain Demo Startup Script
# Usage: bash scripts/start_demo.sh

set -e

echo "============================================"
echo "   Starting DocWain for Demo"
echo "============================================"

# 1. Start vLLM (DocWain-14B-v2)
echo ""
echo "[1/4] Starting vLLM (DocWain model)..."
sudo systemctl start docwain-vllm-fast
echo "  Waiting for model to load..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "  vLLM ready! ($(curl -s http://localhost:8100/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null))"
        break
    fi
    sleep 2
done

# 2. Start Main App
echo ""
echo "[2/4] Starting Main App..."
sudo systemctl start docwain-app
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  Main App ready!"
        break
    fi
    sleep 2
done

# 3. Start Standalone
echo ""
echo "[3/4] Starting Standalone API..."
sudo systemctl start docwain-standalone
sleep 3
echo "  Standalone: $(curl -s http://localhost:8400/health 2>/dev/null)"

# 4. Start Teams
echo ""
echo "[4/4] Starting Teams..."
sudo systemctl start docwain-teams
sleep 3
echo "  Teams: $(curl -s http://localhost:8300/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)"

# Status
echo ""
echo "============================================"
echo "   DocWain Demo Ready"
echo "============================================"
echo ""
echo "  Main App:    http://localhost:8000"
echo "  Standalone:  http://localhost:8400"
echo "  Teams:       http://localhost:8300"
echo "  vLLM:        http://localhost:8100"
echo ""
echo "  GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo ""
