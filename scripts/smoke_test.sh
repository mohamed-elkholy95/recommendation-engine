#!/usr/bin/env bash
# Minimal smoke test — hits /health then /recommend against a running server.
# Usage: BASE_URL=http://localhost:8000 USER_ID=1 ./scripts/smoke_test.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
USER_ID="${USER_ID:-1}"
N="${N:-10}"

echo "[smoke] GET ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health"
echo

echo "[smoke] POST ${BASE_URL}/recommend user_id=${USER_ID} n=${N}"
curl -fsS -X POST "${BASE_URL}/recommend" \
    -H 'Content-Type: application/json' \
    -d "{\"user_id\": ${USER_ID}, \"n\": ${N}}"
echo
