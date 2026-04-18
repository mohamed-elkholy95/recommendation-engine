#!/usr/bin/env bash
# End-to-end smoke: fit models (or reload from cache), start uvicorn, hit every
# endpoint over real HTTP, tear down. Exits non-zero on any failure.
#
# Usage:
#   PYTHON=/home/ai/miniforge3/envs/ai/bin/python ./scripts/e2e_test.sh
#
# Environment overrides (all optional):
#   PORT=8200             listen port
#   RECO_MODELS_DIR=/tmp/reco_models   cache dir for fitted models
#   PYTHON=<path>         python interpreter to use (default: python)

set -euo pipefail

PYTHON="${PYTHON:-python}"
PORT="${PORT:-8200}"
RECO_MODELS_DIR="${RECO_MODELS_DIR:-/tmp/reco_models_e2e}"
BASE_URL="http://127.0.0.1:${PORT}"
LOG="$(mktemp)"

cleanup() {
    local exit_code=$?
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ $exit_code -ne 0 ]]; then
        echo "=== server log ===" >&2
        tail -40 "${LOG}" >&2 || true
    fi
    rm -f "${LOG}"
    exit $exit_code
}
trap cleanup EXIT

check() {
    local name="$1"
    local expected_status="$2"
    shift 2
    local actual
    actual=$(curl -o /dev/null -s -w '%{http_code}' "$@") || actual="000"
    if [[ "${actual}" != "${expected_status}" ]]; then
        echo "FAIL ${name}: expected ${expected_status} got ${actual}" >&2
        exit 1
    fi
    echo "OK   ${name} (${actual})"
}

echo "[e2e] starting uvicorn on port ${PORT} (log: ${LOG})"
RECO_MODELS_DIR="${RECO_MODELS_DIR}" PORT="${PORT}" RECO_NCF_EPOCHS="${RECO_NCF_EPOCHS:-1}" \
    "${PYTHON}" -u scripts/serve.py > "${LOG}" 2>&1 &
SERVER_PID=$!

echo "[e2e] waiting for /health ..."
for _ in $(seq 1 60); do
    if curl -fsS "${BASE_URL}/health" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

check "GET /health"                        200 "${BASE_URL}/health"
check "POST /recommend (known user)"       200 -X POST "${BASE_URL}/recommend" \
        -H 'Content-Type: application/json' -d '{"user_id": 1, "n": 5}'
check "POST /recommend (unknown user)"     404 -X POST "${BASE_URL}/recommend" \
        -H 'Content-Type: application/json' -d '{"user_id": 99999, "n": 5}'
check "POST /recommend (bad n)"            422 -X POST "${BASE_URL}/recommend" \
        -H 'Content-Type: application/json' -d '{"user_id": 1, "n": 0}'
check "POST /rate (valid)"                 200 -X POST "${BASE_URL}/rate" \
        -H 'Content-Type: application/json' -d '{"user_id": 1, "movie_id": 1, "rating": 4.5}'
check "POST /rate (bad rating)"            422 -X POST "${BASE_URL}/rate" \
        -H 'Content-Type: application/json' -d '{"user_id": 1, "movie_id": 1, "rating": 9.0}'
check "POST /rate (unknown movie)"         404 -X POST "${BASE_URL}/rate" \
        -H 'Content-Type: application/json' -d '{"user_id": 1, "movie_id": 99999, "rating": 4.0}'

echo "[e2e] all checks passed"
