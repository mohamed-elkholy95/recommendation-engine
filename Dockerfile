ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.5

# ---- builder ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG UV_VERSION
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN pip install "uv==${UV_VERSION}"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install runtime-only deps into a local .venv. Torch CPU wheel — the RTX 50
# (Blackwell sm_120) path needs the nightly cu128 index and is documented in
# the project README.
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache-dir \
         -e . \
         --extra-index-url https://download.pytorch.org/whl/cpu

# ---- runtime ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app pyproject.toml README.md ./

USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["python", "scripts/serve.py"]
