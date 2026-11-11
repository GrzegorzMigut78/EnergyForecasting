# Obraz do uruchamiania obliczeń w chmurze lub na serwerze; identyczne wersje bibliotek
# jak lokalnie (uv.lock). Dane i wyniki montowane z zewnątrz:
#   docker build -t prognozowanie .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#       prognozowanie run configs/experiments/main.yaml --jobs 16
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "prognozowanie"]
CMD ["--help"]
