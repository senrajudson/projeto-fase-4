FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências de build (ajuste se precisar de libs específicas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install --no-cache-dir poetry

# Copia só os manifests primeiro (melhor cache)
COPY pyproject.toml poetry.lock* /app/

# Instala deps no ambiente global do container (sem venv)
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-interaction --no-ansi --no-root

# Agora copia o código
COPY src/ /app/src/

# Se seu main.py é um script simples, use a linha abaixo:
CMD ["python", "src/main.py"]

# Se você preferir rodar como módulo (recomendado), use esta e garanta que existe src/__init__.py:
# CMD ["python", "-m", "src.main"]
