# Una imagen para los tres servicios (visión, agente y simulador): mismas dependencias de ejecución.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY sops ./sops
COPY models ./models
COPY data/download.py data/manifest.csv ./data/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/appdata /app/data/raw && chown -R app /app/appdata /app/data
USER app

EXPOSE 8000 8001
CMD ["uvicorn", "src.vision.api:app", "--host", "0.0.0.0", "--port", "8001"]
