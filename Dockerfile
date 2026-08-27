FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    CUHKX_DATASET_ROOT=/data \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY visualization/requirements.txt visualization/requirements.txt
COPY modeling/requirements.txt modeling/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY visualization visualization
COPY modeling modeling
COPY scripts scripts
COPY Training/class_mapping.csv Training/class_mapping.csv
COPY Testing/test.csv Testing/test.csv

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/outputs \
    && chown -R appuser:appuser /app \
    && chmod +x /app/scripts/docker-entrypoint.sh

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
# Kept as CMD so `docker run <image> bash` still works for debugging.
CMD ["streamlit", "run", "visualization/app.py"]
