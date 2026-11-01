FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KB_SOURCE_DIR=/kb/articles \
    KB_DATA_DIR=/app/data

WORKDIR /srv

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY config.py auth.py ingest.py server.py models_schema.py ./

RUN useradd --uid 1000 --create-home kb \
 && mkdir -p /app/data /kb/articles \
 && chown -R kb:kb /app/data /srv

USER kb
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "server.py"]
