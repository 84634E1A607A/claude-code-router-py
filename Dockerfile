FROM debian:trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install transformers>=4.40.0

COPY . .

EXPOSE 3456

ENTRYPOINT ["python3", "main.py"]
CMD ["--config", "/app/config.json"]
