FROM debian:trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-server \
        python3 \
        python3-venv \
        tcpdump \
        tini \
    && python3 -m venv /opt/venv \
    && mkdir -p /run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install "transformers>=4.40.0"

COPY . .
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3456
EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "--", "/start.sh"]
CMD ["--config", "/app/config.json"]
