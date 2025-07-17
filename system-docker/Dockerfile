FROM python:3.13-slim

# 언어 및 TimeZone 설정
ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Seoul

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgirepository1.0-dev \
    gcc \
    pkg-config \
    python3-dev \
    gir1.2-gtk-3.0 \
    locales \
    wget \
    nano && \
    localedef -f UTF-8 -i ko_KR ko_KR.UTF-8 && \
    update-locale LANG=ko_KR.UTF-8 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel uv

RUN mkdir /SEARCHRIGHT
COPY requirements.txt /SEARCHRIGHT/requirements.txt
COPY backend/test-requirements.txt /SEARCHRIGHT/test-requirements.txt

RUN uv pip install -r /SEARCHRIGHT/requirements.txt --system
RUN uv pip install -r /SEARCHRIGHT/test-requirements.txt --system

# Copy application code
COPY backend/ /SEARCHRIGHT/
COPY system-docker/entrypoint.sh /SEARCHRIGHT/entrypoint.sh
COPY .env /SEARCHRIGHT/.env
RUN chmod +x /SEARCHRIGHT/entrypoint.sh

WORKDIR /SEARCHRIGHT

ENTRYPOINT ["/SEARCHRIGHT/entrypoint.sh"]
