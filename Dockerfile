FROM python:3.8.11-buster
RUN apt-get update \
  && apt-get install -y git \
  && rm -rf /var/lib/apt/lists/*
RUN pip install pip-tools
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
