FROM python:latest
WORKDIR /workspace
RUN apt-get update
COPY src /workspace
RUN pip install -r requirements.txt
VOLUME /workspace/data