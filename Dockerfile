# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY hpfs-scheduler.py .
RUN pip install kubernetes
CMD ["python", "hpfs-scheduler.py"]
