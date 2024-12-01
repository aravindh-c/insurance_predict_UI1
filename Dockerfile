
FROM python:3.11-slim
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENTRYPOINT ["streamlit","run","app1.py","--server.address","0.0.0.0", "--server.port","80"]