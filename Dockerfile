
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/"


WORKDIR /app


COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY app/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
