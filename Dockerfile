FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "ragqa", "serve", "--host", "0.0.0.0", "--port", "8000"]
