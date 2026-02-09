FROM python:3.12-alpine

# PyMuPDF needs some build deps on Alpine
RUN apk add --no-cache gcc musl-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apk del gcc musl-dev

COPY src/ src/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src"]
