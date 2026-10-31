FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt rank_bm25==0.2.2

COPY . .

ENV PYTHONUNBUFFERED=1
ENV VECTOR_STORE_PATH=/data/vector_store
ENV LANGCHAIN_CACHE_PATH=/data/cache/langchain.db

EXPOSE 8501

# Build index at container start if missing, then launch Streamlit
CMD ["bash", "-lc", "python -m src.ingestion.build_index || true; streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
