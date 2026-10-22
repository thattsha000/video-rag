# VideoRAG API: FastAPI + LangChain + Chroma + sentence-transformers (CPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    CHROMA_DIR=/data/chroma

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding and reranker models into the image so the first request is not a download.
ARG EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

COPY videorag ./videorag
COPY scripts ./scripts
COPY data ./data

RUN useradd --create-home --uid 1000 app && mkdir -p /data/chroma && chown -R app /data /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["sh", "-c", "uvicorn videorag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
