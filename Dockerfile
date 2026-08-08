# Kept for reference / container hosts (Cloud Run, Fly, a paid Docker Space).
# The live demo runs on Render as a native Python service (see render.yaml).
FROM python:3.12-slim

RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app

# bake the embedding model into the image so first request isn't a cold download
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
