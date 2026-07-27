# GuruAI container — tuned for Hugging Face Spaces (Docker SDK), works anywhere.
#
# HF Spaces runs the container as UID 1000, so everything the app writes at
# runtime (scholar.db, faiss_index_db/, the embedding-model cache) must live
# under a directory that user owns. We keep the whole app under /home/user.
FROM python:3.11-slim

# Create the non-root user HF Spaces expects.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install dependencies first so layer caching survives source edits.
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers embedding model into the image so the
# first request doesn't pay a ~90MB download. Matches embedder.py's model name.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

COPY --chown=user:user . .

# HF Spaces serves the container on 7860 (declared as app_port in the Space
# README). server.py reads PORT; keep the two in sync.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]
