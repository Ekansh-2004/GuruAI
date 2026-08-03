# GuruAI container — tuned for Hugging Face Spaces (Docker SDK), works anywhere.
#
# All persistent data (users, sessions, embedded document chunks) now lives in
# a Postgres database reached via DATABASE_URL — nothing is written to local
# disk at runtime except the pre-baked embedding-model cache below, which is
# already produced at build time. HF Spaces runs the container as UID 1000
# regardless, so we still keep the whole app under /home/user, that user's home.
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
