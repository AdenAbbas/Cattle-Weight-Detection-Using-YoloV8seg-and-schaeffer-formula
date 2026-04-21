# ── Cowville — Hugging Face Spaces Docker image ──────────────────
# Targets HF Spaces "Docker" runtime. Exposes port 7860.

FROM python:3.10-slim

# System deps for OpenCV / Pillow / Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expects a non-root user named "user"
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Install Python deps
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=user . .

# HF Spaces port
EXPOSE 7860

# Gunicorn: single worker (ML model is memory-heavy), 2-minute timeout
CMD ["gunicorn", "-w", "1", "-t", "120", "-b", "0.0.0.0:7860", "app:app"]
