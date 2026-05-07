# =============================================================================
# VARIANT-GNN — TEKNOFEST 2026 Sağlıkta Yapay Zeka (Üniversite ve Üzeri)
#
# Reproducible çıkarım imajı (§7.5 jüri kod re-run uyumlu).
#   • Sabit Python sürümü (3.10)
#   • Sabit pip versiyon set'i (requirements.txt)
#   • Non-root user (security best-practice)
#   • HEALTHCHECK (Streamlit endpoint kontrolü)
# =============================================================================
FROM python:3.10-slim

LABEL maintainer="XYRA3 <iletisim@teknofest.org>"
LABEL competition="TEKNOFEST 2026 Sağlıkta Yapay Zeka — Üniversite ve Üzeri"
LABEL repository="https://github.com/msgxr/VARIANT-GNN"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ── Sistem bağımlılıkları ──────────────────────────────────────────────────
# build-essential : derleme için C/C++ toolchain
# git             : pip git+https kaynaklı paketler için
# curl            : HEALTHCHECK için
# libgomp1        : LightGBM/OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ── Python bağımlılıkları ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ── Uygulama kodu ───────────────────────────────────────────────────────────
COPY . .

# ── Jüri-hazır tahmin scripti ──────────────────────────────────────────────
COPY <<'EOF' /app/predict.sh
#!/bin/bash
# TEKNOFEST 2026 §7.2 jüri inference entrypoint.
# Kullanım: docker run -v /local/data:/data variant-gnn /app/predict.sh /data/test.csv
set -e
INPUT="${1:-data/test_variants_blind.csv}"
OUTPUT="${2:-submission/predictions.csv}"
CONFIG="${3:-configs/pdr.yaml}"
python submission/predict.py \
    --input  "${INPUT}" \
    --model_dir models \
    --output "${OUTPUT}" \
    --config "${CONFIG}"
EOF
RUN chmod +x /app/predict.sh

# ── Güvenlik: non-root kullanıcı ────────────────────────────────────────────
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Health check (Streamlit varsayılan endpoint) ───────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# ── Default entrypoint: Streamlit UI ────────────────────────────────────────
# Üretim modunda offline tahmin için override edin:
#   docker run --entrypoint /app/predict.sh variant-gnn data/test.csv
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
