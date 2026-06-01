.PHONY: install install-dev install-ci test test-smoke test-unit test-integration \
        lint typecheck format security results-check check clean-cache clean-artifacts \
        train predict eval crossval external-val adversarial-val explain \
        predict-jury external-val-full ablation pdr-report pdr-evidence \
        app api docker-build docker-run docker-up help

# Jüri için parametreler (override edebilirsin)
TEST_FILE  ?= data/samples/sample_predict.csv
TRAIN_FILE ?= data/samples/sample_train.csv
OUTPUT     ?= submission/predictions.csv
CONFIG     ?= configs/final.yaml

PYTHON := python
PYTEST := pytest
VENV := venv

# ─── Kurulum ────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install torch-scatter torch-sparse torch-geometric \
		-f https://data.pyg.org/whl/torch-2.2.0+cpu.html
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

install-ci:
	$(PYTHON) -m pip install -r requirements-ci.txt

# ─── Testler ────────────────────────────────────────────────────────────────

test: test-smoke test-unit test-integration

test-smoke:
	$(PYTEST) tests/smoke/ -v --tb=short

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

test-coverage:
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ─── Kod Kalitesi ────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ main.py app.py data_contracts/

lint-fix:
	ruff check --fix src/ tests/ main.py app.py data_contracts/

format:
	ruff format src/ tests/ main.py app.py

typecheck:
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ data_contracts/ main.py app.py -ll --skip B101,B301,B403

results-check:
	python scripts/check_results_consistency.py

check: lint typecheck security results-check

# ─── Model Operasyonları ─────────────────────────────────────────────────────

train:
	$(PYTHON) main.py --mode train

train-cftr:
	$(PYTHON) main.py --mode train --panel cftr --data_file data/train_cftr.csv

crossval:
	$(PYTHON) main.py --mode crossval --data_file data/samples/sample_train.csv

predict:
	$(PYTHON) main.py --mode predict --test_file data/samples/sample_predict.csv

# ─── JÜRİ MODU (tek komutla çalışır) ────────────────────────────────────────
predict-jury:
	@echo "═══════════════════════════════════════════"
	@echo "  VARIANT-GNN — JÜRİ TAHMİN MODU"
	@echo "  Giriş : $(TEST_FILE)"
	@echo "  Çıktı : $(OUTPUT)"
	@echo "═══════════════════════════════════════════"
	$(PYTHON) main.py --mode predict \
		--config $(CONFIG) \
		--test_file $(TEST_FILE) \
		--output $(OUTPUT)
	@echo "✅ Submission hazır → $(OUTPUT)"

# ─── EXTERNAL VALIDATION (tam metrik paketi) ──────────────────────────────────
external-val:
	$(PYTHON) main.py --mode external_val --test_file data/samples/sample_train.csv

external-val-full:
	@echo "═══════════════════════════════════════════"
	@echo "  VARIANT-GNN — EXTERNAL VALIDATION"
	@echo "  Test : $(TEST_FILE)"
	@echo "═══════════════════════════════════════════"
	$(PYTHON) main.py --mode external_val \
		--config $(CONFIG) \
		--test_file $(TEST_FILE)
	@echo "✅ Rapor → reports/external_validation_report.json"

eval:
	$(PYTHON) main.py --mode eval --data_file data/samples/sample_train.csv

# ─── ABLATION STUDY ──────────────────────────────────────────────────────────
ablation:
	@echo "═══════════════════════════════════════════"
	@echo "  VARIANT-GNN — ABLATION STUDY"
	@echo "═══════════════════════════════════════════"
	$(PYTHON) scripts/run_ablation.py --data_file $(TRAIN_FILE)
	@echo "✅ Rapor → reports/ablation_report.md"

# ─── PDR KANIT PAKETİ ────────────────────────────────────────────────────────
pdr-evidence:
	@echo "═══════════════════════════════════════════"
	@echo "  VARIANT-GNN — PDR KANIT PAKETİ"
	@echo "═══════════════════════════════════════════"
	$(PYTHON) scripts/build_pdr_evidence.py
	@echo "✅ PDR paketi → reports/pdr_evidence/"

pdr-report: pdr-evidence ablation
	@echo "✅ Tam PDR raporu hazırlandı"

adversarial-val:
	$(PYTHON) main.py --mode adversarial_val \
		--data_file data/samples/sample_train.csv \
		--test_file data/samples/sample_predict.csv

explain:
	$(PYTHON) main.py --mode explain --data_file data/samples/sample_train.csv

# ─── Web Arayüzü ─────────────────────────────────────────────────────────────

app:
	streamlit run app.py --server.port 8501

api:
	uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --reload

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker build -t variant-gnn:latest .
	docker build -f Dockerfile.api -t variant-gnn-api:latest .

docker-run:
	docker run -p 8501:8501 variant-gnn:latest

docker-up:
	docker-compose up -d
	@echo "✅ Streamlit: http://localhost:8501"
	@echo "✅ FastAPI  : http://localhost:8000"
	@echo "✅ API Docs : http://localhost:8000/docs"

# ─── Temizlik ────────────────────────────────────────────────────────────────

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

clean-artifacts:
	rm -f models/current/*.pth models/current/*.pkl models/current/*.json models/current/*.txt

# ─── Yardım ──────────────────────────────────────────────────────────────────

help:
	@echo "VARIANT-GNN Makefile Komutları"
	@echo "================================"
	@echo ""
	@echo "Kurulum:"
	@echo "  make install         CPU PyTorch + requirements.txt"
	@echo "  make install-dev     Geliştirme araçları dahil"
	@echo ""
	@echo "Test:"
	@echo "  make test            Tüm testler (smoke + unit + integration)"
	@echo "  make test-smoke      Sadece smoke testleri"
	@echo "  make test-unit       Sadece unit testleri"
	@echo ""
	@echo "Kod Kalitesi:"
	@echo "  make lint            Ruff ile lint kontrolü"
	@echo "  make typecheck       Mypy type check"
	@echo "  make security        Bandit güvenlik taraması"
	@echo "  make check           lint + typecheck + security"
	@echo ""
	@echo "Model:"
	@echo "  make train           Eğitim (genel panel)"
	@echo "  make crossval        Çapraz doğrulama (örnek veri)"
	@echo "  make predict         Tahmin (örnek veri)"
	@echo "  make explain         Açıklanabilirlik (örnek veri)"
	@echo ""
	@echo "UI:"
	@echo "  make app             Streamlit web arayüzü"
