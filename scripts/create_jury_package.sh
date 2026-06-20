#!/usr/bin/env bash
# scripts/create_jury_package.sh
# ──────────────────────────────────────────────────────────────────────────────
# Jüri teslim paketi oluşturur: kod + model ağırlıkları + requirements
# Kullanım: bash scripts/create_jury_package.sh
# Çıktı  : submission/VARIANT_GNN_jury_package_<tarih>.zip
# ──────────────────────────────────────────────────────────────────────────────
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATE=$(date +%Y%m%d)
PKG_NAME="VARIANT_GNN_jury_package_${DATE}"
PKG_DIR="${REPO}/submission/teknofest/${PKG_NAME}"
ZIP_OUT="${REPO}/submission/teknofest/${PKG_NAME}.zip"

echo "=== VARIANT-GNN Jüri Paketi Oluşturuluyor ==="
echo "Hedef: ${ZIP_OUT}"

# Temizle ve oluştur
rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}"

# ── 1. Kaynak kod ──────────────────────────────────────────────────────────────
echo "[1/5] Kaynak kod kopyalanıyor..."
cp -r "${REPO}/src"             "${PKG_DIR}/src"
cp -r "${REPO}/configs"         "${PKG_DIR}/configs"
cp -r "${REPO}/submission"      "${PKG_DIR}/submission"
cp    "${REPO}/main.py"         "${PKG_DIR}/main.py"
cp    "${REPO}/requirements.txt" "${PKG_DIR}/requirements.txt"
cp    "${REPO}/README.md"       "${PKG_DIR}/README.md"

# ── 2. Model ağırlıkları ───────────────────────────────────────────────────────
echo "[2/5] Model ağırlıkları kopyalanıyor..."
mkdir -p "${PKG_DIR}/models"
for f in preprocessor.pkl xgb_model.json ensemble.pkl meta_learner.pkl \
          calibrator.pkl ood_detector.pkl \
          autoencoder.pth gnn_model.pth dnn_model.pth \
          threshold.json panel_thresholds.json manifest.json \
          gnn_arch.json ensemble_config.json PROVENANCE.json README.md; do
    src="${REPO}/models/${f}"
    if [ -f "${src}" ]; then
        cp "${src}" "${PKG_DIR}/models/${f}"
        echo "  ✅ ${f}"
    else
        echo "  ⚠️  BULUNAMADI: ${f}"
    fi
done

# ── 3. Örnek jüri verisi ───────────────────────────────────────────────────────
echo "[3/5] Örnek veri kopyalanıyor..."
mkdir -p "${PKG_DIR}/data/samples"
[ -f "${REPO}/data/samples/jury_blind_sample.csv" ] && \
    cp "${REPO}/data/samples/jury_blind_sample.csv" "${PKG_DIR}/data/samples/"

# ── 4. Rapor figürleri ─────────────────────────────────────────────────────────
echo "[4/5] Rapor figürleri kopyalanıyor..."
mkdir -p "${PKG_DIR}/reports/figures/pdr"
cp -r "${REPO}/reports/figures/pdr/"* "${PKG_DIR}/reports/figures/pdr/" 2>/dev/null || true
# NOT: threshold_analysis / ablation_report / cross_panel_eval 2026-06-20'de pakete
# alınmıyor — leakage-öncesi/geri-çekilmiş (0.8980/0.9269) sayılar içeriyorlar (§IX).
# Güncel kanıtlar: ablasyon→ensemble_weight_justification.json, domain-shift→dann_lopo_validation.json.
for f in cv_report.json conformal_coverage_report.json benchmark_comparison.json \
          ensemble_weight_justification.json dann_lopo_validation.json \
          seed_stability.json; do
    [ -f "${REPO}/reports/${f}" ] && cp "${REPO}/reports/${f}" "${PKG_DIR}/reports/"
done

# ── 5. Hızlı test ─────────────────────────────────────────────────────────────
echo "[5/5] Predict.py syntax kontrolü..."
python3 -c "import ast; ast.parse(open('${PKG_DIR}/submission/predict.py').read())" && echo "  ✅ predict.py syntax OK"

# ── ZIP oluştur ────────────────────────────────────────────────────────────────
echo ""
echo "ZIP oluşturuluyor..."
cd "${REPO}/submission/teknofest"
zip -r "${ZIP_OUT}" "${PKG_NAME}/" -x "*.pyc" -x "*/__pycache__/*" -x "*.DS_Store"
rm -rf "${PKG_DIR}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ JÜRİ PAKETİ HAZIR"
echo "   Dosya : ${ZIP_OUT}"
echo "   Boyut : $(du -sh "${ZIP_OUT}" | cut -f1)"
echo ""
echo "Jüri çalıştırma komutu:"
echo "  pip install -r requirements.txt"
echo "  python submission/predict.py --input <jury_test.csv>"
echo "═══════════════════════════════════════════════════"
