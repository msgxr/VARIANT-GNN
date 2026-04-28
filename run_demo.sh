#!/usr/bin/env bash
# =============================================================================
# run_demo.sh — VARIANT-GNN Jüri Demo Scripti
# TEKNOFEST 2026 | Takım XYRA3
#
# Kullanım:
#   chmod +x run_demo.sh
#   ./run_demo.sh
#
# Yapılanlar:
#   1. Ortam kontrolü (Python, Docker)
#   2. Docker Compose ile servisler ayağa kaldırılır
#   3. API sağlık kontrolü (hazır olana kadar bekler)
#   4. Örnek varyantlar API'ye gönderilir
#   5. Sonuçlar renkli terminalde gösterilir
#   6. PDF rapor oluşturulur (fpdf2 varsa)
# =============================================================================

set -euo pipefail

# ─── Renkler ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
MAGENTA='\033[0;35m'

API_URL="${API_URL:-http://localhost:8000}"
UI_URL="${UI_URL:-http://localhost:8501}"
MAX_WAIT=120   # saniye

# ─── Banner ──────────────────────────────────────────────────────────────────
print_banner() {
cat << 'EOF'
  _   _____    ____  _   ___    _  _ _____      ___  _  _ _  _
 | | | __\ \  / /  \| | | _ \_ _| \| |_   _|  / __|| \| | \| |
 |_| |_|  \ \/ /| |) |_| |   / _` |  ` | |   | (_ ||  ` |  ` |
 |__|___/   \_/ |___/|___|_|_\__,_|_|\_|_|    \___||_|\_|_|\_|

EOF
    echo -e "${CYAN}${BOLD}  TEKNOFEST 2026 | Sağlıkta Yapay Zeka | Takım XYRA3${RESET}"
    echo -e "${BLUE}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
}

log_step()  { echo -e "${CYAN}▶  ${BOLD}$1${RESET}"; }
log_ok()    { echo -e "${GREEN}✅  $1${RESET}"; }
log_warn()  { echo -e "${YELLOW}⚠️   $1${RESET}"; }
log_err()   { echo -e "${RED}❌  $1${RESET}"; }
log_info()  { echo -e "   ${BLUE}$1${RESET}"; }

# ─── Ortam Kontrolü ──────────────────────────────────────────────────────────
check_env() {
    log_step "Ortam kontrol ediliyor…"

    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        log_err "Python bulunamadı. Lütfen Python 3.10+ kurun."
        exit 1
    fi
    log_ok "Python mevcut"

    if command -v docker &>/dev/null && command -v docker compose &>/dev/null 2>&1; then
        DOCKER_AVAILABLE=true
        log_ok "Docker & Docker Compose mevcut"
    else
        DOCKER_AVAILABLE=false
        log_warn "Docker bulunamadı — yerel Python ile devam edilecek"
    fi
    echo ""
}

# ─── Docker Başlat ───────────────────────────────────────────────────────────
start_docker() {
    log_step "Docker servisleri başlatılıyor…"
    echo -e "   ${MAGENTA}(İlk çalıştırmada image build edilir, birkaç dakika sürebilir)${RESET}"
    echo ""

    docker compose up -d variant-gnn-api 2>&1 | \
        grep -E "Starting|Started|Created|Building|built" | \
        while read -r line; do echo -e "   ${BLUE}${line}${RESET}"; done

    log_ok "Docker servisi başlatıldı"
    echo ""
}

# ─── API Bekle ───────────────────────────────────────────────────────────────
wait_for_api() {
    log_step "API hazır olana kadar bekleniyor (max ${MAX_WAIT}s)…"
    local elapsed=0
    local dots=""

    while true; do
        if curl -sf "${API_URL}/health" &>/dev/null; then
            echo ""
            log_ok "API hazır! → ${API_URL}"
            break
        fi
        dots="${dots}."
        printf "\r   ${YELLOW}Bekleniyor${dots}${RESET}   "
        sleep 3
        elapsed=$((elapsed + 3))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo ""
            log_err "API ${MAX_WAIT}s içinde yanıt vermedi."
            log_info "Manuel kontrol: docker compose logs variant-gnn-api"
            exit 1
        fi
    done
    echo ""
}

# ─── Yerel API Başlat ────────────────────────────────────────────────────────
start_local_api() {
    log_step "Yerel FastAPI başlatılıyor (arka planda)…"
    PYTHON_BIN=$(command -v python3 || command -v python)

    if ! "$PYTHON_BIN" -c "import fastapi, uvicorn" 2>/dev/null; then
        log_info "fastapi ve uvicorn yükleniyor…"
        "$PYTHON_BIN" -m pip install fastapi uvicorn python-multipart -q
    fi

    nohup "$PYTHON_BIN" -m uvicorn src.api.rest_api:app \
        --host 0.0.0.0 --port 8000 --log-level error \
        > /tmp/variant_gnn_api.log 2>&1 &
    API_PID=$!
    log_info "API PID: ${API_PID} (log: /tmp/variant_gnn_api.log)"
    echo ""
}

# ─── Demo İstek ──────────────────────────────────────────────────────────────
send_demo_request() {
    log_step "Örnek varyant analizi gönderiliyor…"
    echo ""

    PAYLOAD='{
  "variants": [
    {
      "Variant_ID": "DEMO-BRCA1-c.5266dup",
      "Panel": "Hereditary_Cancer",
      "CADD_phred": 35.0,
      "REVEL_score": 0.95,
      "SIFT_score": 0.001,
      "PolyPhen2_HDIV_score": 0.998,
      "GERP_RS": 5.8,
      "PhyloP100way_vertebrate": 9.2,
      "gnomAD_exomes_AF": 0.000003,
      "In_Critical_Protein_Domain": 1.0,
      "MutPred2_score": 0.88
    },
    {
      "Variant_ID": "DEMO-CFTR-F508del",
      "Panel": "CFTR",
      "CADD_phred": 28.5,
      "REVEL_score": 0.72,
      "SIFT_score": 0.01,
      "PolyPhen2_HDIV_score": 0.92,
      "GERP_RS": 4.2,
      "gnomAD_exomes_AF": 0.0002,
      "In_Critical_Protein_Domain": 1.0
    },
    {
      "Variant_ID": "DEMO-BENIGN-rs1234567",
      "Panel": "General",
      "CADD_phred": 2.1,
      "REVEL_score": 0.05,
      "SIFT_score": 0.85,
      "PolyPhen2_HDIV_score": 0.02,
      "GERP_RS": -1.2,
      "gnomAD_exomes_AF": 0.15,
      "In_Critical_Protein_Domain": 0.0
    }
  ]
}'

    RESPONSE=$(curl -sf -X POST "${API_URL}/predict/json" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" 2>/dev/null || echo '{"status":"error"}')

    if echo "$RESPONSE" | grep -q '"status": *"success"'; then
        log_ok "Tahmin başarılı!"
        echo ""
        echo -e "   ${BOLD}┌─────────────────────────────────────────────────────────────┐${RESET}"
        echo -e "   ${BOLD}│  VARIANT-GNN — Klinik Tahmin Sonuçları                      │${RESET}"
        echo -e "   ${BOLD}└─────────────────────────────────────────────────────────────┘${RESET}"
        echo ""

        # Parse & display (basit grep ile)
        echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('results', [])
summary = data.get('summary', {})

colors = {
    'Pathogenic': '\033[0;31m',
    'Benign':     '\033[0;32m',
}
flag_colors = {
    'Uzman': '\033[1;33m',
    'Yüksek': '\033[0;32m',
    'Orta': '\033[0;36m',
}
RESET = '\033[0m'
BOLD  = '\033[1m'

for r in results:
    pred  = r['Prediction']
    vid   = r.get('Variant_ID', '?')
    risk  = r.get('Calibrated_Risk', 0)
    flag  = r.get('Clinical_Flag', '')
    prob  = r.get('Probability', 0)
    col   = colors.get(pred, '')
    fcol  = next((v for k,v in flag_colors.items() if k in flag), '')

    print(f'   {BOLD}{vid}{RESET}')
    print(f'     Tahmin    : {col}{BOLD}{pred}{RESET}')
    print(f'     Risk Skoru: {risk:.1f} / 100')
    print(f'     Olasılık  : {prob:.2%}')
    print(f'     Bayrak    : {fcol}{flag}{RESET}')
    print()

lat = data.get('latency_ms', 0)
print(f'   ⏱  Toplam Gecikme : {lat:.1f} ms ({lat/max(len(results),1):.1f} ms/varyant)')
print(f'   🔬 Human-in-Loop  : {summary.get(\"expert_review_required\", 0)}/{summary.get(\"total\", 0)} uzman yönlendirmesi')
" 2>/dev/null || echo "   (Python çıktı ayrıştırma hatası — ham yanıt: ${RESPONSE:0:300})"

    else
        log_warn "API yanıtı beklenmedik — model yüklü mü?"
        log_info "Önce: python main.py --mode train"
        log_info "Ham yanıt: ${RESPONSE:0:300}"
    fi
    echo ""
}

# ─── Özet ────────────────────────────────────────────────────────────────────
print_summary() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  VARIANT-GNN — Servis Adresleri${RESET}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
    echo -e "  ${GREEN}🌐  Streamlit Dashboard ${RESET}: ${CYAN}${UI_URL}${RESET}"
    echo -e "  ${GREEN}⚡  FastAPI REST API    ${RESET}: ${CYAN}${API_URL}${RESET}"
    echo -e "  ${GREEN}📖  API Dokümantasyonu  ${RESET}: ${CYAN}${API_URL}/docs${RESET}"
    echo -e "  ${GREEN}📊  ReDoc               ${RESET}: ${CYAN}${API_URL}/redoc${RESET}"
    echo ""
    echo -e "  ${YELLOW}Yük testi için:${RESET}"
    echo -e "  ${BLUE}locust -f locustfile.py --host ${API_URL} -u 100 -r 10${RESET}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

# ─── ANA AKIŞ ────────────────────────────────────────────────────────────────
main() {
    print_banner
    check_env

    if [ "${DOCKER_AVAILABLE}" = true ]; then
        start_docker
    else
        start_local_api
        sleep 5
    fi

    wait_for_api
    send_demo_request
    print_summary
}

main "$@"
