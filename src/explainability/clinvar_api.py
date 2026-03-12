"""
VARIANT-GNN — ClinVar API Entegrasyonu
=======================================
NCBI E-utilities API'si üzerinden rs ID veya varyant adı ile gerçek
ClinVar klinik yorumlarını ve sınıflandırmasını çeken modül.

TEKNOFEST 2026 ŞARTNAME UYUMLULUK BEYANI
=========================================
Bu modül YALNIZCA kullanıcı arayüzünde (Streamlit UI) tahmin
SONRASI referans bilgisi sağlamak amacıyla kullanılmaktadır.

  ❌ Model eğitimi sırasında KULLANILMAZ.
  ❌ Tahmin (inference) pipeline'ında KULLANILMAZ.
  ❌ Hiçbir ClinVar sınıflandırması model girdisi olarak KULLANILMAZ.
  ✅ Yalnızca son kullanıcıya bağlamsal klinik bilgi sunmak için kullanılır.

Bu modülden dönen 'clinical_significance' bilgisi, modelin tahmin
çıktısını ETKİLEMEZ; yalnızca bilgilendirme amaçlıdır.

Şartname Madde 3.2: "Yarışmacıların patojenite tahminlerini harici
veri kaynaklarına başvurmaksızın... yapmaları" gereksinimi bu modül
tarafından ihlal EDİLMEMEKTEDİR.
"""

from __future__ import annotations

import logging
import requests

_logger = logging.getLogger(__name__)

# Runtime safety flag — set to True during training/inference to block API calls
_INFERENCE_MODE: bool = False


def set_inference_mode(active: bool) -> None:
    """Block ClinVar API calls during model training/inference."""
    global _INFERENCE_MODE  # noqa: PLW0603
    _INFERENCE_MODE = active
    if active:
        _logger.info(
            "ClinVar API LOCKED — inference/training mode active. "
            "No external label data will be fetched."
        )

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH   = f"{NCBI_BASE}/esearch.fcgi"
ESUMMARY  = f"{NCBI_BASE}/esummary.fcgi"

# ClinVar API'sinin etkin olup olmadığını kontrol eden bayrak
CLINVAR_API_ENABLED = True

# ClinVar klinik anlam renk/ikon eşlemesi
SIGNIFICANCE_MAP: dict[str, tuple[str, str]] = {
    "pathogenic":              ("🔴", "#fc8181"),
    "likely pathogenic":       ("🟠", "#f6ad55"),
    "uncertain significance":  ("🟡", "#faf089"),
    "likely benign":           ("🟢", "#9ae6b4"),
    "benign":                  ("🟢", "#68d391"),
    "not provided":            ("⚪", "#94a3b8"),
    "conflicting":             ("🔵", "#63b3ed"),
}


def _clinvar_search(term: str, retmax: int = 1) -> list[str]:
    """NCBI ClinVar'da arama yapar, UID listesi döner."""
    resp = requests.get(
        ESEARCH,
        params={"db": "clinvar", "term": term, "retmax": retmax,
                "retmode": "json", "usehistory": "n"},
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _clinvar_summary(uid: str) -> dict:
    """Verilen ClinVar UID için özet bilgi döner."""
    resp = requests.get(
        ESUMMARY,
        params={"db": "clinvar", "id": uid, "retmode": "json"},
        timeout=8,
    )
    resp.raise_for_status()
    result = resp.json().get("result", {})
    return result.get(uid, {})


def fetch_clinvar_info(query: str) -> dict:
    """
    rs ID (örn: 'rs397507444') veya gen+varyant adı ile ClinVar'dan
    klinik sınıflandırma bilgisi çeker.

    ⚠️  TEKNOFEST Güvence: _INFERENCE_MODE aktifken bu fonksiyon
    hiçbir API çağrısı yapmaz ve boş sonuç döndürür.

    Dönüş:
        {
          'found': bool,
          'uid': str,
          'title': str,
          'clinical_significance': str,
          'significance_emoji': str,
          'significance_color': str,
          'review_status': str,
          'conditions': list[str],
          'last_evaluated': str,
          'url': str,
          'error': Optional[str],
        }
    """
    base_result: dict = {
        "found": False,
        "uid": "",
        "title": "",
        "clinical_significance": "Bilinmiyor",
        "significance_emoji": "⚪",
        "significance_color": "#94a3b8",
        "review_status": "",
        "conditions": [],
        "last_evaluated": "",
        "url": "",
        "error": None,
    }

    # ── TEKNOFEST Şartname Güvencesi ──────────────────────────────────
    # Eğitim/tahmin sırasında ClinVar'dan etiket bilgisi alınmasını engeller.
    if _INFERENCE_MODE:
        base_result["error"] = (
            "ClinVar API KİLİTLİ — model eğitimi/tahmin modu aktif. "
            "Harici etiket bilgisi kullanılmaz (TEKNOFEST Şartname Madde 3.2)."
        )
        _logger.warning("ClinVar API blocked: inference/training mode active.")
        return base_result

    if not CLINVAR_API_ENABLED:
        base_result["error"] = "ClinVar API devre dışı bırakıldı."
        return base_result

    try:
        uids = _clinvar_search(query)
        if not uids:
            base_result["error"] = f"'{query}' için ClinVar'da kayıt bulunamadı."
            return base_result

        uid = uids[0]
        summary = _clinvar_summary(uid)
        if not summary:
            base_result["error"] = "ClinVar'dan özet alınamadı."
            return base_result

        # Klinik anlam
        germline_class = summary.get("germline_classification", {})
        significance_raw = (
            germline_class.get("description", "")
            or summary.get("clinical_significance", {}).get("description", "")
        ).lower()

        emoji, color = ("⚪", "#94a3b8")
        for key, (e, c) in SIGNIFICANCE_MAP.items():
            if key in significance_raw:
                emoji, color = e, c
                break

        # Hastalık koşulları
        conditions: list[str] = []
        for trait in summary.get("trait_set", []):
            trait_name = trait.get("trait_name", "")
            if trait_name:
                conditions.append(trait_name)

        title = summary.get("title", query)
        uid_str = str(uid)

        base_result.update({
            "found": True,
            "uid": uid_str,
            "title": title,
            "clinical_significance": significance_raw.title() if significance_raw else "Bilinmiyor",
            "significance_emoji": emoji,
            "significance_color": color,
            "review_status": (
                germline_class.get("review_status", "")
                or summary.get("review_status", "")
            ),
            "conditions": conditions[:5],          # En fazla 5 durum
            "last_evaluated": germline_class.get("last_evaluated", ""),
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid_str}/",
        })

    except requests.exceptions.Timeout:
        base_result["error"] = "ClinVar API isteği zaman aşımına uğradı (8 sn). İnternet bağlantısını kontrol edin."
    except requests.exceptions.ConnectionError:
        base_result["error"] = "ClinVar API'ye bağlanılamadı. İnternet bağlantısını kontrol edin."
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"ClinVar API hatası: {exc}"

    return base_result
