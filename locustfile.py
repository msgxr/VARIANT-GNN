"""
locustfile.py — VARIANT-GNN API Yük Testi
==========================================
Locust ile FastAPI endpoint'ine gerçekçi HTTP yük testi.

Kurulum:
    pip install locust

Çalıştırma (web arayüzlü):
    locust -f locustfile.py --host http://localhost:8000

Çalıştırma (headless, 100 kullanıcı, 60 saniye):
    locust -f locustfile.py --host http://localhost:8000 \
           --headless -u 100 -r 10 --run-time 60s \
           --csv reports/locust_results

Senaryolar:
  VariantUser  → /predict/json  ile 1-5 varyant gönderir
  HealthUser   → /health        ile heartbeat atar (lightweight)

Hedef (PDR kanıtı):
  "100 eş zamanlı doktor isteğini p95 < 200ms ile yanıtladık"
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict

from locust import HttpUser, between, task

# ─────────────────────────────────────────────────────────────────────────────
# Şablon varyant (43 özellik + Panel)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Ref_Nucleotide", "Alt_Nucleotide", "Codon_Change_Type",
    "AA_Grantham_Score", "GC_Content_Window", "In_CpG_Site",
    "Motif_Disruption_Score", "AA_Polarity_Change",
    "AA_Hydrophobicity_Diff", "AA_Mol_Weight_Diff", "AA_Size_Diff",
    "Protein_Impact_Score", "Delta_Solvent_Accessibility",
    "Secondary_Structure_Disruption", "GERP_RS",
    "PhyloP100way_vertebrate", "phastCons100way_vertebrate",
    "SiPhy_29way_logOdds", "Phylo_Diversity_Index",
    "gnomAD_exomes_AF", "gnomAD_exomes_AF_afr",
    "gnomAD_exomes_AF_eur", "gnomAD_exomes_AF_eas",
    "gnomAD_exomes_AF_sas", "gnomAD_exomes_AF_amr", "ExAC_AF",
    "SIFT_score", "PolyPhen2_HDIV_score", "PolyPhen2_HVAR_score",
    "CADD_phred", "REVEL_score", "MutPred2_score", "VEST4_score",
    "PROVEAN_score", "MutationTaster_score", "MetaSVM_score",
    "MetaLR_score", "MCAP_score", "In_Critical_Protein_Domain",
    "Splice_Site_Distance", "Is_Exonic", "Exon_Conservation_Ratio",
    "OMIM_Disease_Gene",
]

PANELS = ["General", "Hereditary_Cancer", "PAH", "CFTR"]


def _random_variant(vid: str) -> Dict[str, Any]:
    """Rastgele bir varyant kaydı üretir."""
    record: Dict[str, Any] = {
        "Variant_ID": vid,
        "Panel": random.choice(PANELS),
    }
    for feat in FEATURE_NAMES:
        record[feat] = round(random.uniform(0.0, 1.0), 6)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Ana kullanıcı — /predict/json
# ─────────────────────────────────────────────────────────────────────────────

class VariantUser(HttpUser):
    """
    Bir doktoru simüle eder.
    Rastgele 1-5 varyant gönderir, sonuçları alır.
    İstekler arası 1-3 saniye bekler (gerçekçi kullanım).
    """
    wait_time = between(1, 3)

    @task(8)
    def predict_single(self):
        """Tek varyant tahmini (en yaygın senaryo)."""
        payload = {"variants": [_random_variant("V_SINGLE")]}
        with self.client.post(
            "/predict/json",
            json=payload,
            catch_response=True,
            name="/predict/json [1 varyant]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    resp.success()
                else:
                    resp.failure(f"Beklenmeyen yanıt: {data}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(3)
    def predict_batch(self):
        """5 varyant toplu (panel analizi senaryosu)."""
        payload = {
            "variants": [
                _random_variant(f"V_BATCH_{i:03d}") for i in range(5)
            ]
        }
        with self.client.post(
            "/predict/json",
            json=payload,
            catch_response=True,
            name="/predict/json [5 varyant]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """Yük dengeleyici sağlık kontrolü."""
        self.client.get("/health", name="/health")

    @task(1)
    def get_info(self):
        """Model bilgisi sorgusu."""
        self.client.get("/info", name="/info")


# ─────────────────────────────────────────────────────────────────────────────
# Hafif kullanıcı — sadece health (arka plan izleme simülasyonu)
# ─────────────────────────────────────────────────────────────────────────────

class MonitorUser(HttpUser):
    """
    Hastane izleme sistemini simüle eder.
    Sadece /health endpoint'ini çok sık sorgular.
    """
    wait_time = between(0.5, 1.5)
    weight    = 1   # VariantUser'dan daha az örneklenir

    @task
    def heartbeat(self):
        self.client.get("/health", name="/health [monitor]")
