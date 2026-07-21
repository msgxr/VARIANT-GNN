# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
src/ui/protein_viz.py — 3D protein görselleştirme (DEMO İSKELETİ).
NOT: Bu modül şu an hiçbir UI sekmesine bağlı DEĞİLDİR ve sabit/illüstratif bir
varyant bölgesi (resi 50-60) gösterir — gerçek per-varyant residue mapping yoktur.
"""

import streamlit as st

try:
    import py3Dmol
    from stmol import showmol

    _PY3DMOL_AVAILABLE = True
except ImportError:
    _PY3DMOL_AVAILABLE = False


def render_protein_3d(pdb_id: str = "1A2C", height: int = 500, width: int = 800) -> None:
    """Protein yapısını 3D olarak görselleştirir ve varyant bölgesini işaretler."""

    if not _PY3DMOL_AVAILABLE:
        st.warning("py3Dmol / stmol kurulu değil. `pip install py3Dmol stmol` ile yükleyin.")
        return

    st.markdown(f"### 🧬 Protein Yapı Analizi (PDB: {pdb_id})")

    # 3D Görselleştirme Hazırlığı
    xyzview = py3Dmol.view(query=f"pdb:{pdb_id}")

    # Stil Ayarları
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    xyzview.addSurface(py3Dmol.VDW, {"opacity": 0.3, "color": "white"})

    # ÖRNEK/PLACEHOLDER: sabit resi 50-60 yalnızca İLLÜSTRATİFTİR — gerçek varyantın
    # residue numarası DEĞİLDİR. Gerçek per-varyant mapping henüz bağlanmadı.
    xyzview.addStyle({"resi": "50-60"}, {"stick": {"colorscheme": "redCarbon"}})

    xyzview.zoomTo()
    xyzview.setBackgroundColor("#0e1117")

    # Streamlit üzerinde göster
    showmol(xyzview, height=height, width=width)

    st.info(
        "ℹ️ Kırmızı bölge yalnızca İLLÜSTRATİF bir örnektir (sabit resi 50-60); gerçek "
        "varyant residue konumu DEĞİLDİR. Bu görsel yalnızca demo amaçlıdır."
    )


import json  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402


def get_pdb_for_gene(gene_symbol: str) -> str:
    """
    UniProt REST API üzerinden Gen -> UniProt -> PDB eşleşmesini bulur.
    Gerçek zamanlı ve bilimsel doğrulukta bir eşleşme sağlar.
    """
    try:
        # 1. Gen sembolünden UniProt ID bul (Hata payını azaltmak için human proteini filtrele)
        query = urllib.parse.quote(f"gene:{gene_symbol} AND taxonomy_id:9606")
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=json&size=1"

        with urllib.request.urlopen(uniprot_url, timeout=5) as r:
            data = json.loads(r.read())
            results = data.get("results", [])
            if not results:
                return "1A2C"  # Fallback

            # 2. UniProt kaydından PDB referanslarını çek
            db_refs = results[0].get("uniProtKBCrossReferences", [])
            pdb_ids = [ref["id"] for ref in db_refs if ref["database"] == "PDB"]

            if pdb_ids:
                return str(pdb_ids[0])

    except Exception:
        # Bağlantı hatası veya API değişikliği durumunda güvenli fallback
        pass

    return "1A2C"  # Varsayılan yapısız model
