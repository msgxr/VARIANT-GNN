# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

import logging
import os
from typing import Any, Dict, List

import pandas as pd
import vcfpy

logger = logging.getLogger(__name__)


class VCFParser:
    """Endüstri standardı .vcf dosyalarını modele uygun veri setine dönüştüren sınıftır."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"VCF dosyası bulunamadı: {filepath}")

    def parse(self) -> pd.DataFrame:
        """VCF dosyasını okur ve temel biyolojik özellikleri içeren DataFrame döndürür."""
        variants: List[Dict[str, Any]] = []

        try:
            reader = vcfpy.Reader.from_path(self.filepath)
        except Exception as e:
            raise ValueError(f"VCF okuma hatası: {e}")

        malformed = 0
        for record in reader:
            # Bozuk tek kayıt TÜM dosyayı düşürmesin: kayıt-bazlı try/except ile atla.
            try:
                # Not: VCF dosyasında çoklu ALT olabilir, biz ilkini alıyoruz.
                alt = record.ALT[0].value if record.ALT else "."
                variant_data = {
                    "CHROM": str(record.CHROM),
                    "POS": int(record.POS),
                    "ID": record.ID[0] if record.ID else ".",
                    "REF": str(record.REF),
                    "ALT": str(alt),
                    # QUAL eksikse None → "." (yanıltıcı 0 yazma). FILTER boşsa "."
                    # (BİLİNMİYOR) — eskiden "PASS" yazılıyordu; boş FILTER PASS DEĞİLDİR.
                    "QUAL": record.QUAL if record.QUAL is not None else ".",
                    "FILTER": record.FILTER[0] if record.FILTER else ".",
                }
                # INFO alanlarından ek özellikleri çek (opsiyonel)
                for key, value in record.INFO.items():
                    variant_data[f"INFO_{key}"] = value
                variants.append(variant_data)
            except Exception as _rec_exc:
                malformed += 1
                logger.warning("VCF: bozuk kayıt atlandı (%s): %s", type(_rec_exc).__name__, _rec_exc)

        if malformed:
            logger.warning("VCF: toplam %d bozuk kayıt atlandı (%d geçerli).", malformed, len(variants))

        df = pd.DataFrame(variants)
        return df

    @staticmethod
    def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
        """VCF DataFrame'ini modelin beklediği standar sütun isimlerine hizalar."""
        # Mapping: CHROM -> Chr, POS -> Pos, etc.
        mapping = {"CHROM": "Chr", "POS": "Pos", "REF": "Ref", "ALT": "Alt"}
        df_renamed = df.rename(columns=mapping)
        return df_renamed
