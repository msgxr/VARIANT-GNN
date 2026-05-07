import os
from typing import Any, Dict, List

import pandas as pd
import vcfpy


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

        for record in reader:
            # Temel varyant bilgilerini çek
            # Not: VCF dosyasında çoklu ALT olabilir, biz ilkini alıyoruz.
            alt = record.ALT[0].value if record.ALT else "."
            
            variant_data = {
                "CHROM": str(record.CHROM),
                "POS": int(record.POS),
                "ID": record.ID[0] if record.ID else ".",
                "REF": str(record.REF),
                "ALT": str(alt),
                "QUAL": record.QUAL,
                "FILTER": record.FILTER[0] if record.FILTER else "PASS"
            }
            
            # INFO ve Genotype alanlarından ek özellikleri çek (opsiyonel)
            # Bu kısım modelin beklediği spesifik sütunlara göre genişletilebilir.
            for key, value in record.INFO.items():
                variant_data[f"INFO_{key}"] = value
                
            variants.append(variant_data)

        df = pd.DataFrame(variants)
        return df

    @staticmethod
    def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
        """VCF DataFrame'ini modelin beklediği standar sütun isimlerine hizalar."""
        # Mapping: CHROM -> Chr, POS -> Pos, etc.
        mapping = {
            "CHROM": "Chr",
            "POS": "Pos",
            "REF": "Ref",
            "ALT": "Alt"
        }
        df_renamed = df.rename(columns=mapping)
        return df_renamed
