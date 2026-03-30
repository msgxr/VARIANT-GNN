import json
import pandas as pd
from typing import List, Dict, Any, Optional

class FHIRParser:
    """T.C. Sağlık Bakanlığı ve uluslararası HL7/FHIR standartlarına sahip JSON verileri sisteme dahil eder."""

    def __init__(self, json_data: Dict[str, Any]):
        self.data = json_data

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(data)

    def parse_molecular_sequence(self) -> pd.DataFrame:
        """FHIR 'MolecularSequence' kaynağını modele uygun formata dönüştürür."""
        # FHIR yapısına göre varyantları çek
        variants: List[Dict[str, Any]] = []
        
        # ResourceType kontrolü
        if self.data.get("resourceType") != "MolecularSequence":
             # Eğer bir Bundle (toplu veri) ise içindeki entry'leri gez
             if self.data.get("resourceType") == "Bundle":
                 for entry in self.data.get("entry", []):
                     res = entry.get("resource", {})
                     if res.get("resourceType") == "MolecularSequence":
                         variants.extend(self._extract_from_res(res))
             else:
                 raise ValueError("Desteklenmeyen FHIR kaynağı. 'MolecularSequence' veya 'Bundle' bekleniyor.")
        else:
             variants.extend(self._extract_from_res(self.data))

        return pd.DataFrame(variants)

    def _extract_from_res(self, res: Dict[str, Any]) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        # FHIR R4 standardına göre varyantlar 'variant' alanı altındadır
        for var in res.get("variant", []):
            extracted.append({
                "Chr": var.get("referenceSeq", {}).get("referenceSeqId", {}).get("value", "."),
                "Pos": var.get("start", 0) + 1, # FHIR 0-indexed, biz 1-indexed kullanıyoruz
                "Ref": var.get("observedAllele", "."),
                "Alt": var.get("referenceAllele", "."),
                "FHIR_Score": var.get("score", 0)
            })
        return extracted
