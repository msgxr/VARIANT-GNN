"""
scripts/merge_yarisma_data.py
=============================
TEKNOFEST 2026 yarışma eğitim CSV'lerini (4 panel) tek bir dosyada birleştirir
ve şartname §3.2 panel yapısına uygun `Panel` kolonu ekler.

Girdi (data/raw/):
  - YARISMA_TRAIN_MASTER.csv   → Panel = General
  - YARISMA_TRAIN_KANSER.csv   → Panel = Hereditary_Cancer
  - YARISMA_TRAIN_PAH.csv      → Panel = PAH
  - YARISMA_TRAIN_CFTR.csv     → Panel = CFTR

Çıktı:
  - data/train_variants.csv    → birleştirilmiş eğitim seti (loader uyumlu)

Loader (src/data/loader.py) `Panel` kolonunu görürse otomatik
Panel_General / Panel_Hereditary_Cancer / Panel_PAH / Panel_CFTR
one-hot özellikleri ekler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
OUT_PATH  = REPO_ROOT / "data" / "train_variants.csv"

PANEL_MAP = {
    "YARISMA_TRAIN_MASTER.csv": "General",
    "YARISMA_TRAIN_KANSER.csv": "Hereditary_Cancer",
    "YARISMA_TRAIN_PAH.csv":    "PAH",
    "YARISMA_TRAIN_CFTR.csv":   "CFTR",
}


def main() -> int:
    if not RAW_DIR.exists():
        print(f"HATA: {RAW_DIR} bulunamadı.", file=sys.stderr)
        return 1

    frames: list[pd.DataFrame] = []
    for fname, panel in PANEL_MAP.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"HATA: {path} bulunamadı.", file=sys.stderr)
            return 1
        df = pd.read_csv(path)
        df["Panel"] = panel
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype("Int64")
        frames.append(df)
        print(f"  {fname:32s}  shape={df.shape}  Panel='{panel}'  Label dağılımı={df['Label'].value_counts().to_dict()}")

    merged = pd.concat(frames, ignore_index=True)

    cols = list(merged.columns)
    if "Variant_ID" in cols and "Panel" in cols:
        cols.remove("Panel")
        insert_at = cols.index("Variant_ID") + 1
        cols.insert(insert_at, "Panel")
        merged = merged[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print()
    print(f"Birleştirildi → {OUT_PATH}")
    print(f"Toplam satır: {len(merged):,}")
    print(f"Toplam kolon: {len(merged.columns)}  (Variant_ID + Panel + 351 AL_X + Label)")
    print(f"Panel dağılımı:")
    print(merged["Panel"].value_counts().to_string())
    print(f"\nSınıf dağılımı (Label):")
    print(merged["Label"].value_counts().to_string())
    print(f"\nPanel × Label çapraz tablosu:")
    print(pd.crosstab(merged["Panel"], merged["Label"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
