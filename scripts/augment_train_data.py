"""
scripts/augment_train_data.py
==============================
Eğitim verisine Gaussian noise tabanlı augmentation uygular.

Strateji:
  1. data/train_variants.csv'yi oku.
  2. Her satır için, sayısal kolonlara N(0, σ_col × scale) gürültü ekleyerek
     bir "noise-lu kopya" üret. Variant_ID'ye "_aug" suffix eklenir.
  3. Orijinal + augmented satırları birleştir → data/train_variants_aug.csv.

Bu sayede model küçük feature değişimlerine karşı daha robust olur.
Label ve Panel kolonlarına dokunulmaz; augmentation sadece feature uzayında.

NOT: Bu, SMOTE'a alternatif değil; SMOTE eğitim pipeline'ı içinde
zaten azınlık sınıfı dengeler. Bu augmentation **tüm sınıfların**
gerçek varyantlarına lokal gürültü ekleyerek **generalization gücünü** artırır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def augment(
    in_path: Path,
    out_path: Path,
    noise_scale: float = 0.05,
    n_copies: int = 1,
    seed: int = 42,
) -> None:
    df = pd.read_csv(in_path)
    print(f"Orijinal: shape={df.shape}  "
          f"label dist={df['Label'].value_counts().to_dict()}  "
          f"panel dist={df['Panel'].value_counts().to_dict() if 'Panel' in df.columns else 'n/a'}")

    metadata_cols = [c for c in ["Variant_ID", "Panel", "Label"] if c in df.columns]
    candidate_cols = [c for c in df.columns if c not in metadata_cols]
    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    feature_cols = [c for c in candidate_cols if numeric_df[c].notna().any()]
    string_only_cols = [c for c in candidate_cols if c not in feature_cols]
    print(f"Sayısal feature kolonu: {len(feature_cols)}  (string-only atlandı: {len(string_only_cols)})")
    if string_only_cols:
        print(f"  Atlanan kolonlar: {string_only_cols[:8]}{' ...' if len(string_only_cols)>8 else ''}")

    rng = np.random.default_rng(seed)

    col_std = numeric_df[feature_cols].std().fillna(0).values
    print(f"Ortalama feature σ: {np.nanmean(col_std):.4f}")

    augmented_frames = [df]
    feat_matrix_orig = numeric_df[feature_cols].values
    nan_mask = np.isnan(feat_matrix_orig)
    for k in range(n_copies):
        aug = df.copy(deep=True)
        noise = rng.normal(loc=0.0, scale=noise_scale * col_std, size=feat_matrix_orig.shape)
        aug_matrix = feat_matrix_orig + noise
        aug_matrix[nan_mask] = np.nan
        for i, col in enumerate(feature_cols):
            aug[col] = aug_matrix[:, i]
        if "Variant_ID" in aug.columns:
            aug["Variant_ID"] = aug["Variant_ID"].astype(str) + f"_aug{k+1}"
        augmented_frames.append(aug)

    merged = pd.concat(augmented_frames, ignore_index=True)
    if "Label" in merged.columns:
        merged["Label"] = merged["Label"].astype("Int64")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\nAugmented: shape={merged.shape}  "
          f"label dist={merged['Label'].value_counts().to_dict()}")
    print(f"Kaydedildi → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",  default=str(REPO_ROOT / "data" / "train_variants.csv"))
    parser.add_argument("--out_file", default=str(REPO_ROOT / "data" / "train_variants_aug.csv"))
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--n_copies",   type=int,   default=1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    augment(
        in_path=Path(args.in_file),
        out_path=Path(args.out_file),
        noise_scale=args.noise_scale,
        n_copies=args.n_copies,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
