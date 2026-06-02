"""src/cli/interactive.py — VARIANT-GNN interaktif oturum (claude benzeri).

`variant-gnn` komutu ARGUMANSIZ calistirildiginda acilan capraz-platform menu.
scripts/baslat.bat'in Windows-disi platformlarda da calisan, gercek bir global
komut olarak sunulan karsiligidir.

Tasarim ilkeleri
----------------
* Yalnizca standart kutuphane kullanir. Agir bagimliliklar (torch, xgboost,
  streamlit ...) menu ACILIRKEN degil, yalnizca ilgili islem secildiginde alt
  surec icinde devreye girer. Boylece menu, bagimliliklar henuz kurulmamis olsa
  bile acilir ([6] Kurulum bu yuzden menude durur).
* Tum islemler repo kokunde (REPO_ROOT) calistirilir; kullanici terminali
  hangi dizinde olursa olsun app.py / configs/ / data/ yollari bulunur.
* Konsola yazilan metinler ASCII-guvenlidir (Windows cp1254/cp437'de bozulmaz).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repo koku: bu dosya src/cli/interactive.py -> kok, iki ust dizin (parents[2]).
# Editable kurulumda (pip install -e . / pipx install -e .) gercek repoya cozulur,
# boylece app.py, configs/, data/ gibi repo-ici varliklar yerinde bulunur.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Aktif yorumlayicinin python'u (venv / pipx icindeki DOGRU surum).
PY = sys.executable

BANNER = (
    "============================================================\n"
    "  VARIANT-GNN  |  Missense Varyant Patojenite Tahmini\n"
    "  TEKNOFEST 2026 - Saglikta Yapay Zeka  |  Takim XYRA3\n"
    "============================================================"
)

MENU = (
    "  [1]  Hizli Baslat   - Web arayuzu (Streamlit, :8501)   <varsayilan>\n"
    "  [2]  Egitim         - Modeli sifirdan egit (train)\n"
    "  [3]  Juri Demo      - Ornek varyantlar uzerinde tahmin (predict)\n"
    "  [4]  REST API       - FastAPI servisi (uvicorn, :8000)\n"
    "  [5]  Testler        - pytest test paketi\n"
    "  [6]  Kurulum        - Bagimliliklari kur / guncelle\n"
    "  [0]  Cikis"
)


def _run(cmd: list[str], *, what: str) -> int:
    """Komutu repo kokunde calistir; Ctrl+C ve eksik program durumunu nazikce yakala."""
    print(f"\n>>> {what}")
    print(f">>> {' '.join(cmd)}\n")
    try:
        return subprocess.call(cmd, cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        print("\n[durduruldu]")
        return 130
    except FileNotFoundError as exc:
        print(f"[HATA] Calistirilamadi ({exc}). Once [6] Kurulum'u deneyin.")
        return 1


def _models_exist() -> bool:
    return (REPO_ROOT / "models" / "xgb_model.json").exists()


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _pause() -> None:
    try:
        input("\n  Menuye don [Enter] / Cikis [q]: ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)


def run_interactive() -> int:
    """Menu dongusu. Cikista 0 doner."""
    while True:
        print("\n" + BANNER)
        print(f"  Kok dizin : {REPO_ROOT}")
        print(f"  Python    : {sys.version.split()[0]}")
        print("-" * 60)
        print(MENU)
        try:
            sec = (input("\n  Seciminiz [1]: ").strip() or "1")
        except (EOFError, KeyboardInterrupt):
            print("\n  Cikiliyor.")
            return 0

        if sec == "0":
            print("\n  VARIANT-GNN kapaniyor. Iyi calismalar.")
            return 0

        if sec == "1":
            if not _models_exist():
                print("  [UYARI] Egitilmis model yok; once [2] Egitim onerilir.")
            _run(
                [PY, "-m", "streamlit", "run", "app.py",
                 "--server.port", "8501", "--server.address", "localhost"],
                what="Streamlit web arayuzu  ->  http://localhost:8501  (durdur: CTRL+C)",
            )
        elif sec == "2":
            if not _exists("data/synthetic/train_variants.csv"):
                print("  [HATA] Egitim verisi yok: data/synthetic/train_variants.csv")
            else:
                _run(
                    [PY, "main.py", "--mode", "train",
                     "--config", "configs/pdr.yaml",
                     "--data_file", "data/synthetic/train_variants.csv"],
                    what="Model egitimi (seed=42, configs/pdr.yaml)",
                )
        elif sec == "3":
            if not _models_exist():
                print("  [HATA] Egitilmis model yok. Once [2] Egitim calistirin.")
            elif not _exists("data/samples/jury_blind_sample.csv"):
                print("  [HATA] Ornek veri yok: data/samples/jury_blind_sample.csv")
            else:
                (REPO_ROOT / "reports").mkdir(exist_ok=True)
                _run(
                    [PY, "main.py", "--mode", "predict",
                     "--test_file", "data/samples/jury_blind_sample.csv",
                     "--output", "reports/jury_demo_predictions.csv"],
                    what="Juri demo tahmini  ->  reports/jury_demo_predictions.csv",
                )
        elif sec == "4":
            _run(
                [PY, "-m", "uvicorn", "src.api.rest_api:app",
                 "--host", "127.0.0.1", "--port", "8000"],
                what="REST API  ->  http://127.0.0.1:8000/docs  (durdur: CTRL+C)",
            )
        elif sec == "5":
            _run([PY, "-m", "pytest", "-q"], what="pytest test paketi")
        elif sec == "6":
            _run(
                [PY, "-m", "pip", "install", "-r", "requirements.txt"],
                what="Cekirdek bagimliliklar (requirements.txt)",
            )
        else:
            print(f"  [UYARI] Gecersiz secim: {sec!r}")
            continue

        _pause()
