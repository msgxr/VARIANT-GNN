@echo off
REM ============================================================================
REM  VARIANT-GNN HIZLI BASLATICI  (V2.0)
REM  TEKNOFEST 2026 - Saglikta Yapay Zeka Yarismasi - Takim XYRA3
REM ----------------------------------------------------------------------------
REM  Bu dosya scripts\ altindadir; repo koku bir UST dizindir.
REM  Nereden cift tiklanirsa tiklanrsin once repo koküne gecer.
REM  Tüm metinler ASCII-guvenli (kod sayfasi farkli olsa da bozulmaz).
REM ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title VARIANT-GNN Baslatici  -  TEKNOFEST 2026 / XYRA3
color 0B

REM --- [0] Repo koküne gec (script konumunun bir ust dizini) ------------------
pushd "%~dp0.." 2>nul || (
  echo [HATA] Repo koküne gecilemedi: "%~dp0.."
  pause & exit /b 1
)
set "ROOT=%CD%"

REM --- [1] Python yorumlayici tespiti -----------------------------------------
REM  Oncelik: yerel sanal ortam (.venv) > py launcher > python
REM  (Ic ice parantez / & zinciri yerine duz goto: batch tuzaklarini onler)
set "PY="
set "PYSRC="
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
  set "PYSRC=.venv sanal ortam"
  goto :PYDONE
)
REM  Once DESTEKLENEN surumu tercih et (3.12 ^> 3.11 ^> 3.10) - py launcher ile.
REM  Proje 3.10-3.12 sabitler; daha yeni surumde bagimlilik kurulumu kirilir.
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "pass" >nul 2>&1
    if not errorlevel 1 (
      set "PY=py -%%V"
      set "PYSRC=py -%%V"
    )
  )
)
if defined PY goto :PYDONE
REM  Desteklenen surum yok: jenerik launcher / PATH python (surum sonra denetlenir).
where py >nul 2>&1
if not errorlevel 1 (
  set "PY=py -3"
  set "PYSRC=py -3 jenerik"
  goto :PYDONE
)
where python >nul 2>&1
if not errorlevel 1 (
  set "PY=python"
  set "PYSRC=python PATH"
  goto :PYDONE
)
:PYDONE
if not defined PY (
  echo.
  echo [HATA] Python bulunamadi.
  echo        Python 3.10 - 3.12 kurun: https://www.python.org/downloads/
  echo        Kurulumda "Add Python to PATH" secenegini isaretleyin.
  goto :SON
)

REM --- [2] Python surum kontrolu (3.10 - 3.12 destekli) -----------------------
%PY% -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<(3,13) else 1)" 2>nul
if errorlevel 1 (
  echo.
  echo [UYARI] Desteklenen Python araligi 3.10 - 3.12 disinda olabilir.
  echo         Devam edilebilir ama bagimlilik kurulumu hata verebilir.
  echo.
)

REM ============================================================================
REM  ANA MENU
REM ============================================================================
:MENU
cls
echo.
echo  ============================================================
echo    VARIANT-GNN  ^|  Missense Varyant Patojenite Tahmini
echo    TEKNOFEST 2026 - Saglikta Yapay Zeka  ^|  Takim XYRA3
echo  ============================================================
echo    Kok dizin : %ROOT%
echo    Python    : %PYSRC%
echo  ------------------------------------------------------------
echo.
echo    [1]  Hizli Baslat   - Web arayuzu (Streamlit, :8501)   ^<varsayilan^>
echo    [2]  Egitim         - Modeli sifirdan egit (train)
echo    [3]  Juri Demo      - Ornek varyantlar uzerinde tahmin (predict)
echo    [4]  REST API       - FastAPI servisi (uvicorn, :8000)
echo    [5]  Testler        - pytest test paketi
echo    [6]  Kurulum        - Bagimliliklari kur / guncelle
echo    [0]  Cikis
echo.
set "SEC="
set /p "SEC=  Seciminiz [1]: "
if not defined SEC set "SEC=1"

if "%SEC%"=="1" goto :WEB
if "%SEC%"=="2" goto :TRAIN
if "%SEC%"=="3" goto :PREDICT
if "%SEC%"=="4" goto :API
if "%SEC%"=="5" goto :TEST
if "%SEC%"=="6" goto :SETUP
if "%SEC%"=="0" goto :SON
echo   [UYARI] Gecersiz secim: "%SEC%"
timeout /t 2 >nul
goto :MENU

REM ============================================================================
REM  [1] WEB - Streamlit arayuzu
REM ============================================================================
:WEB
cls
echo  --- [1/3] Streamlit kontrol ediliyor... ---
%PY% -c "import streamlit" 2>nul
if errorlevel 1 (
  echo   Streamlit bulunamadi - kuruluyor (requirements-streamlit.txt)...
  call :PIP_INSTALL requirements-streamlit.txt || goto :HATA_DONUS
)

echo.
echo  --- [2/3] Egitilmis modeller kontrol ediliyor... ---
if not exist "models\xgb_model.json" (
  echo   [UYARI] models\xgb_model.json bulunamadi.
  set "ANS="
  set /p "ANS=  Once modeli egitmek ister misiniz? [E/h]: "
  if /i "!ANS!"=="E" call :DO_TRAIN
  if /i "!ANS!"=="" call :DO_TRAIN
) else (
  echo   Modeller mevcut - egitim atlaniyor.
)

echo.
echo  --- [3/3] Streamlit baslatiliyor... ---
echo   Tarayici otomatik acilmazsa: http://localhost:8501
echo   Durdurmak icin bu pencerede CTRL+C.
echo.
%PY% -m streamlit run app.py --server.port 8501 --server.address localhost
goto :ISLEM_BITTI

REM ============================================================================
REM  [2] TRAIN - Model egitimi
REM ============================================================================
:TRAIN
cls
echo  --- Model Egitimi (train) ---
echo   Bu islem models\ klasorundeki mevcut modelleri YENIDEN URETIR.
set "ANS="
set /p "ANS=  Devam edilsin mi? [e/H]: "
if /i not "%ANS%"=="E" (
  echo   Iptal edildi.
  goto :ISLEM_BITTI
)
call :ENSURE_CORE || goto :HATA_DONUS
call :DO_TRAIN
goto :ISLEM_BITTI

REM ============================================================================
REM  [3] PREDICT - Juri demo tahmini
REM ============================================================================
:PREDICT
cls
echo  --- Juri Demo Tahmini (predict) ---
if not exist "models\xgb_model.json" (
  echo   [HATA] Egitilmis model yok. Once [2] Egitim secenegini calistirin.
  goto :ISLEM_BITTI
)
if not exist "data\samples\jury_blind_sample.csv" (
  echo   [HATA] Ornek veri yok: data\samples\jury_blind_sample.csv
  goto :ISLEM_BITTI
)
call :ENSURE_CORE || goto :HATA_DONUS
if not exist "reports" mkdir "reports"
echo   Tahmin calistiriliyor...
%PY% main.py --mode predict ^
  --test_file data\samples\jury_blind_sample.csv ^
  --output reports\jury_demo_predictions.csv
if errorlevel 1 (
  echo   [HATA] Tahmin basarisiz oldu.
) else (
  echo.
  echo   [OK] Sonuc: reports\jury_demo_predictions.csv
)
goto :ISLEM_BITTI

REM ============================================================================
REM  [4] API - FastAPI / uvicorn
REM ============================================================================
:API
cls
echo  --- REST API (FastAPI + uvicorn) ---
%PY% -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
  echo   fastapi/uvicorn bulunamadi - kuruluyor...
  %PY% -m pip install fastapi uvicorn python-multipart --quiet || goto :HATA_DONUS
)
echo   Adres : http://127.0.0.1:8000
echo   Docs  : http://127.0.0.1:8000/docs
echo   Durdurmak icin CTRL+C.
echo.
%PY% -m uvicorn src.api.rest_api:app --host 127.0.0.1 --port 8000
goto :ISLEM_BITTI

REM ============================================================================
REM  [5] TEST - pytest
REM ============================================================================
:TEST
cls
echo  --- Test Paketi (pytest) ---
%PY% -c "import pytest" 2>nul
if errorlevel 1 (
  echo   pytest bulunamadi - kuruluyor...
  %PY% -m pip install -r requirements-dev.txt --quiet 2>nul || %PY% -m pip install pytest --quiet
)
%PY% -m pytest -q
goto :ISLEM_BITTI

REM ============================================================================
REM  [6] SETUP - Bagimlilik kurulumu
REM ============================================================================
:SETUP
cls
echo  --- Bagimliliklar kuruluyor / guncelleniyor ---
echo   1) pip yukseltiliyor...
%PY% -m pip install --upgrade pip --quiet
echo   2) Cekirdek bagimliliklar (requirements.txt)...
call :PIP_INSTALL requirements.txt || goto :HATA_DONUS
echo   3) Streamlit bilesenleri (requirements-streamlit.txt)...
call :PIP_INSTALL requirements-streamlit.txt || goto :HATA_DONUS
echo.
echo   [OK] Kurulum tamamlandi.
goto :ISLEM_BITTI

REM ============================================================================
REM  YARDIMCI ALT YORDAMLAR
REM ============================================================================

REM --- Cekirdek bagimliliklar yuklu mu? Degilse kur. -------------------------
:ENSURE_CORE
%PY% -c "import numpy, pandas, sklearn, xgboost" 2>nul
if errorlevel 1 (
  echo   Cekirdek bagimliliklar eksik - kuruluyor (requirements.txt)...
  call :PIP_INSTALL requirements.txt || exit /b 1
)
exit /b 0

REM --- requirements dosyasi kur (arg %1 = dosya adi) -------------------------
:PIP_INSTALL
if not exist "%~1" (
  echo   [HATA] Bulunamadi: %~1
  exit /b 1
)
%PY% -m pip install -r "%~1" --quiet
if errorlevel 1 (
  echo   [HATA] Kurulum basarisiz: %~1
  exit /b 1
)
exit /b 0

REM --- Egitimi calistir (dogru veri yolu + pdr config) ----------------------
:DO_TRAIN
set "TRAIN_CSV=data\synthetic\train_variants.csv"
if not exist "%TRAIN_CSV%" (
  echo   [HATA] Egitim verisi yok: %TRAIN_CSV%
  exit /b 1
)
echo   Egitim baslatiliyor (seed=42, config=configs\pdr.yaml)...
%PY% main.py --mode train --config configs\pdr.yaml --data_file "%TRAIN_CSV%"
if errorlevel 1 (
  echo   [HATA] Egitim basarisiz oldu.
  exit /b 1
)
echo   [OK] Egitim tamamlandi - modeller models\ klasorune yazildi.
exit /b 0

REM ============================================================================
REM  AKIS KONTROLU
REM ============================================================================
:HATA_DONUS
echo.
echo   [HATA] Islem bir hata ile durdu. Yukaridaki ciktiyi inceleyin.

:ISLEM_BITTI
echo.
echo  ------------------------------------------------------------
set "GERI="
set /p "GERI=  Menuye don [Enter] / Cikis [q]: "
if /i "%GERI%"=="q" goto :SON
goto :MENU

:SON
echo.
echo   VARIANT-GNN baslatici kapaniyor. Iyi calismalar.
popd
endlocal
exit /b 0
