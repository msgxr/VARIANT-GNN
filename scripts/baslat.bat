@echo off
echo ==========================================
echo   VARIANT-GNN HIZLI BASLATICI (V1.0)
echo ==========================================
echo.
echo [1/3] Bagimliliklar kontrol ediliyor...
python -m pip install -r requirements.txt --quiet

echo [2/3] Modeller kontrol ediliyor (Yoksa egitilecek)...
if not exist "models\xgb_model.json" (
    echo Model dosyasi bulunamadi. Ilk egitim baslatiliyor...
    python main.py --mode train
)

echo [3/3] Web Arayüzü (Streamlit) baslatiliyor...
echo Tarayici otomatik acilmazsa link: http://localhost:8501
streamlit run app.py --server.port 8501 --server.address localhost

pause
