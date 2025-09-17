# Potato Segmentation — Demo (Ultralytics, Twój model)

## Wymagania
- Python 3.12
- Git

## Instalacja (Windows, PowerShell)
```powershell
git clone https://github.com/Nygus193Pro/Potato_Segmentation.git
cd Potato_Segmentation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Uruchomienie: 
yolo predict model=assets/models/best.pt source=assets/videos/demo.mp4 show=True
Uwaga

Model jest w repo: assets/models/best.pt (nie trzeba pobierać osobno).

Wyniki zapisują się do runs/segment/predict.
