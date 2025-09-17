# Potato Segmentation — Demo (OpenVINO IR)

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
Uruchomienie (model ziemniaków — OpenVINO)
python -m ultralytics predict model=assets/models/best.xml source=assets/videos/demo.mp4 show=True
Uwagi

Model IR to para plików: assets/models/best.xml + assets/models/best.bin.

Wyniki zapisywane są do runs/segment/predict.
