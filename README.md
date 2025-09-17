# Potato Segmentation — Demo + Python Sources

Prosty projekt testowy YOLO (segmentacja) na wideo demo.

## Wymagania
- Python 3.12
- Git
- (opcjonalnie) karta graficzna z CUDA

## Instalacja (Windows, PowerShell)
~~~powershell
git clone https://github.com/Nygus193Pro/Potato_Segmentation.git
cd Potato_Segmentation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
~~~

## Uruchomienie demo
~~~powershell
yolo predict model=yolo11n-seg.pt source=assets/videos/demo.mp4
~~~

## Uwaga
- Plik `assets/videos/demo.mp4` jest w repo i wymagany do testu.
- Foldery z danymi, wagami i wynikami (`runs/`, `*.pt`, `train/`, `valid/` itd.) są ignorowane przez `.gitignore`.
