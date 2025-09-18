PotatoSegmentation
Projekt do detekcji i segmentacji ziemniak�w w wideo z u�yciem OpenVINO (YOLO). �ledzi ziemniaki, kategoryzuje je na Small/Large i zapisuje obrazki z maskami.
Wymagania

Python 3.8+ (zalecany 3.10+)
Git
Modele OpenVINO (best.xml, best.bin) w assets/models/
Wideo (np. demo.mp4) w assets/videos/
Instalacja
1. Sklonuj repo:
git clone https://github.com/Nygus193Pro/Potato_Segmentation.git
cd PotatoSegmentation
2. Utw�rz i aktywuj �rodowisko
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
3. Zainstaluj zale�no�ci
pip install -r requirements.txt
4. Uruchomienie
python start_processing.py

Okno 640x640 poka�e detekcje (zielone kontury, czerwone centroidy, ID). Naci�nij 'q' by wyj��.
Ziemniaki s� zapisywane do Small/, Large/, Small_raw/, Large_raw/

Licencja
MIT License
