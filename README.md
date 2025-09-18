PotatoSegmentation
Projekt do detekcji i segmentacji ziemniaków w wideo z u¿yciem OpenVINO (YOLO). Œledzi ziemniaki, kategoryzuje je na Small/Large i zapisuje obrazki z maskami.
Wymagania

Python 3.8+ (zalecany 3.10+)
Git
Modele OpenVINO (best.xml, best.bin) w assets/models/
Wideo (np. demo.mp4) w assets/videos/
Instalacja
1. Sklonuj repo:
git clone https://github.com/Nygus193Pro/Potato_Segmentation.git
cd PotatoSegmentation
2. Utwórz i aktywuj œrodowisko
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
3. Zainstaluj zale¿noœci
pip install -r requirements.txt
4. Uruchomienie
python start_processing.py

Okno 640x640 poka¿e detekcje (zielone kontury, czerwone centroidy, ID). Naciœnij 'q' by wyjœæ.
Ziemniaki s¹ zapisywane do Small/, Large/, Small_raw/, Large_raw/

Licencja
MIT License
