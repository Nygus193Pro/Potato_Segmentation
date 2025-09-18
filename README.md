Projekt PotatoSegmentation

Projekt wykrywa i segmentuje ziemniaki na wideo za pomocą OpenVINO (YOLO). Klasyfikuje ziemniaki na kategorie (Małe/Duże) i generuje maski. Wyobraźnia to podstawa!

Wymagania





Python 3.8+ (zalecany 3.10+)



Git



Model OpenVINO (best.xml, best.bin) w assets/models/Wideo (np. demo.mp4) w assets/videos/Instalacja

Instalacja





Sklonuj repozytorium:

git clone https://github.com/Nygus193Pro/Potato_Segmentation.git
cd PotatoSegmentation



Utwórz i aktywuj środowisko wirtualne:





Windows:

venv\Scripts\activate



Linux/macOS:

source venv/bin/activate 3



Zainstaluj zależności:

pip install -r requirements.txt



Uruchom skrypt:

python start_processing.py

Wyniki

Okno 640x640 pokazuje wykryte ziemniaki (obrys, centroid, ID). Naciśnij 'q', aby wyjść. Wyniki są zapisywane w:





Small/



Large/



Small_raw/



Large_raw/

Licencja

Licencja MIT
