import cv2
import numpy as np

# import klas i modułów z projektu potato_tracker
from potato_tracker import (
    Potato,             # klasa reprezentująca pojedynczego ziemniaka
    Potato_tracker,     # logika śledzenia ziemniaków w strumieniu
    PotatoCategorizer,  # Kategoryzacja (np. wielkość, defekty)
    ModelHandler,       # obsługa modelu (ładowanie, inferencja)
    MainProgram,        # główny program łączący wszystkie elementy
)

# utworzenie instancji głównego programu
program = MainProgram(
    r"D:\Pycharm\Projekt3KMK\runs\segment\train4\weights\best_openvino_model\best.xml",  # ścieżka do modelu OpenVINO IR
    "video_640.mp4",          # plik wideo wejściowy
    categorizer_threshold=10000,  # próg kategoryzacji (np. powierzchnia pikseli dla klasyfikacji)
    use_redis=False,          # czy korzystać z Redis
)

# uruchomienie przetwarzania wideo z użyciem modelu i logiki trackera
program.start_processing()
