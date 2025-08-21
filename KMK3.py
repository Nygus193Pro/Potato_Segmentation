import os
import cv2

video_path = "video_42.mp4"
output_folder = 'klatki2_640x640'  # nowy folder na przeskalowane klatki
os.makedirs(output_folder, exist_ok=True)

licznik = 0
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie udalo się otworzyc pliku wideo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Koniec filmu.")
        break

    frame_resized = cv2.resize(frame, (640, 640))

    filename = f"Klatka_{licznik:04d}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, frame_resized)

    licznik += 1

cap.release()
print(f"Zapisano {licznik} klatek w folderze '{output_folder}'.")
