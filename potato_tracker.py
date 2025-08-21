# import math
# import os
# import cv2
# import numpy as np
# from openvino.runtime import Core
#
# # ------------------- Potato -------------------
# class Potato:
#     def __init__(self, id, mask, centroid):
#         self.id = id
#         self.mask = mask
#         self.centroid = centroid
#         self.history = [centroid]
#         self.travelling_distance = 0
#         self.counted = False
#
#     def update_position(self, new_mask, new_centroid): # Aktualizuje ziemniaka nowymi danymi z bieżącej klatki:
#         self.mask = new_mask
#         self.centroid = new_centroid
#         self.history.append(new_centroid)
#         if len(self.history) > 1: # Jeśli mamy więcej niż 1 punkt, to możemy obliczyć dystans.
#             last = self.history[-2] # [-2] przedostatni punkt
#             dx = new_centroid[0] - last[0] # Odległośc w poziomie
#             dy = new_centroid[1] - last[1] # Odległość w pionie
#             self.travelling_distance += math.sqrt(dx ** 2 + dy ** 2)
#
#     def is_counted(self, threshold=200):
#         if not self.counted and self.travelling_distance > threshold: # prawdza, czy ziemniak przeszedł wystarczająco daleko i czy nie był już zapisany.
#             self.counted = True
#             return True
#         return False
#
# # ------------------- Potato_tracker -------------------
# class Potato_tracker:
#     def __init__(self, threshold=50):
#         self.tracker_potatoes = {} # Słownik: {id: Potato} – przechowuje wszystkie śledzone ziemniaki
#         self.next_id = 0
#         self.counting_threshold = threshold # Maksymalna odległość (pikseli), przy której uznamy, że dwa centroidy to ten sam ziemniak
#
#     def track(self, detection):
#         for det in detection:
#             mask, _, centroid = det
#             best_distance = self.counting_threshold #Przechowuje najlepsze dopasowanie (najmniejsza odległość, ale mniejsza niż threshold
#             fit_potato = None
#
#             for potato in self.tracker_potatoes.values(): # Dla każdego Potato, liczy odległość między jego ostatnim centroidem a aktualnym,
#                 distance = self.calculate_distance(centroid, potato.centroid)
#                 if distance < best_distance: # Jeśli dystans jest mniejszy niz poprzedni najlepszy, przypisujemy
#                     best_distance = distance # Zamiana zmiennych
#                     fit_potato = potato
#
#             if fit_potato: # Jeśli znalazłeś pasującego ziemniaka to update
#                 fit_potato.update_position(mask, centroid)
#             else: # Jeśli nie - nowy
#                 new_potato = Potato(self.next_id, mask, centroid)
#                 self.tracker_potatoes[self.next_id] = new_potato
#                 self.next_id += 1
#
#     def calculate_distance(self, c1, c2):
#         return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) # Oblicza odległość euklidesową między dwoma punktami (x1, y1) i (x2, y2),
#
# # ------------------- Foldery -------------------
# output_folder_small = 'Small'
# output_folder_large = 'Large'
# os.makedirs(output_folder_small, exist_ok=True)
# os.makedirs(output_folder_large, exist_ok=True)
#
# # ------------------- PotatoCategorizer -------------------
# class PotatoCategorizer:
#     def __init__(self, output_folder_small, output_folder_large, threshold=5000):
#         self.output_folder_small = output_folder_small
#         self.output_folder_large = output_folder_large
#         self.threshold = threshold
#
#     def categorize_potato(self, potato): # Kategoryzacja wielkości
#         if potato.size_pixels > self.threshold:
#             potato.category = "Large"
#         else:
#             potato.category = "Small"
#
#     def save_potato(self, potato, frame):
#         x, y, w, h = cv2.boundingRect(potato.mask) # Oblicza najmniejszy prostokąt, który otacza maskę (potato.mask)
#         cropped_frame = frame[y:y + h, x:x + w] # Wycinamy fragment z frame gdzie jest ziemniak
#         cropped_mask = potato.mask[y:y + h, x:x + w] # to samo, ale z maski ziemniaka (białe) (1)
#
#         if cropped_mask.dtype != np.uint8: # Sprawdzamy i zmieniamy format na uint8 0-255
#             cropped_mask = cropped_mask.astype(np.uint8)
#         if cropped_mask.max() <= 1:
#             cropped_mask *= 255
#
#         colored_mask = np.zeros_like(cropped_frame)
#         colored_mask[:, :, 0] = cropped_mask
#
#         alpha = 0.4
#         blended = cv2.addWeighted(cropped_frame, 1 - alpha, colored_mask, alpha, 0) # Nakładamy półprzezroczystą maskę na wycięty fragment klatki.
#
#         contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Szukanie i rysowanie konturów
#         cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)
#
#         folder = self.output_folder_small if potato.category == "Small" else self.output_folder_large
#         filename = f"potato_{potato.id}.png"
#         filepath = os.path.join(folder, filename)
#
#         cv2.imwrite(filepath, blended)
#
# # ------------------- ModelHandler (OpenVINO) -------------------
# class ModelHandler:
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.ie = None # Miejsce na obiekt OpenVINO Core()
#         self.compiled_model = None # Przechowuje gotowy, skompilowany mode (używane w predict
#
#     def load_model(self):
#         self.ie = Core() # worzy główny interfejs do OpenVINO Runtime
#         model_ir = self.ie.read_model(self.model_path) #Ładuje model z pliku .xml
#         self.compiled_model = self.ie.compile_model(model_ir, "CPU") # Kompiluje model do postaci zoptymalizowanej dla danego urządzenia
#
#
#     def predict(self, frame): # Robi predykcje i zwraca dane z segmentacji
#
#         original_h, original_w = frame.shape[:2]  # Pobiera 2 pierwsze liczby z obrazu czyli w,h
#         img_resized = cv2.resize(frame, (640, 640))  # resize obrazu
#         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Zmiana kolorów
#         img_float = img_rgb.astype(np.float32) / 255.0  # Normalizacja pikseli do 0.0 - 1.0 (Tego oczekuje model)
#         chw = np.transpose(img_float, (2, 0, 1))  # Zmiana kolejności osi tablicy (openVino chce CHW)
#         input_tensor = np.expand_dims(chw, axis=0)  # Dodanie nowego wymiaru na poczatku (openVino tak chce)
#
#         results = self.compiled_model(input_tensor) # # Uruchamiamy inferencje (czyli przewidywania) modelu, czyli po prostu go odpalamy, input_tensor to wejście o shape (1, 3, 640, 640), 1 obraz, RGB, 640x640, # [0] to najczęściej tablica z wynikami ( współrzędne, confidence, numer id) która jest przypisana pod results
#         predictions = np.squeeze(results[0]).T # Usuwa batch, bo juz nie potrzebny, .T zmienia osie, zmienia surowy wynik do wygodnej wersji postaci, gotowej do dalszej analizy
#         num_classes = 1  # Określa, że model rozpoznaje tylko jedną klasę
#         scores = np.max(predictions[:, 4:4 + num_classes],axis=1)  # 4:4 (kolumna z confidence), np.max wybiera największą wartość wśród predykcji,i zapisuje do scores
#         predictions = predictions[scores > 0.5,:] # Z tablicy predictions wybierz tylko te wiersze, które mają score > 0.5
#         prototypes = results[1][0]  # shape: (32, 160, 160)
#         prototypes = np.transpose(prototypes, (1, 2, 0))  # na (H, W, C) = (160, 160, 32)
#         box_predictions = predictions[..., :num_classes + 4]
#         # Bierzemy pierwsze num_classes + 4 kolumny
#         # → [x1, y1, x2, y2, score_klasy_0, ..., score_klasy_N]
#         # czyli współrzędne + confidence score'y dla wszystkich klas.
#         mask_predictions = predictions[..., num_classes + 4:]
#         # Bierzemy wszystko, co po num_classes + 4, czyli:
#         # → 111 wartości wektora maski (embedding dla segmentacji).
#
#         class_ids = np.argmax(box_predictions[:, 4:], axis=1)
#         # Dla każdej predykcji wybieramy klasę z najwyższym confidence score'em.
#         # (czyli: która klasa ma największe prawdopodobieństwo).
#
#         boxes = box_predictions[:, :4]
#         # Wyciągamy tylko kolumny 0, 1, 2, 3 – czyli:
#         # x1, y1, x2, y2 → współrzędne prostokąta (bounding box) dla każdej predykcji.
#
#         outputs = [] # Lista do której pójdą finalne wyniki
#         print(scores)
#         print(boxes)
#         for i in range(len(boxes)):
#             xc, yc, w, h = boxes[i]
#             x1 = int(xc-w/2) # Ryzyko float, więc zmiana na int dla pewności
#             y1 = int(yc-h/2)
#             w = int(w)
#             h = int(h)
#             embedding = mask_predictions[i]  # shape (33,)
#             bias = embedding[0]  # 1 liczba
#             mask_vector = embedding[1:]  # 32 liczby (jak liczba kanałów w prototype)
#
#             # mask = sigmoid(np.dot(prototypes, mask_vector) + bias)
#             mask = np.tensordot(prototypes, mask_vector, axes=([2], [0]))  # (160, 160)
#             mask = 1 / (1 + np.exp(-(mask + bias)))  # sigmoid + bias
#
#             # Resize to original image
#             mask = cv2.resize(mask, (original_w, original_h))
#             binary_mask = (mask > 0.5).astype(np.uint8) * 255
#             centroid = (int(xc), int(yc))
#
#             outputs.append((binary_mask, (x1, y1, w, h), centroid)) # Finalny wynik detekcji jako krotka (tuple)
#
#         return outputs
#
#
#     # def predict(self, frame):
#     #     original_h, original_w = frame.shape[:2]
#     #     img_resized = cv2.resize(frame, (640, 640))
#     #     img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
#     #     img_float = img_rgb.astype(np.float32) / 255.0
#     #     chw = np.transpose(img_float, (2, 0, 1))
#     #     input_tensor = np.expand_dims(chw, axis=0)
#     #
#     #
#     #     input_key = self.compiled_model.input(0).get_any_name()
#     #     results = self.compiled_model({input_key: input_tensor})
#     #     masks_tensor = list(results.values())[1]
#     #     boxes_tensor = list(results.values())[0]
#     #
#     #
#     #     masks = masks_tensor.squeeze(0)
#     #     boxes = boxes_tensor.squeeze(0)
#     #
#     #     outputs = []
#     #     for i in range(len(masks)):
#     #         box = boxes[i]
#     #         conf = box[4]
#     #         if conf < 0.5:  # <- TUTAJ FILTRUJEMY confidence
#     #             continue
#     #
#     #         mask = masks[i]
#     #         binary_mask = (mask > 0.6).astype(np.uint8) * 255
#     #         binary_mask_resized = cv2.resize(binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
#     #
#     #         coords = np.column_stack(np.where(binary_mask_resized > 0))
#     #         if coords.size == 0:
#     #             continue
#     #         centroid = coords.mean(axis=0)
#     #         centroid = (int(centroid[1]), int(centroid[0]))
#     #         outputs.append((binary_mask_resized, centroid))
#     #
#     #     return outputs
#
# # ------------------- MainProgram -------------------
# class MainProgram:
#     def __init__(self, model_path, video_path, categorizer_threshold):
#         self.model_path = model_path
#         self.video_path = video_path
#         self.categorizer_threshold = categorizer_threshold
#
#         self.model_handler = ModelHandler(self.model_path) # Tworzy obiekt do obsługi modelu OpenVINO.
#         self.tracker = Potato_tracker(threshold=100) # Tworzy tracker do śledzenia ziemniaków.
#         self.categorizer = PotatoCategorizer("Small", "Large", self.categorizer_threshold)
#         self.cap = cv2.VideoCapture(self.video_path) # Ładuje wideo do przetwarzania.
#         self.is_running = False # "Flaga logiczna"
#
#     def start_processing(self):
#         self.is_running = True
#         self.model_handler.load_model() # Wczytanie modelu
#
#         while self.is_running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#
#             detections = self.model_handler.predict(frame) # Przesyłamy klatkę do modelu segmentacyjnego, wykonanie predykcji
#
#             for mask, box, centroid in detections:
#                 x1, y1, w, h = box
#                 cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,255,0) ,2)
#                 cv2.circle(frame, centroid, 5, (0,0,255), -1)
#                 print(mask.shape)
#                #red_mask = cv2.merge([mask, np.zeros_like(mask), np.zeros_like(mask)])
#                 frame = cv2.addWeighted(frame, 1.0, mask, 0.3, 0)
#
#             self.tracker.track(detections)
#
#             for potato in self.tracker.tracker_potatoes.values(): # Iteracja po ziemniakach w słowniku tracker_potatoes
#                 if potato.is_counted(threshold=200): # def is_counted
#                     potato.size_pixels = np.sum(potato.mask > 0) # ile pikseli w masce należy do ziemniaka, np.sum liczy True czyli białe piksele (1)
#                     self.categorizer.categorize_potato(potato) # Klasyfikuje ziemniaka jako "Small" lub "Large".
#                     self.categorizer.save_potato(potato, frame) # Zapisuje obraz do small/large
#
#                     # Rysowanie konturów
#                     contours, _ = cv2.findContours(potato.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {potato.id}", potato.centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # Wypisuje ID
#
#             cv2.imshow("Detekcja ziemniaków", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         self.cap.release()
#         cv2.destroyAllWindows()


