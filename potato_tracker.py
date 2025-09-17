import math
import os
import cv2
import numpy as np
from openvino import Core
from utilss import draw_detections
import redis
import time


# ===== Podstawowe funkcje pomocnicze =====

def xywh2xyxy(x):
    """Zamiana boxa z formatu (środek_x, środek_y, szerokość, wysokość)
       na format (x1, y1, x2, y2)."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    """Typowa funkcja aktywacji,ściska wartości w zakres 0–1."""
    return 1 / (1 + np.exp(-x))


def compute_iou(box, boxes):
    """Oblicza nakładanie się (IoU) jednego boxa z listą innych."""
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / union


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression: zostawia tylko najlepsze boxy, usuwa duplikaty."""
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


# ===== Obiekt pojedynczego ziemniaka =====

class Potato:
    def __init__(self, id, mask, centroid):
        self.id = id                  # unikalny numer
        self.mask = mask              # maska (kształt ziemniaka)
        self.centroid = centroid      # środek ziemniaka (x, y)
        self.history = [centroid]     # historia ruchu
        self.travelling_distance = 0  # ile przeszedł po ekranie
        self.counted = False          # czy już został policzony

    def update_position(self, new_mask, new_centroid):
        """Aktualizacja pozycji i liczenie przebytego dystansu."""
        self.mask = new_mask
        self.centroid = new_centroid
        self.history.append(new_centroid)
        if len(self.history) > 1:
            last = self.history[-2]
            dx = new_centroid[0] - last[0]
            dy = new_centroid[1] - last[1]
            self.travelling_distance += math.hypot(dx, dy)

    def is_counted(self, threshold=200):
        """Zwraca True tylko raz – gdy ziemniak przesunie się wystarczająco daleko."""
        if not self.counted and self.travelling_distance > threshold:
            self.counted = True
            return True
        return False


# ===== Bardzo prosty tracker na podstawie środka obiektu =====

class Potato_tracker:
    def __init__(self, threshold=50):
        self.tracker_potatoes = {}      # id -> Potato
        self.next_id = 0                # licznik nowych id
        self.counting_threshold = threshold

    def track(self, detection):
        """Dopasowanie nowych detekcji do starych obiektów na podstawie odległości centroidu."""
        keys_used = []
        for mask, centroid in detection:
            best_distance = self.counting_threshold
            fit_potato = None

            for key, potato in self.tracker_potatoes.items():
                distance = self.calculate_distance(centroid, potato.centroid)
                if distance < best_distance:
                    best_distance = distance
                    fit_potato = potato
                    keys_used.append(key)

            if fit_potato:
                fit_potato.update_position(mask, centroid)
            else:
                # nowy ziemniak
                keys_used.append(self.next_id)
                self.tracker_potatoes[self.next_id] = Potato(self.next_id, mask, centroid)
                self.next_id += 1

        # usuń ziemniaki, których nie ma w tej klatce
        keys_not_used = [k for k in list(self.tracker_potatoes.keys()) if k not in keys_used]
        for potato_id in keys_not_used:
            del self.tracker_potatoes[potato_id]

    def calculate_distance(self, c1, c2):
        """Odległość euklidesowa między dwoma punktami."""
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ===== Foldery wyjściowe na zapis obrazków =====

output_folder_small = 'Small'
output_folder_large = 'Large'
output_folder_large_raw = 'Large_raw'
output_folder_small_raw = 'Small_raw'
os.makedirs(output_folder_small, exist_ok=True)
os.makedirs(output_folder_large, exist_ok=True)
os.makedirs(output_folder_small_raw, exist_ok=True)
os.makedirs(output_folder_large_raw, exist_ok=True)


# ===== Klasa do kategoryzacji i zapisywania obrazków =====

class PotatoCategorizer:
    def __init__(self, output_folder_small, output_folder_large, output_folder_small_raw, output_folder_large_raw, threshold=2000):
        self.output_folder_small = output_folder_small
        self.output_folder_large = output_folder_large
        self.output_folder_small_raw = output_folder_small_raw
        self.output_folder_large_raw = output_folder_large_raw
        self.threshold = threshold  # granica pikseli – decyduje, czy ziemniak jest duży

    def categorize_potato(self, potato):
        """Nadaje etykietę Small albo Large w zależności od wielkości maski."""
        size = getattr(potato, "size_pixels", 0)
        potato.category = "Large" if size > self.threshold else "Small"

    def save_potato(self, potato, frame):
        """Wycinanie ziemniaka z maską i zapis do folderu (ładny obrazek z konturem)."""
        x, y, w, h = cv2.boundingRect(potato.mask)
        cropped_frame = frame[y:y + h, x:x + w]
        cropped_mask = potato.mask[y:y + h, x:x + w]

        # upewniamy się, że maska jest w formacie uint8 0–255
        if cropped_mask.dtype != np.uint8:
            cropped_mask = cropped_mask.astype(np.uint8)
        if cropped_mask.max() <= 1:
            cropped_mask = (cropped_mask * 255).astype(np.uint8)

        # nakładanie maski w kolorze niebieskim
        colored_mask = np.zeros_like(cropped_frame)
        colored_mask[:, :, 0] = cropped_mask
        blended = cv2.addWeighted(cropped_frame, 0.6, colored_mask, 0.4, 0)

        # rysowanie konturu
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)

        folder = self.output_folder_small if potato.category == "Small" else self.output_folder_large
        filepath = os.path.join(folder, f"potato_{potato.id}.png")
        cv2.imwrite(filepath, blended)

    def save_raw_potato(self, potato, frame):
        """Zapis surowego wycinka ziemniaka bez masek i efektów."""
        x, y, w, h = cv2.boundingRect(potato.mask)
        cropped_frame = frame[y:y + h, x:x + w]
        folder = self.output_folder_small_raw if potato.category == "Small" else self.output_folder_large_raw
        filepath = os.path.join(folder, f"potato_{potato.id}_raw.png")
        cv2.imwrite(filepath, cropped_frame)


# ===== Obsługa modelu OpenVINO =====

class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ie = None
        self.compiled_model = None

    def load_model(self):
        """Ładowanie i kompilacja modelu IR na CPU."""
        self.ie = Core()
        model_ir = self.ie.read_model(self.model_path)
        self.compiled_model = self.ie.compile_model(model_ir, "CPU")

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        """Skalowanie boxów z rozmiaru wejściowego (np. 640x640) na rozmiar obrazu (np. maski)."""
        in_arr = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, in_arr, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes

    def process_mask_output(self, mask_predictions, mask_output, boxes):
        """Tworzenie binarnych masek dla wykrytych ziemniaków."""
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)
        num_mask, mask_height, mask_width = mask_output.shape

        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        scale_boxes = self.rescale_boxes(boxes, (640, 640), (mask_height, mask_width))
        mask_maps = np.zeros((len(boxes), 640, 640), dtype=np.uint8)

        blur_size = (max(1, int(640 / mask_width)), max(1, int(640 / mask_height)))

        for i in range(len(scale_boxes)):
            sx1, sy1, sx2, sy2 = map(int, [scale_boxes[i][0], scale_boxes[i][1], scale_boxes[i][2], scale_boxes[i][3]])
            x1, y1, x2, y2 = map(int, [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

            scale_crop = masks[i][sy1:sy2, sx1:sx2]
            if scale_crop.size == 0 or x2 <= x1 or y2 <= y1:
                continue

            crop_mask = cv2.resize(scale_crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)

            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def predict(self, frame):
        """Pełna predykcja: przygotowanie obrazu, inferencja, NMS i maski."""
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_float, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        results = self.compiled_model(input_tensor)
        predictions = np.squeeze(results[0]).T

        num_classes = 1
        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        keep = scores > 0.5
        predictions = predictions[keep, :]
        scores = scores[keep]

        box_predictions = predictions[..., :num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 + 1:]

        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = xywh2xyxy(box_predictions[:, :4])
        boxes = np.clip(boxes, 0, 640)

        indices = nms(boxes, scores, 0.4)
        mask_predictions = mask_predictions[indices]
        mask_maps = self.process_mask_output(mask_predictions, results[1], boxes[indices])

        return boxes[indices], scores[indices], class_ids[indices], mask_maps


# ===== Główna pętla programu =====

class MainProgram:
    def __init__(self, model_path, video_path, categorizer_threshold, use_redis=True):
        self.model_path = model_path
        self.video_path = video_path
        self.categorizer_threshold = categorizer_threshold
        self.use_redis = use_redis

        if self.use_redis:
            self.redis_client = redis.Redis(host='localhost', port=6379)

        self.model_handler = ModelHandler(self.model_path)
        self.tracker = Potato_tracker(threshold=100)
        self.categorizer = PotatoCategorizer(
            output_folder_small="Small",
            output_folder_large="Large",
            output_folder_small_raw="Small_raw",
            output_folder_large_raw="Large_raw",
            threshold=self.categorizer_threshold
        )

        self.cap = cv2.VideoCapture(self.video_path)
        self.is_running = False

    def start_processing(self):
        """Główna pętla przetwarzania: odczyt klatek, detekcja, tracking i zapis/stream."""
        self.is_running = True
        self.model_handler.load_model()

        while True:
            # sprawdzanie czy program ma działać (Redis lub lokalnie)
            if self.use_redis:
                try:
                    cmd = self.redis_client.get('command')
                    if cmd:
                        cmd = cmd.decode('utf-8')
                        self.is_running = (cmd == 'start')
                    else:
                        self.is_running = False
                except Exception as e:
                    print(f"Błąd połączenia z Redis: {e}")
                    self.is_running = False
            else:
                self.is_running = True

            if not self.is_running:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            clean_frame = frame.copy()
            boxes, scores, class_ids, masks = self.model_handler.predict(frame)

            detection_data = []
            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                mask = masks[i].astype(np.uint8) * 255
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                detection_data.append((mask, centroid))

            self.tracker.track(detection_data)

            for potato in self.tracker.tracker_potatoes.values():
                potato.size_pixels = int(np.sum(potato.mask > 0))
                contours, _ = cv2.findContours(potato.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                cv2.circle(frame, potato.centroid, 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {potato.id}", potato.centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if potato.is_counted(threshold=500):
                    print("Potato counted")
                    self.categorizer.categorize_potato(potato)
                    self.categorizer.save_potato(potato, frame)
                    self.categorizer.save_raw_potato(potato, clean_frame)

            frame = draw_detections(frame, boxes, scores, class_ids, mask_maps=masks)
            _, buffer = cv2.imencode('.jpg', frame)

            if self.use_redis:
                try:
                    self.redis_client.publish('video_stream', buffer.tobytes())
                except Exception as e:
                    print(f"Błąd przy publikowaniu do Redis: {e}")
            else:
                cv2.imshow("Detekcja ziemniaków", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()
