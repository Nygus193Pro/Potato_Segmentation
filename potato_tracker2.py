import math
import os
import cv2
import numpy as np
from openvino.runtime import Core
from utilss import draw_detections
import redis
import time


# Konwertuje bbox z formatu (xc, yc, w, h) → (x1, y1, x2, y2)
def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

# Własnoręczna implementacja Non-Maximum Suppression.

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


# ------------------- Potato -------------------
class Potato:
    def __init__(self, id, mask, centroid):
        self.id = id
        self.mask = mask
        self.centroid = centroid
        self.history = [centroid]
        self.travelling_distance = 0
        self.counted = False

    def update_position(self, new_mask, new_centroid):
        self.mask = new_mask
        self.centroid = new_centroid
        self.history.append(new_centroid)
        if len(self.history) > 1:
            last = self.history[-2]
            dx = new_centroid[0] - last[0]
            dy = new_centroid[1] - last[1]
            self.travelling_distance += math.sqrt(dx ** 2 + dy ** 2)

    def is_counted(self, threshold=200):
        if not self.counted and self.travelling_distance > threshold:
            self.counted = True
            return True
        return False

# ------------------- Potato_tracker -------------------
class Potato_tracker:
    def __init__(self, threshold=50):
        self.tracker_potatoes = {}
        self.next_id = 0
        self.counting_threshold = threshold

    def track(self, detection):
        keys_used = []
        for det in detection:
            mask, centroid = det
            best_distance = self.counting_threshold
            fit_potato = None

            for (key, potato) in self.tracker_potatoes.items():
                distance = self.calculate_distance(centroid, potato.centroid)
                if distance < best_distance:
                    best_distance = distance
                    fit_potato = potato
                    keys_used.append(key)

            if fit_potato:
                fit_potato.update_position(mask, centroid)
            else:
                keys_used.append(self.next_id)
                new_potato = Potato(self.next_id, mask, centroid)
                self.tracker_potatoes[self.next_id] = new_potato
                self.next_id += 1

        all_keys = self.tracker_potatoes.keys()
        keys_not_used = [key for key in list(all_keys) if key not in keys_used]
        for potato_id in keys_not_used:
            del self.tracker_potatoes[potato_id]

    def calculate_distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

# ------------------- Folders -------------------
output_folder_small = 'Small'
output_folder_large = 'Large'
output_folder_large_raw = 'Large_raw'
output_folder_small_raw = 'Small_raw'
os.makedirs(output_folder_small, exist_ok=True)
os.makedirs(output_folder_large, exist_ok=True)
os.makedirs(output_folder_small_raw, exist_ok=True)
os.makedirs(output_folder_large_raw, exist_ok=True)

# ------------------- PotatoCategorizer -------------------
class PotatoCategorizer:
    def __init__(self, output_folder_small, output_folder_large, output_folder_small_raw, output_folder_large_raw, threshold=2000):
        self.output_folder_small = output_folder_small
        self.output_folder_large = output_folder_large
        self.output_folder_small_raw = output_folder_small_raw
        self.output_folder_large_raw = output_folder_large_raw
        self.threshold = threshold

    def categorize_potato(self, potato):
        if potato.size_pixels > self.threshold:
            potato.category = "Large"
        else:
            potato.category = "Small"

    def save_potato(self, potato, frame):
        x, y, w, h = cv2.boundingRect(potato.mask)
        cropped_frame = frame[y:y + h, x:x + w]
        cropped_mask = potato.mask[y:y + h, x:x + w]

        if cropped_mask.dtype != np.uint8:
            cropped_mask = cropped_mask.astype(np.uint8)
        if cropped_mask.max() <= 1:
            cropped_mask *= 255

        colored_mask = np.zeros_like(cropped_frame)
        colored_mask[:, :, 0] = cropped_mask

        alpha = 0.4
        blended = cv2.addWeighted(cropped_frame, 1 - alpha, colored_mask, alpha, 0)

        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)

        folder = self.output_folder_small if potato.category == "Small" else self.output_folder_large
        filename = f"potato_{potato.id}.png"
        filepath = os.path.join(folder, filename)

        cv2.imwrite(filepath, blended)

    def save_raw_potato(self, potato, frame):
        x, y, w, h = cv2.boundingRect(potato.mask)
        cropped_frame = frame[y:y + h, x:x + w]

        folder = self.output_folder_small_raw if potato.category == "Small" else self.output_folder_large_raw
        filename = f"potato_{potato.id}_raw.png"
        filepath = os.path.join(folder, filename)

        cv2.imwrite(filepath, cropped_frame)

# ------------------- ModelHandler (OpenVINO) -------------------
class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ie = None
        self.compiled_model = None

    def load_model(self):
        self.ie = Core()
        model_ir = self.ie.read_model(self.model_path)
        self.compiled_model = self.ie.compile_model(model_ir, "CPU")

    # Skalowanie boxów
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes

    def process_mask_output(self, mask_predictions, mask_output, boxes):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        # Przekształcenie maski
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(boxes,
                                         (640, 640),
                                         (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(boxes), 640, 640))
        blur_size = (int(640 / mask_width), int(640 / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def predict(self, frame):
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_float, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        results = self.compiled_model(input_tensor)
        predictions = np.squeeze(results[0]).T
        num_classes = 1
        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        predictions = predictions[scores > 0.5, :]
        scores = scores[scores > 0.5]

        box_predictions = predictions[..., :num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 + 1:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = box_predictions[:, :4]
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, 640)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, 640)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, 640)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, 640)

        indices = nms(boxes, scores, 0.4)
        mask_predictions = mask_predictions[indices]
        mask_predictions = self.process_mask_output(mask_predictions, results[1], boxes[indices])

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions


# ------------------- MainProgram -------------------
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
        self.is_running = True
        self.model_handler.load_model()

        while True:
            if self.use_redis:
                try:
                    command = self.redis_client.get('command')
                    if command:
                        command = command.decode('utf-8')
                        if command == 'start':
                            self.is_running = True
                        elif command == 'stop':
                            self.is_running = False
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
                potato.size_pixels = np.sum(potato.mask > 0)
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