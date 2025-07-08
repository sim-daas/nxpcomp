import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# --- Configuration Constants ---
# IMPORTANT: Update this path to your ONNX model file!
MODEL_PATH = "/home/user/cognipilot/shelf.onnx" # Changed from yolov8m-face.onnx as per your traceback
INPUT_WIDTH = 320
INPUT_HEIGHT = 320

# Minimum confidence score for a detection. Adjust as needed.
CONF_THRESHOLD = 0.25 
# IoU threshold for Non-Maximum Suppression. Adjust as needed.
NMS_THRESHOLD = 0.45 

# IMPORTANT: Define your class names here. 
# For yolov8s.onnx, this is typically the COCO dataset classes.
CLASS_NAMES = [
    "shelf"
]

# --- 1. Load the ONNX model ---
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except cv2.error as e:
    print(f"Error loading ONNX model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct and the ONNX file is valid.")
    exit()

# --- Set to CPU Backend and Target (Explicitly) ---
print("Using CPU backend and target for OpenCV DNN inference.")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT) # DNN_BACKEND_OPENCV also works for CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()
        self.latest_msg = None
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1
        )
        self.publisher = self.create_publisher(
            Image,
            '/detection/image_raw',
            1
        )
        self.prev_frame_time = time.time()
        self.fps = 0
        self.fps_sum = 0
        self.fps_count = 0

    def image_callback(self, msg):
        # FPS calculation (real callback rate)
        new_frame_time = time.time()
        self.fps = 1 / (new_frame_time - self.prev_frame_time) if (new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = new_frame_time
        self.fps_sum += self.fps
        self.fps_count += 1
        avg_fps = self.fps_sum / self.fps_count if self.fps_count > 0 else 0
        print(f"FPS: {self.fps:.2f} | Avg FPS: {avg_fps:.2f}")

        # Convert ROS Image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_height, image_width, _ = frame.shape

        # Preprocessing
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        output_layer_names = net.getUnconnectedOutLayersNames()
        output_data = net.forward(output_layer_names)
        preds = output_data[0].transpose((0, 2, 1))

        class_ids, confs, boxes = [], [], []
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT
        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            classes_score = row[4:]
            _, max_class_score, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id_in_slice = max_idx[1]
            if max_class_score > CONF_THRESHOLD:
                confs.append(float(max_class_score))
                class_ids.append(int(class_id_in_slice))
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])

        indexes = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
        annotated_frame = frame.copy()

        if len(indexes) > 0:
            for i in indexes.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                confidence = confs[i]
                class_id = class_ids[i]
                left = max(0, left)
                top = max(0, top)
                width = min(image_width - left, width)
                height = min(image_height - top, height)
                if width > 0 and height > 0:
                    cv2.rectangle(annotated_frame, (left, top), (left + width, top + height), (0, 255, 0), 1)
                    label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
                    label_x = left
                    label_y = top + height - 5
                    cv2.putText(annotated_frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Draw current and average FPS at top left
        fps_text = f"FPS: {int(self.fps)} | Avg: {avg_fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()