import cv2
import numpy as np
import os
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray

MODEL_PATH = "/home/user/cognipilot/shelf.onnx"
BLURCOCO_MODEL_PATH = "/home/user/cognipilot/blurcoco.onnx"
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
CONF_THRESHOLD_SHELF = 0.75
CONF_THRESHOLD_BLURCOCO = 0.8
NMS_THRESHOLD = 0.45

CLASS_NAMES = ["shelf"]

BLURCOCO_CLASS_NAMES = [
    "1", "2", "3", "4", "5", "B", "CA", "CL", "CU", "H", "L", "PP", "TB", "Z"
]

BLURCOCO_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 165, 255),
    (128, 128, 0),
    (128, 0, 128),
    (0, 69, 255),
    (203, 192, 255),
    (255, 255, 255),
]

try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
except cv2.error as e:
    print(f"Error loading ONNX model: {e}")
    exit(1)

try:
    blurcoco_net = cv2.dnn.readNetFromONNX(BLURCOCO_MODEL_PATH)
except cv2.error as e:
    print(f"Error loading blurcoco ONNX model: {e}")
    exit(1)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
blurcoco_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
blurcoco_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Using CPU backend for inference.")

def shelf_angle_from_center_x(center_x, resolution):
    # The field of view is assumed to be 60 degrees (30 degrees to each side).
    # All calculations are in radians.
    fov_rad = math.radians(60)
    half_res = resolution / 2
    # The angle offset from the center (0 at center, negative left, positive right)
    angle = -math.degrees(math.atan((center_x - half_res) * math.tan(fov_rad / 2) / half_res))
    return angle

class ExtractorNode(Node):
    def __init__(self):
        super().__init__('extractor_node')
        self.bridge = CvBridge()
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
        self.shelf_info_pub = self.create_publisher(
            Float32MultiArray,
            '/shelf_det',
            1
        )
        self.object_dets_pub = self.create_publisher(
            Float32MultiArray,
            '/object_dets',
            1
        )
        self.prev_frame_time = time.time()
        self.fps = 0
        self.fps_sum = 0
        self.fps_count = 0

    def image_callback(self, msg):
        new_frame_time = time.time()
        self.fps = 1 / (new_frame_time - self.prev_frame_time) if (new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = new_frame_time
        self.fps_sum += self.fps
        self.fps_count += 1
        avg_fps = self.fps_sum / self.fps_count if self.fps_count > 0 else 0
        # Removed FPS printing

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_height, image_width, _ = frame.shape

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
            if max_class_score > CONF_THRESHOLD_SHELF:
                confs.append(float(max_class_score))
                class_ids.append(int(class_id_in_slice))
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])

        indexes = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD_SHELF, NMS_THRESHOLD)
        annotated_frame = frame.copy()
        object_dets = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                left = max(0, left)
                top = max(0, top)
                width = min(image_width - left, width)
                height = min(image_height - top, height)
                if width > 0 and height > 0:
                    cv2.rectangle(annotated_frame, (left, top), (left + width, top + height), (0, 255, 0), 1)
                    area_ratio = (width * height) / (image_width * image_height)
                    label = f"{CLASS_NAMES[0]}: {area_ratio:.3f}"
                    # Draw area ratio at right bottom of the shelf box
                    cv2.putText(
                        annotated_frame,
                        label,
                        (left + width, top + height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                    # Calculate shelf center x and angle
                    center_x = left + width / 2
                    angle = shelf_angle_from_center_x(center_x, image_width)

                    # Publish area ratio and angle
                    shelf_info_msg = Float32MultiArray()
                    shelf_info_msg.data = [area_ratio, angle]
                    self.shelf_info_pub.publish(shelf_info_msg)

                    crop = frame[top:top+height, left:left+width]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        crop_resized = cv2.resize(crop, (INPUT_WIDTH, INPUT_HEIGHT))
                        blob_blur = cv2.dnn.blobFromImage(crop_resized, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
                        blurcoco_net.setInput(blob_blur)
                        blurcoco_output_layer_names = blurcoco_net.getUnconnectedOutLayersNames()
                        blurcoco_output_data = blurcoco_net.forward(blurcoco_output_layer_names)
                        blurcoco_preds = blurcoco_output_data[0].transpose((0, 2, 1))

                        b_class_ids, b_confs, b_boxes = [], [], []
                        b_x_factor = width / INPUT_WIDTH
                        b_y_factor = height / INPUT_HEIGHT
                        b_rows = blurcoco_preds[0].shape[0]
                        for j in range(b_rows):
                            row_b = blurcoco_preds[0][j]
                            b_classes_score = row_b[4:]
                            _, b_max_class_score, _, b_max_idx = cv2.minMaxLoc(b_classes_score)
                            b_class_id_in_slice = b_max_idx[1]
                            if b_max_class_score > CONF_THRESHOLD_BLURCOCO:
                                obj_name = BLURCOCO_CLASS_NAMES[b_class_id_in_slice] if b_class_id_in_slice < len(BLURCOCO_CLASS_NAMES) else str(b_class_id_in_slice)
                                b_confs.append(float(b_max_class_score))
                                b_class_ids.append(int(b_class_id_in_slice))
                                bx, by, bw, bh = row_b[0].item(), row_b[1].item(), row_b[2].item(), row_b[3].item()
                                bleft = int((bx - 0.5 * bw) * b_x_factor) + left
                                btop = int((by - 0.5 * bh) * b_y_factor) + top
                                bwidth = int(bw * b_x_factor)
                                bheight = int(bh * b_y_factor)
                                b_boxes.append([bleft, btop, bwidth, bheight])
                        b_indexes = cv2.dnn.NMSBoxes(b_boxes, b_confs, CONF_THRESHOLD_BLURCOCO, NMS_THRESHOLD)
                        if len(b_indexes) > 0:
                            for k in b_indexes.flatten():
                                bbox = b_boxes[k]
                                bleft, btop, bwidth, bheight = bbox[0], bbox[1], bbox[2], bbox[3]
                                bleft = max(0, bleft)
                                btop = max(0, btop)
                                bwidth = min(image_width - bleft, bwidth)
                                bheight = min(image_height - btop, bheight)
                                if bwidth > 0 and bheight > 0:
                                    obj_name = BLURCOCO_CLASS_NAMES[b_class_ids[k]] if b_class_ids[k] < len(BLURCOCO_CLASS_NAMES) else str(b_class_ids[k])
                                    # Only publish non-digit and not 'L'
                                    if not (obj_name.isdigit() or obj_name == "L"):
                                        # [class_id, x1, y1, x2, y2, conf]
                                        object_dets.extend([
                                            float(b_class_ids[k]),
                                            float(bleft),
                                            float(btop),
                                            float(bleft + bwidth),
                                            float(btop + bheight),
                                            float(b_confs[k])
                                        ])
                                    color = BLURCOCO_COLORS[b_class_ids[k] % len(BLURCOCO_COLORS)] if b_class_ids[k] < len(BLURCOCO_COLORS) else (0, 0, 255)
                                    cv2.rectangle(annotated_frame, (bleft, btop), (bleft + bwidth, btop + bheight), color, 1)
                                    if b_class_ids[k] < len(BLURCOCO_CLASS_NAMES):
                                        b_label = f"{BLURCOCO_CLASS_NAMES[b_class_ids[k]]}"
                                    else:
                                        b_label = f"{b_class_ids[k]}: {b_confs[k]:.2f}"
                                    cv2.putText(annotated_frame, b_label, (bleft, btop + bheight), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Publish object detections
        if object_dets:
            object_dets_msg = Float32MultiArray()
            object_dets_msg.data = object_dets
            self.object_dets_pub.publish(object_dets_msg)

        fps_text = f"FPS: {int(self.fps)} | Avg: {avg_fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()