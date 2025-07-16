import cv2
import numpy as np
import time
from pyzbar import pyzbar

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import Trigger

# --- Configuration Constants ---
MODEL_PATH = "/home/user/cognipilot/shelf.onnx"
INPUT_WIDTH = 320
INPUT_HEIGHT = 320

# Minimum confidence score for a detection. Adjust as needed.
CONF_THRESHOLD = 0.25 
# IoU threshold for Non-Maximum Suppression. Adjust as needed.
NMS_THRESHOLD = 0.45 

# Class names
CLASS_NAMES = [
    "shelf"
]

# --- Load the ONNX model ---
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except cv2.error as e:
    print(f"Error loading ONNX model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct and the ONNX file is valid.")
    exit()

# --- Set to CPU Backend and Target ---
print("Using CPU backend and target for OpenCV DNN inference.")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

class QRScannerService(Node):
    def __init__(self):
        super().__init__('qr_scanner_service')
        self.bridge = CvBridge()
        
        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1
        )
        
        # Publisher for QR code results
        self.qr_pub = self.create_publisher(
            String,
            'qr_code_results',
            10
        )
        
        # Service for on-demand QR scanning
        self.qr_service = self.create_service(
            Trigger,
            'scan_qr_code',
            self.scan_qr_service_callback
        )
        
        self.latest_camera_image = None
        self.last_qr_result = ""
        self.qr_publish_interval = 1.0  # Publish same QR every 1 second when actively scanning
        self.last_qr_time = 0
        self.actively_scanning = False  # Only scan when requested
        
        self.get_logger().info("QR Scanner Service initialized")

    def image_callback(self, msg):
        # Store the latest image for service calls
        self.latest_camera_image = msg
        
        # Only actively scan when requested
        if not self.actively_scanning:
            return
            
        # Convert ROS Image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Scan for QR codes
        qr_data = self.scan_qr_in_crop(frame)
        if qr_data:
            current_time = time.time()
            # Only publish if it's a new QR or enough time has passed
            if (qr_data != self.last_qr_result or 
                current_time - self.last_qr_time > self.qr_publish_interval):
                
                self.get_logger().info(f"QR Code detected: {qr_data}")
                
                # Publish QR code result
                msg = String()
                msg.data = qr_data
                self.qr_pub.publish(msg)
                
                self.last_qr_result = qr_data
                self.last_qr_time = current_time

    def scan_qr_service_callback(self, request, response):
        """Service callback to start/stop QR scanning"""
        self.actively_scanning = True
        self.get_logger().info("QR scanning activated via service call")
        
        # If we have a recent image, try scanning immediately
        if self.latest_camera_image:
            try:
                frame = self.bridge.imgmsg_to_cv2(self.latest_camera_image, desired_encoding='bgr8')
                qr_data = self.scan_qr_in_crop(frame)
                if qr_data:
                    self.get_logger().info(f"QR Code detected immediately: {qr_data}")
                    msg = String()
                    msg.data = qr_data
                    self.qr_pub.publish(msg)
                    self.last_qr_result = qr_data
                    self.last_qr_time = time.time()
            except Exception as e:
                self.get_logger().warn(f"Error in immediate QR scan: {e}")
        
        response.success = True
        response.message = "QR scanning activated"
        return response

    def scan_qr_in_crop(self, crop):
        """Scan for QR codes in the given crop using pyzbar."""
        try:
            # Convert to grayscale for better QR detection
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Apply some preprocessing for better QR detection
            # Adaptive thresholding
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            
            # Try decoding on both original and processed images
            qr_codes = pyzbar.decode(gray)
            if not qr_codes:
                qr_codes = pyzbar.decode(processed)
            
            if qr_codes:
                for qr in qr_codes:
                    try:
                        data = qr.data.decode('utf-8')
                        return data
                    except Exception as e:
                        self.get_logger().warn(f"Error decoding QR data: {e}")
            return None
        except Exception as e:
            self.get_logger().warn(f"Error in QR scanning: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = QRScannerService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()