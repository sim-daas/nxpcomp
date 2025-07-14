import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge

# Add Tkinter for sliders
import threading
import tkinter as tk

class ShelfLineDetector(Node):
    def __init__(self):
        super().__init__('shelf_line_detector')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.map_callback,
            1)
        self.publisher_lines = self.create_publisher(Float32MultiArray, '/shelf_lines', 1)
        self.publisher_image = self.create_publisher(Image, '/shelf_lines/image_raw', 1)
        self.bridge = CvBridge()

        # Shelf detection parameters (tune as needed)
        self.min_length_m = 0.1
        self.max_length_m = 2.0
        self.angle_tolerance_deg = 360

        # Parameters for tuning (set initial values)
        self.dilate_kernel = 3
        self.canny1 = 50
        self.canny2 = 100
        self.hough_thresh = 20
        self.max_line_gap = 10

        # Contour area and diagonal filter variables
        self.min_contour_area = 50
        self.max_contour_area = 10000
        self.max_contour_diagonal = 200  # new variable

        # Start Tkinter UI in a separate thread
        threading.Thread(target=self.start_tk, daemon=True).start()

    def start_tk(self):
        root = tk.Tk()
        root.title("Shelf Line Detector Tuning")

        slider_length = 400

        tk.Label(root, text="Dilation Kernel Size").pack()
        dilate_slider = tk.Scale(root, from_=1, to=11, orient=tk.HORIZONTAL, resolution=2, length=slider_length,
                                 command=lambda v: setattr(self, 'dilate_kernel', int(v)))
        dilate_slider.set(self.dilate_kernel)
        dilate_slider.pack()

        tk.Label(root, text="Canny Threshold 1").pack()
        canny1_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, length=slider_length,
                                 command=lambda v: setattr(self, 'canny1', int(v)))
        canny1_slider.set(self.canny1)
        canny1_slider.pack()

        tk.Label(root, text="Canny Threshold 2").pack()
        canny2_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, length=slider_length,
                                 command=lambda v: setattr(self, 'canny2', int(v)))
        canny2_slider.set(self.canny2)
        canny2_slider.pack()

        tk.Label(root, text="Hough Threshold").pack()
        hough_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, length=slider_length,
                                command=lambda v: setattr(self, 'hough_thresh', int(v)))
        hough_slider.set(self.hough_thresh)
        hough_slider.pack()

        tk.Label(root, text="Max Line Gap").pack()
        gap_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, length=slider_length,
                              command=lambda v: setattr(self, 'max_line_gap', int(v)))
        gap_slider.set(self.max_line_gap)
        gap_slider.pack()

        tk.Label(root, text="Min Length (m)").pack()
        min_length_slider = tk.Scale(root, from_=0, to=2, resolution=0.01, orient=tk.HORIZONTAL, length=slider_length,
                                     command=lambda v: setattr(self, 'min_length_m', float(v)))
        min_length_slider.set(self.min_length_m)
        min_length_slider.pack()

        tk.Label(root, text="Max Length (m)").pack()
        max_length_slider = tk.Scale(root, from_=0, to=5, resolution=0.01, orient=tk.HORIZONTAL, length=slider_length,
                                     command=lambda v: setattr(self, 'max_length_m', float(v)))
        max_length_slider.set(self.max_length_m)
        max_length_slider.pack()

        tk.Label(root, text="Angle Tolerance (deg)").pack()
        angle_slider = tk.Scale(root, from_=1, to=180, orient=tk.HORIZONTAL, length=slider_length,
                                command=lambda v: setattr(self, 'angle_tolerance_deg', float(v)))
        angle_slider.set(self.angle_tolerance_deg)
        angle_slider.pack()

        tk.Label(root, text="Min Contour Area").pack()
        min_area_slider = tk.Scale(root, from_=0, to=5000, orient=tk.HORIZONTAL, length=slider_length,
                                   command=lambda v: setattr(self, 'min_contour_area', float(v)))
        min_area_slider.set(self.min_contour_area)
        min_area_slider.pack()

        tk.Label(root, text="Max Contour Area").pack()
        max_area_slider = tk.Scale(root, from_=100, to=50000, orient=tk.HORIZONTAL, length=slider_length,
                                   command=lambda v: setattr(self, 'max_contour_area', float(v)))
        max_area_slider.set(self.max_contour_area)
        max_area_slider.pack()

        tk.Label(root, text="Max Contour Diagonal").pack()
        max_diag_slider = tk.Scale(root, from_=10, to=500, orient=tk.HORIZONTAL, length=slider_length,
                                   command=lambda v: setattr(self, 'max_contour_diagonal', float(v)))
        max_diag_slider.set(self.max_contour_diagonal)
        max_diag_slider.pack()

        root.mainloop()

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = np.zeros((height, width), dtype=np.uint8)
        img[data >= 80] = 255  # 80 is a good threshold for "occupied" in nav2 costmap

        # Use current slider values
        k = max(1, self.dilate_kernel)
        if k % 2 == 0:
            k += 1
        img = cv2.dilate(img, np.ones((k, k), np.uint8), iterations=1)

        # --- Basic contour detection for obstacles ---
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        shelf_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            # Find the longest diagonal in the contour (brute-force)
            max_dist = 0
            pt1 = pt2 = None
            cnt_points = cnt.reshape(-1, 2)
            for i in range(len(cnt_points)):
                for j in range(i+1, len(cnt_points)):
                    dist = np.linalg.norm(cnt_points[i] - cnt_points[j])
                    if dist > max_dist:
                        max_dist = dist
                        pt1, pt2 = tuple(cnt_points[i]), tuple(cnt_points[j])
            # Filter by maximum diagonal
            if max_dist > self.max_contour_diagonal:
                continue
            # Draw contour in blue
            cv2.drawContours(debug_img, [cnt], -1, (255,0,0), 2)
            if pt1 is not None and pt2 is not None:
                cv2.line(debug_img, pt1, pt2, (0,0,255), 2)  # Draw longest diagonal in red
            # Optionally, collect for further processing
            # shelf_candidates.append((x, y, w, h, area, pt1, pt2, max_dist))

        edges = cv2.Canny(img, self.canny1, self.canny2, apertureSize=3)
        min_length_px = int(self.min_length_m / resolution)
        max_length_px = int(self.max_length_m / resolution)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_thresh,
                                minLineLength=min_length_px, maxLineGap=self.max_line_gap)

        shelf_lines_world = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                length_px = np.hypot(x2 - x1, y2 - y1)
                if length_px < min_length_px or length_px > max_length_px:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angle = abs(angle)
                if not (abs(angle) < self.angle_tolerance_deg or abs(angle-90) < self.angle_tolerance_deg):
                    continue
                cv2.line(debug_img, (x1, y1), (x2, y2), (0,255,0), 2)
                wx1 = origin_x + (x1 + 0.5) * resolution
                wy1 = origin_y + (y1 + 0.5) * resolution
                wx2 = origin_x + (x2 + 0.5) * resolution
                wy2 = origin_y + (y2 + 0.5) * resolution
                shelf_lines_world.append([wx1, wy1, wx2, wy2])

        # Flip the debug image vertically before publishing and displaying
        debug_img_flipped = cv2.flip(debug_img, 0)

        msg_lines = Float32MultiArray()
        msg_lines.data = [coord for line in shelf_lines_world for coord in line]
        self.publisher_lines.publish(msg_lines)

        img_msg = self.bridge.cv2_to_imgmsg(debug_img_flipped, encoding='bgr8')
        self.publisher_image.publish(img_msg)

        # Show debug image for live feedback
        cv2.imshow('Shelf Lines Debug', debug_img_flipped)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ShelfLineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
