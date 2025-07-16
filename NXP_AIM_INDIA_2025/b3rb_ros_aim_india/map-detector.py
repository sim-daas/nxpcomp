import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2

class ShelfLineDetector(Node):
    def __init__(self):
        super().__init__('shelf_line_detector')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.map_callback,
            1)
        self.publisher_contours = self.create_publisher(Float32MultiArray, '/shelf_contours', 1)
        self.publisher_occmap = self.create_publisher(OccupancyGrid, '/shelf_contour_map', 1)

        # Tuned contour area and diagonal filter variables
        self.dilate_kernel = 1
        self.min_contour_area = 130
        self.max_contour_area = 650
        self.min_contour_diagonal = 19
        self.max_contour_diagonal = 45

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = np.zeros((height, width), dtype=np.uint8)
        img[data >= 80] = 255

        k = max(1, self.dilate_kernel)
        if k % 2 == 0:
            k += 1
        img = cv2.dilate(img, np.ones((k, k), np.uint8), iterations=1)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contour_data = []

        # Create a blank occupancy grid for contours only
        contour_occ = np.full((height, width), -1, dtype=np.int8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            max_dist = 0
            pt1 = pt2 = None
            cnt_points = cnt.reshape(-1, 2)
            for i in range(len(cnt_points)):
                for j in range(i+1, len(cnt_points)):
                    dist = np.linalg.norm(cnt_points[i] - cnt_points[j])
                    if dist > max_dist:
                        max_dist = dist
                        pt1, pt2 = tuple(cnt_points[i]), tuple(cnt_points[j])
            if max_dist < self.min_contour_diagonal or max_dist > self.max_contour_diagonal:
                continue
            # Draw contour and diagonal
            cv2.drawContours(debug_img, [cnt], -1, (255,0,0), 2)
            cv2.drawContours(contour_occ, [cnt], -1, 100, thickness=-1)  # Fill contour as obstacle
            if pt1 is not None and pt2 is not None:
                cv2.line(debug_img, pt1, pt2, (0,0,255), 2)
                cx = int((pt1[0] + pt2[0]) / 2)
                cy = int((pt1[1] + pt2[1]) / 2)
                cv2.circle(debug_img, (cx, cy), 3, (0,255,255), -1)
                wx1 = origin_x + (pt1[0] + 0.5) * resolution
                wy1 = origin_y + (pt1[1] + 0.5) * resolution
                wx2 = origin_x + (pt2[0] + 0.5) * resolution
                wy2 = origin_y + (pt2[1] + 0.5) * resolution
                wcx = origin_x + (cx + 0.5) * resolution
                wcy = origin_y + (cy + 0.5) * resolution
                contour_data.extend([wx1, wy1, wx2, wy2, wcx, wcy])

        debug_img_flipped = cv2.flip(debug_img, 0)
        cv2.imshow('Shelf Contours Debug', debug_img_flipped)
        cv2.waitKey(1)

        msg_contours = Float32MultiArray()
        msg_contours.data = contour_data
        self.publisher_contours.publish(msg_contours)

        # Publish the new occupancy grid with only contours as obstacles
        occ_msg = OccupancyGrid()
        occ_msg.header = msg.header
        occ_msg.info = msg.info
        occ_msg.data = contour_occ.flatten().tolist()
        self.publisher_occmap.publish(occ_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ShelfLineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()