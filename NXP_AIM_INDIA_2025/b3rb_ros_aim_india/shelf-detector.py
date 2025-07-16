import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import tkinter as tk

class ShelfDetector(Node):
    def __init__(self):
        super().__init__('shelf_detector')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            1)
        self.publisher_rects = self.create_publisher(Float32MultiArray, '/shelf_rects', 1)
        self.publisher_occmap = self.create_publisher(OccupancyGrid, '/shelf_rect_map', 1)

        self.dilate_kernel = 5
        self.dbscan_eps = 10
        self.dbscan_min_samples = 2
        self.merge_dist_thresh = 1

        self.min_contour_area = 50
        self.max_contour_area = 1000
        self.min_contour_diagonal = 10
        self.max_contour_diagonal = 70

        self.tk_root = tk.Tk()
        self.tk_root.title("Shelf Detector Params")
        self.tk_vars = {}
        # Add all relevant sliders for debugging
        self._make_tk_slider("dilate_kernel", 1, 11, self.dilate_kernel)
        self._make_tk_slider("dbscan_eps", 1, 30, self.dbscan_eps)
        self._make_tk_slider("dbscan_min_samples", 1, 30, self.dbscan_min_samples)
        self._make_tk_slider("merge_dist_thresh", 1, 100, self.merge_dist_thresh)
        self._make_tk_slider("min_contour_area", 10, 1000, self.min_contour_area)
        self._make_tk_slider("max_contour_area", 10, 2000, self.max_contour_area)
        self._make_tk_slider("min_contour_diagonal", 1, 100, self.min_contour_diagonal)
        self._make_tk_slider("max_contour_diagonal", 1, 200, self.max_contour_diagonal)
        self.tk_root.after(100, self._tk_update)

    def _make_tk_slider(self, name, minv, maxv, init):
        var = tk.DoubleVar(value=init)
        self.tk_vars[name] = var
        scale = tk.Scale(self.tk_root, label=name, from_=minv, to=maxv, orient=tk.HORIZONTAL, resolution=1, variable=var)
        scale.pack()

    def _tk_update(self):
        self.dilate_kernel = int(self.tk_vars["dilate_kernel"].get())
        self.dbscan_eps = float(self.tk_vars["dbscan_eps"].get())
        self.dbscan_min_samples = int(self.tk_vars["dbscan_min_samples"].get())
        self.merge_dist_thresh = float(self.tk_vars["merge_dist_thresh"].get())
        self.min_contour_area = float(self.tk_vars["min_contour_area"].get())
        self.max_contour_area = float(self.tk_vars["max_contour_area"].get())
        self.min_contour_diagonal = float(self.tk_vars["min_contour_diagonal"].get())
        self.max_contour_diagonal = float(self.tk_vars["max_contour_diagonal"].get())
        self.tk_root.after(100, self._tk_update)

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = np.zeros((height, width), dtype=np.uint8)
        img[data == 100] = 255

        k = max(1, self.dilate_kernel)
        if k % 2 == 0:
            k += 1
        img = cv2.dilate(img, np.ones((k, k), np.uint8), iterations=1)

        occ_points = np.column_stack(np.where(img == 255))
        if occ_points.shape[0] == 0:
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            debug_img_flipped = cv2.flip(debug_img, 0)
            # Make output image landscape (transpose if height > width)
            if debug_img_flipped.shape[0] > debug_img_flipped.shape[1]:
                debug_img_flipped = cv2.rotate(debug_img_flipped, cv2.ROTATE_90_ANTICLOCKWISE)
            cv2.imshow('Shelf Detector Debug', debug_img_flipped)
            cv2.waitKey(1)
            return

        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(occ_points)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        clusters = []
        for label in unique_labels:
            cluster_pts = occ_points[labels == label]
            clusters.append(cluster_pts)

        merged_clusters = []
        used = set()
        for i, c1 in enumerate(clusters):
            if i in used:
                continue
            merged = c1.copy()
            used.add(i)
            c1_center = np.mean(c1, axis=0)
            for j, c2 in enumerate(clusters):
                if j <= i or j in used:
                    continue
                c2_center = np.mean(c2, axis=0)
                if np.linalg.norm(c1_center - c2_center) < self.merge_dist_thresh:
                    merged = np.vstack([merged, c2])
                    used.add(j)
            merged_clusters.append(merged)

        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contour_occ = np.full((height, width), -1, dtype=np.int8)
        rect_data = []

        for pts in merged_clusters:
            if pts.shape[0] < 10:
                continue

            pts_float = np.array(pts, dtype=np.float32)
            mean = np.mean(pts_float, axis=0)
            pts_centered = pts_float - mean
            cov = np.cov(pts_centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, order]

            proj = np.dot(pts_centered, eigvecs)
            min_proj = np.min(proj, axis=0)
            max_proj = np.max(proj, axis=0)

            # Area and diagonal filtering
            hull = cv2.convexHull(pts_float.astype(np.int32))
            area = cv2.contourArea(hull)
            max_dist = 0
            hull_pts = hull.reshape(-1, 2)
            for i in range(len(hull_pts)):
                for j in range(i+1, len(hull_pts)):
                    dist = np.linalg.norm(hull_pts[i] - hull_pts[j])
                    if dist > max_dist:
                        max_dist = dist
            if not (self.min_contour_area <= area <= self.max_contour_area):
                continue
            if not (self.min_contour_diagonal <= max_dist <= self.max_contour_diagonal):
                continue

            # Rectangle corners in image (row, col) coordinates
            rect_corners = []
            for dx, dy in [ (min_proj[0], min_proj[1]), (max_proj[0], min_proj[1]),
                            (max_proj[0], max_proj[1]), (min_proj[0], max_proj[1]) ]:
                corner = mean + dx * eigvecs[:,0] + dy * eigvecs[:,1]
                rect_corners.append(corner)
            rect_corners = np.array(rect_corners)

            rect_int = np.round(rect_corners).astype(int)
            rect_int_cv = rect_int[:, [1, 0]]
            cv2.polylines(debug_img, [rect_int_cv], isClosed=True, color=(0,255,0), thickness=2)
            cv2.drawMarker(debug_img, tuple(np.round(mean)[[1,0]].astype(int)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=8)

            rect_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(rect_mask, [rect_int_cv], 1)
            contour_occ[rect_mask == 1] = 100

            for corner in rect_corners:
                row, col = corner
                wx = origin_x + (col + 0.5) * resolution
                wy = origin_y + (row + 0.5) * resolution
                rect_data.extend([wx, wy])

        debug_img_flipped = cv2.flip(debug_img, 0)
        if debug_img_flipped.shape[0] > debug_img_flipped.shape[1]:
            debug_img_flipped = cv2.rotate(debug_img_flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('Shelf Detector Debug', debug_img_flipped)
        cv2.waitKey(1)
        self.tk_root.update()

        msg_rects = Float32MultiArray()
        msg_rects.data = rect_data
        self.publisher_rects.publish(msg_rects)

        occ_msg = OccupancyGrid()
        occ_msg.header = msg.header
        occ_msg.info = msg.info
        occ_msg.data = contour_occ.flatten().tolist()
        self.publisher_occmap.publish(occ_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ShelfDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()