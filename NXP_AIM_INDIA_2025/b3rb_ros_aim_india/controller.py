import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math
import time
import numpy as np
from enum import Enum
from typing import Optional, Tuple, List
import cv2

from sensor_msgs.msg import Joy, Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray, String
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from synapse_msgs.msg import Status, WarehouseShelf
from cv_bridge import CvBridge

# For QR service - create a simple interface
from std_srvs.srv import Trigger

# Global tunable parameters
QR_DISTANCE = 1  # Distance from shelf for QR scanning
OBJECT_DETECT_DISTANCE = 1  # Distance from shelf for object detection
MAX_QR_RETRIES = 3
MAX_OBJECT_RETRIES = 2
SHELF_AREA_RATIO_THRESHOLD = 0.150
SHELF_DETECTION_RADIUS = 1.0  # Radius around nav angle line for shelf detection
LENGTH_BREADTH_SHELF_MIN_RATIO = 2.5  # Min length/breadth ratio for shelf
LENGTH_BREADTH_SHELF_MAX_RATIO = 3.0  # Max length/breadth ratio for shelf
MIN_SHELF_LENGTH = 1.1  # Minimum shelf length in meters
MAX_SHELF_LENGTH = 1.5  # Maximum shelf length in meters
GOAL_STEP_DISTANCE = 1.0  # Distance per exploration step
OBSTACLE_AVOIDANCE_OFFSET = 1.0  # Offset distance for obstacle avoidance

# Fixed shelf dimensions
SHELF_LENGTH = 1.35  # meters
SHELF_BREADTH = 0.5  # meters

class RobotState(Enum):
    EXPLORING_FOR_SHELF = 1
    APPROACHING_SHELF = 2
    SCANNING_QR = 3
    DETECTING_OBJECTS = 4
    NAVIGATING_TO_NEXT_SHELF = 5

class WarehouseController(Node):
    def __init__(self):
        super().__init__('warehouse_controller')
        
        # Initialize ROS components
        self.bridge = CvBridge()
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # QR scanner service client
        self.qr_service_client = self.create_client(Trigger, 'scan_qr_code')
        
        # Subscriptions
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.subscription_blob_rects = self.create_subscription(
            Float32MultiArray, '/blob_rects', self.blob_rects_callback, 10)
        self.subscription_shelf_det = self.create_subscription(
            Float32MultiArray, '/shelf_det', self.shelf_det_callback, 10)
        self.subscription_object_dets = self.create_subscription(
            Float32MultiArray, '/object_dets', self.object_dets_callback, 10)
        self.subscription_qr_results = self.create_subscription(
            String, 'qr_code_results', self.qr_results_callback, 10)
        self.subscription_camera = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.subscription_status = self.create_subscription(
            Status, '/cerebri/out/status', self.cerebri_status_callback, 10)
        
        # Publishers
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', 10)
        self.publisher_shelf_data = self.create_publisher(WarehouseShelf, '/shelf_data', 10)
        
        # Parameters
        self.declare_parameter('initial_angle', 0.0)
        self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value
        
        # State machine
        self.current_state = RobotState.EXPLORING_FOR_SHELF
        self.detection_status = "0"  # "qr", "objdet", or "0"
        
        # Robot state
        self.armed = False
        self.current_pose = None
        self.goal_completed = True
        self.goal_handle_curr = None
        self.cancelling_goal = False
        
        # Data storage
        self.detected_shelf_IDs = set()
        self.current_target_shelf_id = None
        self.current_nav_angle = math.radians(self.initial_angle)
        self.last_known_shelf_pose = None
        self.last_known_shelf_blob = None  # Store blob for corner coordinates
        self.qr_decode_attempts = 0
        self.object_detect_attempts = 0
        
        # Latest sensor data
        self.latest_blob_rects = []
        self.latest_shelf_det = None
        self.latest_object_dets = []
        self.latest_camera_image = None
        self.latest_qr_result = None
        
        # Control timer
        self.control_timer = self.create_timer(0.5, self.control_loop)
        
        # QR scanning state
        self.qr_scan_requested = False
        self.qr_scan_timeout = 30.0  # seconds - increased for multiple attempts
        self.qr_scan_start_time = None
        self.qr_scan_attempts = 0
        self.max_qr_attempts = 5
        self.qr_attempt_interval = 3.0  # seconds between attempts
        self.last_qr_attempt_time = None
        self.navigation_to_qr_complete = False  # New flag to track navigation phase
        
        # Navigation reference tracking
        self.first_shelf_found = False  # Track if we've found the first shelf
        
        # Pose correction tracking
        self.pose_correction_attempts = 0
        self.max_pose_correction_attempts = 3
        self.last_correction_shelf_angle = None
        self.correction_angle_tolerance = 2.0  # degrees - minimum change to retry correction
        
        self.get_logger().info("Warehouse Controller initialized")

    def pose_callback(self, msg):
        self.current_pose = msg

    def blob_rects_callback(self, msg):
        # Data format: [cx,cy, x1,y1,x2,y2,x3,y3,x4,y4, ...]
        data = msg.data
        self.latest_blob_rects = []
        for i in range(0, len(data), 10):
            if i + 9 < len(data):
                blob = {
                    'center': (data[i], data[i+1]),
                    'corners': [(data[i+2], data[i+3]), (data[i+4], data[i+5]),
                               (data[i+6], data[i+7]), (data[i+8], data[i+9])]
                }
                self.latest_blob_rects.append(blob)

    def shelf_det_callback(self, msg):
        # Data format: [area_ratio, angle, x1, y1, x2, y2]
        if len(msg.data) >= 6:
            self.latest_shelf_det = {
                'area_ratio': msg.data[0],
                'angle': msg.data[1],
                'bbox': [msg.data[2], msg.data[3], msg.data[4], msg.data[5]]  # [x1, y1, x2, y2]
            }
        elif len(msg.data) >= 2:
            # Fallback for old format
            self.latest_shelf_det = {
                'area_ratio': msg.data[0],
                'angle': msg.data[1],
                'bbox': None
            }

    def object_dets_callback(self, msg):
        # Data format: [class_id, x1, y1, x2, y2, conf, ...]
        data = msg.data
        self.latest_object_dets = []
        for i in range(0, len(data), 6):
            if i + 5 < len(data):
                detection = {
                    'class_id': int(data[i]),
                    'bbox': [data[i+1], data[i+2], data[i+3], data[i+4]],
                    'confidence': data[i+5]
                }
                self.latest_object_dets.append(detection)

    def qr_results_callback(self, msg):
        self.latest_qr_result = msg.data

    def camera_callback(self, msg):
        self.latest_camera_image = msg

    def cerebri_status_callback(self, msg):
        if msg.mode == 3 and msg.arming == 2:
            self.armed = True
        else:
            # Initialize and arm the CMD_VEL mode.
            msg = Joy()
            msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)

    def calculate_blob_dimensions(self, blob):
        """Calculate length and breadth of blob from corners"""
        corners = np.array(blob['corners'])
        
        # Calculate distances between consecutive corners
        distances = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            dist = np.linalg.norm(p2 - p1)
            distances.append(dist)
        
        # Assume opposite sides are equal, so we have two unique distances
        side1 = (distances[0] + distances[2]) / 2
        side2 = (distances[1] + distances[3]) / 2
        
        length = max(side1, side2)
        breadth = min(side1, side2)
        
        return length, breadth

    def calculate_blob_orientation(self, blob):
        """Calculate orientation of blob using PCA"""
        corners = np.array(blob['corners'])
        
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Center the points
        centered = corners - centroid
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Get eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Principal direction (largest eigenvalue)
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Calculate orientation angle
        orientation = math.atan2(principal_direction[1], principal_direction[0])
        
        return orientation

    def is_blob_on_nav_line(self, blob):
        """Check if blob center is within radius of current navigation line"""
        if not self.current_pose:
            return False
            
        robot_x = self.current_pose.pose.pose.position.x
        robot_y = self.current_pose.pose.pose.position.y
        
        blob_x, blob_y = blob['center']
        
        # Calculate distance from blob to navigation line
        # Line equation: y - robot_y = tan(angle) * (x - robot_x)
        # Distance = |tan(angle)*x - y + (robot_y - tan(angle)*robot_x)| / sqrt(tan²(angle) + 1)
        
        if abs(math.cos(self.current_nav_angle)) < 1e-6:  # Nearly vertical line
            distance = abs(blob_x - robot_x)
        else:
            tan_angle = math.tan(self.current_nav_angle)
            distance = abs(tan_angle * blob_x - blob_y + robot_y - tan_angle * robot_x) / math.sqrt(tan_angle**2 + 1)
        
        return distance <= SHELF_DETECTION_RADIUS

    def classify_blob_as_shelf(self, blob):
        """Classify blob as shelf or obstacle using combined logic"""
        # Calculate dimensions
        length, breadth = self.calculate_blob_dimensions(blob)
        
        # Initial classification based on length/breadth ratio
        if breadth > 0:
            ratio = length / breadth
            is_shelf_by_dimensions = (LENGTH_BREADTH_SHELF_MIN_RATIO <= ratio <= LENGTH_BREADTH_SHELF_MAX_RATIO and
                                     MIN_SHELF_LENGTH <= length <= MAX_SHELF_LENGTH)
        else:
            is_shelf_by_dimensions = False
        
        # Visual confirmation conditions
        visual_conditions_met = False
        if self.latest_shelf_det and self.current_pose:
            area_ratio_ok = self.latest_shelf_det['area_ratio'] >= SHELF_AREA_RATIO_THRESHOLD
            on_nav_line = self.is_blob_on_nav_line(blob)
            
            # Convert camera-relative angle to world coordinates
            robot_yaw = self.get_yaw_from_pose(self.current_pose.pose.pose)
            shelf_angle_world = robot_yaw + math.radians(self.latest_shelf_det['angle'])
            
            # Check if shelf angle points toward blob
            blob_x, blob_y = blob['center']
            robot_x = self.current_pose.pose.pose.position.x
            robot_y = self.current_pose.pose.pose.position.y
            
            # Calculate angle from robot to blob
            blob_angle = math.atan2(blob_y - robot_y, blob_x - robot_x)
            angle_diff = abs(shelf_angle_world - blob_angle)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Normalize to [0, pi]
            
            angle_intersects = angle_diff < math.radians(30)  # Within 30 degrees
            
            visual_conditions_met = area_ratio_ok and on_nav_line and angle_intersects
        
        # Combined logic
        if is_shelf_by_dimensions and visual_conditions_met:
            return True
        elif not is_shelf_by_dimensions and visual_conditions_met:
            # Visual override
            return True
        else:
            return False

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose quaternion"""
        q = pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def create_nav_goal(self, x, y, yaw):
        """Create a NavigateToPose goal"""
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw
        
        return goal_pose

    def send_nav_goal(self, goal_pose):
        """Send navigation goal to Nav2"""
        if not self.goal_completed:
            return False
            
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available!')
            return False
        
        self.goal_completed = False
        goal_future = self.action_client.send_goal_async(goal)
        goal_future.add_done_callback(self.goal_response_callback)
        
        return True

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.goal_completed = True
            self.goal_handle_curr = None
        else:
            self.get_logger().info('Goal accepted')
            self.goal_handle_curr = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle goal completion"""
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal completed successfully")
        else:
            self.get_logger().warn(f"Goal failed with status: {status}")
        
        self.goal_completed = True
        self.goal_handle_curr = None

    def parse_qr_string(self, qr_string):
        """Parse QR string to extract shelf ID and angle"""
        try:
            # Format: shelf_id_angle_randomstring
            parts = qr_string.split('_')
            if len(parts) >= 3:
                shelf_id = int(parts[0])
                angle = float(parts[1])
                return shelf_id, angle
        except:
            pass
        return None, None

    def call_qr_service(self):
        """Call the QR scanning service"""
        if not self.qr_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("QR scanning service not available")
            return False
            
        request = Trigger.Request()
        try:
            future = self.qr_service_client.call_async(request)
            self.get_logger().info("QR scanning service called")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to call QR service: {e}")
            return False

    def get_shelf_length_breadth_vectors(self, blob):
        """Calculate length and breadth vectors from shelf corners using cyclic coordinate selection"""
        corners = blob['corners']  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
        # Calculate all 4 side vectors and lengths
        side_vectors = []
        side_lengths = []
        for i in range(4):
            p1 = np.array(corners[i])
            p2 = np.array(corners[(i + 1) % 4])  # Cyclic selection
            vector = p2 - p1
            length = np.linalg.norm(vector)
            side_vectors.append(vector)
            side_lengths.append(length)
        
        # Group opposite sides: (0,2) and (1,3) are opposite pairs
        # But we need to check consecutive sides to determine orientation
        # Side 0: from corner 0 to corner 1
        # Side 1: from corner 1 to corner 2  
        # Side 2: from corner 2 to corner 3
        # Side 3: from corner 3 to corner 0
        
        # Compare consecutive side pairs: (0,2) vs (1,3)
        avg_length_02 = (side_lengths[0] + side_lengths[2]) / 2  # opposite sides 0&2
        avg_length_13 = (side_lengths[1] + side_lengths[3]) / 2  # opposite sides 1&3
        
        # Determine which pair represents length vs breadth
        if avg_length_02 > avg_length_13:
            # Sides 0&2 are longer (length), sides 1&3 are shorter (breadth)
            length_vector = side_vectors[0] / np.linalg.norm(side_vectors[0])  # Normalize
            breadth_vector = side_vectors[1] / np.linalg.norm(side_vectors[1])  # Normalize
            length_value = avg_length_02
            breadth_value = avg_length_13
        else:
            # Sides 1&3 are longer (length), sides 0&2 are shorter (breadth)
            length_vector = side_vectors[1] / np.linalg.norm(side_vectors[1])  # Normalize
            breadth_vector = side_vectors[0] / np.linalg.norm(side_vectors[0])  # Normalize
            length_value = avg_length_13
            breadth_value = avg_length_02
        
        return length_vector, breadth_vector, length_value, breadth_value

    def calculate_qr_scan_pose(self, shelf_pose, shelf_blob):
        """Calculate pose for QR scanning using shelf corner coordinates"""
        shelf_center = np.array([shelf_pose.pose.position.x, shelf_pose.pose.position.y])
        
        # Get length and breadth vectors from corners
        length_vector, breadth_vector, length_value, breadth_value = self.get_shelf_length_breadth_vectors(shelf_blob)
        
        # QR code is on the breadth side
        # Position directly at perpendicular distance from breadth side, NOT going to breadth center first
        qr_distance = 1.5  # meters - minimum safe distance
        
        # Calculate the center point of the breadth side for reference
        breadth_side_center = shelf_center + breadth_vector * (breadth_value / 2)
        
        # Position robot directly at perpendicular distance from the breadth side
        # Move from shelf center along breadth direction by the full distance (breadth_value/2 + qr_distance)
        total_distance_from_center = (breadth_value / 2) + qr_distance
        target_position = shelf_center + breadth_vector * total_distance_from_center
        
        # Calculate yaw to face INWARD toward the breadth side center (where QR code should be)
        direction_to_qr_location = breadth_side_center - target_position
        target_yaw = math.atan2(direction_to_qr_location[1], direction_to_qr_location[0])
        
        self.get_logger().info(f"QR Scan - Shelf center: {shelf_center}")
        self.get_logger().info(f"Breadth side center: {breadth_side_center}")
        self.get_logger().info(f"Target position: {target_position}")
        self.get_logger().info(f"Total distance from shelf center: {total_distance_from_center:.2f}m")
        self.get_logger().info(f"Distance from breadth side: {qr_distance}m, Target yaw: {math.degrees(target_yaw):.1f} degrees")
        
        return self.create_nav_goal(target_position[0], target_position[1], target_yaw)

    def calculate_object_detect_pose(self, shelf_pose, shelf_blob):
        """Calculate pose for object detection using shelf corner coordinates"""
        shelf_center = np.array([shelf_pose.pose.position.x, shelf_pose.pose.position.y])
        
        # Get length and breadth vectors from corners
        length_vector, breadth_vector, length_value, breadth_value = self.get_shelf_length_breadth_vectors(shelf_blob)
        
        # Objects are detected from the length side (front of shelf)
        # Position directly at perpendicular distance from length side, NOT going to length center first
        obj_distance = 1.5  # meters - minimum safe distance
        
        # Calculate the center point of the length side for reference
        length_side_center = shelf_center + length_vector * (length_value / 2)
        
        # Position robot directly at perpendicular distance from the length side
        # Move from shelf center along length direction by the full distance (length_value/2 + obj_distance)
        total_distance_from_center = (length_value / 2) + obj_distance
        target_position = shelf_center + length_vector * total_distance_from_center
        
        # Calculate yaw to face INWARD toward the length side center (where objects should be visible)
        direction_to_objects = length_side_center - target_position
        target_yaw = math.atan2(direction_to_objects[1], direction_to_objects[0])
        
        self.get_logger().info(f"Object Detect - Shelf center: {shelf_center}")
        self.get_logger().info(f"Length side center: {length_side_center}")
        self.get_logger().info(f"Target position: {target_position}")
        self.get_logger().info(f"Total distance from shelf center: {total_distance_from_center:.2f}m")
        self.get_logger().info(f"Distance from length side: {obj_distance}m, Target yaw: {math.degrees(target_yaw):.1f} degrees")
        
        return self.create_nav_goal(target_position[0], target_position[1], target_yaw)

    def control_loop(self):
        """Main control loop implementing state machine"""
        if not self.armed or not self.current_pose:
            if not self.armed:
                self.get_logger().info("Robot not armed yet")
            if not self.current_pose:
                self.get_logger().info("No pose available yet")
            return
            
        # Debug current state
        self.get_logger().info(f"Current State: {self.current_state.name}, Goal Completed: {self.goal_completed}, Detection Status: {self.detection_status}")
        
        if self.current_state == RobotState.EXPLORING_FOR_SHELF:
            self.handle_exploring_for_shelf()
        elif self.current_state == RobotState.APPROACHING_SHELF:
            self.handle_approaching_shelf()
        elif self.current_state == RobotState.SCANNING_QR:
            self.handle_scanning_qr()
        elif self.current_state == RobotState.DETECTING_OBJECTS:
            self.handle_detecting_objects()
        elif self.current_state == RobotState.NAVIGATING_TO_NEXT_SHELF:
            self.handle_navigating_to_next_shelf()

    def handle_exploring_for_shelf(self):
        """Handle EXPLORING_FOR_SHELF state"""
        self.get_logger().info(f"EXPLORING: Goal completed: {self.goal_completed}, Blobs detected: {len(self.latest_blob_rects)}")
        
        if not self.goal_completed:
            return
            
        # Check for new shelf in blob_rects
        for i, blob in enumerate(self.latest_blob_rects):
            is_shelf = self.classify_blob_as_shelf(blob)
            self.get_logger().info(f"Blob {i}: Center {blob['center']}, Is shelf: {is_shelf}")
            
            if is_shelf:
                self.get_logger().info(f"Shelf detected at {blob['center']}!")
                
                shelf_pose = PoseStamped()
                shelf_pose.header.frame_id = "map"
                shelf_pose.pose.position.x = blob['center'][0]
                shelf_pose.pose.position.y = blob['center'][1]
                shelf_pose.pose.orientation.w = 1.0
                
                self.last_known_shelf_pose = shelf_pose
                self.last_known_shelf_blob = blob
                self.first_shelf_found = True  # Mark that we've found our first shelf
                self.reset_pose_correction_tracking()  # Reset correction tracking for new shelf
                self.current_state = RobotState.APPROACHING_SHELF
                self.get_logger().info("Transitioning to APPROACHING_SHELF")
                return
        
        # For navigation line continuation, always use robot's current position
        # except for the very first goal after QR scanning where we start from shelf center
        if self.first_shelf_found and hasattr(self, 'first_exploration_after_qr') and self.first_exploration_after_qr:
            # First exploration goal after QR scan - start from shelf center
            start_x = self.last_known_shelf_pose.pose.position.x
            start_y = self.last_known_shelf_pose.pose.position.y
            self.first_exploration_after_qr = False  # Reset flag
            self.get_logger().info(f"First exploration after QR scan - starting from shelf center ({start_x:.2f}, {start_y:.2f})")
        elif self.first_shelf_found:
            # Continue exploration from robot's current position along the navigation line
            start_x = self.current_pose.pose.pose.position.x
            start_y = self.current_pose.pose.pose.position.y
            self.get_logger().info(f"Continuing exploration from robot position ({start_x:.2f}, {start_y:.2f})")
        else:
            # Very first exploration (before any shelf is found) - use robot position
            start_x = self.current_pose.pose.pose.position.x
            start_y = self.current_pose.pose.pose.position.y
            self.get_logger().info(f"Initial exploration from robot position ({start_x:.2f}, {start_y:.2f})")
        
        next_x = start_x + GOAL_STEP_DISTANCE * math.cos(self.current_nav_angle)
        next_y = start_y + GOAL_STEP_DISTANCE * math.sin(self.current_nav_angle)
        
        self.get_logger().info(f"Sending exploration goal to ({next_x:.2f}, {next_y:.2f}) from ({start_x:.2f}, {start_y:.2f}) at angle {math.degrees(self.current_nav_angle):.1f}°")
        goal = self.create_nav_goal(next_x, next_y, self.current_nav_angle)
        self.send_nav_goal(goal)

    def handle_approaching_shelf(self):
        """Handle APPROACHING_SHELF state"""
        self.get_logger().info(f"APPROACHING: Goal completed: {self.goal_completed}")
        
        if not self.goal_completed:
            return
            
        if not self.last_known_shelf_pose:
            self.get_logger().warn("No shelf pose available, returning to exploration")
            self.current_state = RobotState.EXPLORING_FOR_SHELF
            return
        
        # Use corner coordinates to determine distances to actual shelf segments
        robot_pos = np.array([self.current_pose.pose.pose.position.x, 
                             self.current_pose.pose.pose.position.y])
        shelf_center = np.array([self.last_known_shelf_pose.pose.position.x,
                               self.last_known_shelf_pose.pose.position.y])
        
        # Get length and breadth vectors
        length_vector, breadth_vector, length_value, breadth_value = self.get_shelf_length_breadth_vectors(self.last_known_shelf_blob)
        
        # Calculate all 4 side centers (not just one per dimension)
        # Length sides (there are 2 opposite length sides)
        length_side1_center = shelf_center + breadth_vector * (breadth_value / 2)  # positive breadth direction
        length_side2_center = shelf_center - breadth_vector * (breadth_value / 2)  # negative breadth direction
        
        # Breadth sides (there are 2 opposite breadth sides) 
        breadth_side1_center = shelf_center + length_vector * (length_value / 2)   # positive length direction
        breadth_side2_center = shelf_center - length_vector * (length_value / 2)   # negative length direction
        
        # Calculate distances to all 4 sides
        dist_to_length_side1 = np.linalg.norm(robot_pos - length_side1_center)
        dist_to_length_side2 = np.linalg.norm(robot_pos - length_side2_center)
        dist_to_breadth_side1 = np.linalg.norm(robot_pos - breadth_side1_center)
        dist_to_breadth_side2 = np.linalg.norm(robot_pos - breadth_side2_center)
        
        # Find minimum distances to length and breadth sides
        min_dist_to_length_side = min(dist_to_length_side1, dist_to_length_side2)
        min_dist_to_breadth_side = min(dist_to_breadth_side1, dist_to_breadth_side2)
        
        # Determine which specific side is closest
        closest_side = ""
        if dist_to_length_side1 == min_dist_to_length_side:
            closest_side = "length_side1"
        elif dist_to_length_side2 == min_dist_to_length_side:
            closest_side = "length_side2"
        elif dist_to_breadth_side1 == min_dist_to_breadth_side:
            closest_side = "breadth_side1"
        else:
            closest_side = "breadth_side2"
        
        self.get_logger().info(f"Robot pos: {robot_pos}")
        self.get_logger().info(f"Shelf center: {shelf_center}")
        self.get_logger().info(f"Length side 1 center: {length_side1_center}, Distance: {dist_to_length_side1:.2f}")
        self.get_logger().info(f"Length side 2 center: {length_side2_center}, Distance: {dist_to_length_side2:.2f}")
        self.get_logger().info(f"Breadth side 1 center: {breadth_side1_center}, Distance: {dist_to_breadth_side1:.2f}")
        self.get_logger().info(f"Breadth side 2 center: {breadth_side2_center}, Distance: {dist_to_breadth_side2:.2f}")
        self.get_logger().info(f"Min distance to length sides: {min_dist_to_length_side:.2f}")
        self.get_logger().info(f"Min distance to breadth sides: {min_dist_to_breadth_side:.2f}")
        self.get_logger().info(f"Closest side: {closest_side}")
        self.get_logger().info(f"Length vector: {length_vector}, Length value: {length_value:.2f}")
        self.get_logger().info(f"Breadth vector: {breadth_vector}, Breadth value: {breadth_value:.2f}")
        
        # Add 2 second delay for debugging
        time.sleep(2.0)
        
        # Decide which operation to do first based on actual proximity to shelf sides
        if min_dist_to_length_side < min_dist_to_breadth_side:
            # Closer to any length side - do object detection first (objects are on length side)
            self.detection_status = "qr"  # QR scanning left to do after object detection
            self.reset_pose_correction_tracking()  # Reset correction tracking for new state
            self.current_state = RobotState.DETECTING_OBJECTS
            self.get_logger().info(f"Closer to length side ({closest_side}) - Transitioning to DETECTING_OBJECTS, will do QR scanning after")
        else:
            # Closer to any breadth side - do QR scanning first (QR is on breadth side)
            self.detection_status = "objdet"  # Object detection left to do after QR scanning
            self.reset_pose_correction_tracking()  # Reset correction tracking for new state
            self.current_state = RobotState.SCANNING_QR
            self.get_logger().info(f"Closer to breadth side ({closest_side}) - Transitioning to SCANNING_QR, will do object detection after")

    def is_shelf_centered_in_image(self, target_angle_tolerance=5.0, image_width=640, boundary_pixels=10):
        """Check if shelf is centered in image for accurate scanning"""
        if not self.latest_shelf_det:
            return True  # If no shelf detection, assume it's centered
            
        # Check angle tolerance (shelf should be roughly in center)
        shelf_angle = self.latest_shelf_det['angle']
        if abs(shelf_angle) > target_angle_tolerance:
            return False
            
        # Check bounding box position if available
        if self.latest_shelf_det.get('bbox'):
            x1, y1, x2, y2 = self.latest_shelf_det['bbox']
            center_x = (x1 + x2) / 2
            
            # Check if shelf center is not too close to image boundaries
            if (center_x < boundary_pixels or 
                center_x > (image_width - boundary_pixels)):
                return False
                
        return True

    def calculate_pose_correction(self, target_angle_tolerance=5.0):
        """Calculate small pose correction to center the shelf"""
        if not self.latest_shelf_det or not self.current_pose:
            return None
            
        shelf_angle = self.latest_shelf_det['angle']
        
        # Check if we've already tried too many corrections
        if self.pose_correction_attempts >= self.max_pose_correction_attempts:
            self.get_logger().warn(f"Max pose correction attempts ({self.max_pose_correction_attempts}) reached, proceeding without correction")
            return None
            
        # Only correct if angle is outside tolerance but not too large
        if abs(shelf_angle) <= target_angle_tolerance or abs(shelf_angle) > 30.0:
            return None
            
        # Check if shelf angle has changed significantly since last correction
        if (self.last_correction_shelf_angle is not None and 
            abs(shelf_angle - self.last_correction_shelf_angle) < self.correction_angle_tolerance):
            self.get_logger().info(f"Shelf angle ({shelf_angle:.1f}°) hasn't changed significantly since last correction, skipping")
            return None
            
        # Calculate small rotation correction
        # Positive angle means shelf is to the right, so rotate left (negative)
        correction_angle = -shelf_angle * 0.5  # Use half the angle for gentle correction
        correction_radians = math.radians(correction_angle)
        
        # Get current robot position and orientation
        current_x = self.current_pose.pose.pose.position.x
        current_y = self.current_pose.pose.pose.position.y
        current_yaw = self.get_yaw_from_pose(self.current_pose.pose.pose)
        
        # Calculate new orientation
        new_yaw = current_yaw + correction_radians
        
        # Track this correction attempt
        self.pose_correction_attempts += 1
        self.last_correction_shelf_angle = shelf_angle
        
        self.get_logger().info(f"Pose correction {self.pose_correction_attempts}/{self.max_pose_correction_attempts}: " +
                              f"shelf angle {shelf_angle:.1f}°, correction {correction_angle:.1f}°")
        
        return self.create_nav_goal(current_x, current_y, new_yaw)

    def reset_pose_correction_tracking(self):
        """Reset pose correction tracking for a new shelf"""
        self.pose_correction_attempts = 0
        self.last_correction_shelf_angle = None
        self.get_logger().info("Reset pose correction tracking")

    def handle_scanning_qr(self):
        """Handle SCANNING_QR state"""
        self.get_logger().info(f"SCANNING_QR: Goal completed: {self.goal_completed}, QR result: {self.latest_qr_result}, " +
                              f"Attempts: {self.qr_scan_attempts}/{self.max_qr_attempts}")
        
        if not self.last_known_shelf_pose:
            self.get_logger().warn("No shelf pose available, returning to exploration")
            self.current_state = RobotState.EXPLORING_FOR_SHELF
            self.reset_qr_scanning_state()
            return

        # Check if goal is completed before proceeding
        if self.goal_completed:
            # Commented out pose correction for debugging
            # if (not self.is_shelf_centered_in_image() and 
            #     self.pose_correction_attempts < self.max_pose_correction_attempts):
            #     correction_goal = self.calculate_pose_correction()
            #     if correction_goal:
            #         self.get_logger().info("Shelf not centered, applying pose correction")
            #         success = self.send_nav_goal(correction_goal)
            #         if success:
            #             return  # Wait for correction to complete
            #     else:
            #         self.get_logger().info("Pose correction not needed or already attempted, proceeding with QR scanning")
            # elif not self.is_shelf_centered_in_image():
            #     self.get_logger().warn(f"Shelf not centered but max corrections ({self.max_pose_correction_attempts}) reached, proceeding with QR scanning")
            # else:
            #     self.get_logger().info("Shelf appears centered, proceeding with QR scanning")
            
            # Perform QR scanning
            self.get_logger().info("Performing QR scanning")
            
            # Initialize scanning timers if needed
            if not self.qr_scan_start_time:
                self.qr_scan_start_time = time.time()
                self.qr_scan_requested = True
                self.get_logger().info("QR scanning started")
            
            current_time = time.time()
            
            # Determine if we should make a scanning attempt
            should_attempt = False
            if self.qr_scan_attempts == 0:
                if current_time - self.qr_scan_start_time > 2.0:
                    should_attempt = True
                    self.get_logger().info("Robot stabilized, making first QR scan attempt")
            else:
                if (self.last_qr_attempt_time and 
                    current_time - self.last_qr_attempt_time > self.qr_attempt_interval):
                    should_attempt = True
                    self.get_logger().info(f"Interval elapsed, making QR scan attempt #{self.qr_scan_attempts+1}")
            
            # Make the QR scan attempt if conditions are met
            if should_attempt and self.qr_scan_attempts < self.max_qr_attempts:
                self.qr_scan_attempts += 1
                self.last_qr_attempt_time = current_time
                self.get_logger().info(f"QR scan attempt {self.qr_scan_attempts}/{self.max_qr_attempts}")
                self.call_qr_service()
            
            # Check for successful QR scan
            if self.latest_qr_result:
                shelf_id, angle = self.parse_qr_string(self.latest_qr_result)
                if shelf_id is not None:
                    self.get_logger().info(f"QR decoded successfully after {self.qr_scan_attempts} attempts: Shelf {shelf_id}, Angle {angle}")
                    self.reset_qr_scanning_state()
                    
                    if self.detection_status == "objdet":
                        self.detection_status = "0"
                        self.reset_pose_correction_tracking()  # Reset for object detection
                        self.current_state = RobotState.DETECTING_OBJECTS
                        self.get_logger().info("Transitioning to DETECTING_OBJECTS")
                    else:
                        self.current_state = RobotState.NAVIGATING_TO_NEXT_SHELF
                        self.get_logger().info("Transitioning to NAVIGATING_TO_NEXT_SHELF")
                    return
            
            # Check for timeout or max attempts reached
            if self.qr_scan_start_time and (
                current_time - self.qr_scan_start_time > self.qr_scan_timeout or 
                self.qr_scan_attempts >= self.max_qr_attempts):
                
                self.get_logger().warn(f"QR scanning failed: attempts={self.qr_scan_attempts}, timeout={current_time - self.qr_scan_start_time:.1f}s")
                self.reset_qr_scanning_state()
                
                if self.detection_status == "objdet":
                    self.detection_status = "0"
                    self.current_state = RobotState.DETECTING_OBJECTS
                    self.get_logger().info("Failed QR scan - Transitioning to DETECTING_OBJECTS")
                else:
                    self.current_state = RobotState.NAVIGATING_TO_NEXT_SHELF
                    self.get_logger().info("Failed QR scan - Transitioning to NAVIGATING_TO_NEXT_SHELF")

    def handle_detecting_objects(self):
        """Handle DETECTING_OBJECTS state"""
        self.get_logger().info(f"DETECTING_OBJECTS: Goal completed: {self.goal_completed}, Objects: {len(self.latest_object_dets)}")
        
        if not self.last_known_shelf_pose:
            self.get_logger().warn("No shelf pose available, returning to exploration")
            self.current_state = RobotState.EXPLORING_FOR_SHELF
            return
        
        # Only send new goal if current one is completed
        if self.goal_completed:
            # Commented out pose correction for debugging
            # if not hasattr(self, '_object_detection_goal_sent'):
            #     if (not self.is_shelf_centered_in_image() and 
            #         self.pose_correction_attempts < self.max_pose_correction_attempts):
            #         correction_goal = self.calculate_pose_correction()
            #         if correction_goal:
            #             self.get_logger().info("Shelf not centered for object detection, applying pose correction")
            #             success = self.send_nav_goal(correction_goal)
            #             if success:
            #                 return  # Wait for correction to complete
            #         else:
            #             self.get_logger().info("Pose correction not needed, proceeding with object detection")
            #     elif not self.is_shelf_centered_in_image():
            #         self.get_logger().warn(f"Shelf not centered but max corrections ({self.max_pose_correction_attempts}) reached, proceeding with object detection")
            #     else:
            #         self.get_logger().info("Shelf appears centered, proceeding with object detection")
                
            # Proceed with object detection
            if not hasattr(self, '_object_detection_goal_sent'):
                time.sleep(2.0)
                obj_goal = self.calculate_object_detect_pose(self.last_known_shelf_pose, self.last_known_shelf_blob)
                self.get_logger().info(f"Sending object detection goal to ({obj_goal.pose.position.x:.2f}, {obj_goal.pose.position.y:.2f})")
                self.send_nav_goal(obj_goal)
                self._object_detection_goal_sent = True
        
        # Process object detections
        self.process_object_detections()

    def process_object_detections(self):
        """Process object detection results"""
        # Based on extractor.py BLURCOCO_CLASS_NAMES:
        # ["1", "2", "3", "4", "5", "B", "CA", "CL", "CU", "H", "L", "PP", "TB", "Z"]
        # Class indices: 0=1, 1=2, 2=3, 3=4, 4=5, 5=B, 6=CA, 7=CL, 8=CU, 9=H, 10=L, 11=PP, 12=TB, 13=Z
        
        # Map class IDs to meaningful names
        class_name_map = {
            0: "1",     # digit 1
            1: "2",     # digit 2  
            2: "3",     # digit 3
            3: "4",     # digit 4
            4: "5",     # digit 5
            5: "B",     # banana
            6: "CA",    # car
            7: "CL",    # clock
            8: "CU",    # cup
            9: "H",     # horse
            10: "L",    # letter L
            11: "PP",   # teddy bear (PP)
            12: "TB",   # teddy bear (TB)
            13: "Z"     # zebra
        }
        
        # Count valid objects (exclude digits 1-5 and letter L)
        valid_objects = []
        for detection in self.latest_object_dets:
            class_id = detection['class_id']
            class_name = class_name_map.get(class_id, f"unknown_{class_id}")
            
            # Filter out digits (0-4 which are "1"-"5") and letter "L" (class_id 10)
            # extractor.py already filters out digits and "L" in publishing, but double-check here
            if class_id not in [0, 1, 2, 3, 4, 10]:  # Exclude digits 1-5 and L
                valid_objects.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                })
        
        self.get_logger().info(f"Found {len(valid_objects)} valid objects out of {len(self.latest_object_dets)} total detections")
        
        # Log detected object details
        for obj in valid_objects:
            self.get_logger().info(f"Object: {obj['class_name']} (ID: {obj['class_id']}) - Confidence: {obj['confidence']:.2f}")
        
        if len(valid_objects) >= 1:
            # Publish shelf data
            shelf_msg = WarehouseShelf()
            shelf_msg.object_name = [obj['class_name'] for obj in valid_objects]
            shelf_msg.object_count = [1] * len(valid_objects)
            if self.latest_qr_result:
                shelf_msg.qr_decoded = self.latest_qr_result
            
            self.publisher_shelf_data.publish(shelf_msg)
            self.get_logger().info(f"Published shelf data: {shelf_msg.object_name}")
            
            # Reset the object detection goal flag
            if hasattr(self, '_object_detection_goal_sent'):
                delattr(self, '_object_detection_goal_sent')
            
            # Transition based on detection_status
            if self.detection_status == "qr":
                self.detection_status = "0"
                self.reset_pose_correction_tracking()  # Reset for QR scanning
                self.current_state = RobotState.SCANNING_QR
                self.get_logger().info("Transitioning to SCANNING_QR")
            else:
                self.current_state = RobotState.NAVIGATING_TO_NEXT_SHELF
                self.get_logger().info("Transitioning to NAVIGATING_TO_NEXT_SHELF")
        else:
            self.get_logger().warn("No valid objects detected, but proceeding anyway")
            
            # Reset the object detection goal flag
            if hasattr(self, '_object_detection_goal_sent'):
                delattr(self, '_object_detection_goal_sent')
            
            # Still transition to avoid getting stuck
            if self.detection_status == "qr":
                self.detection_status = "0"
                self.current_state = RobotState.SCANNING_QR
                self.get_logger().info("No objects - Transitioning to SCANNING_QR")
            else:
                self.current_state = RobotState.NAVIGATING_TO_NEXT_SHELF
                self.get_logger().info("No objects - Transitioning to NAVIGATING_TO_NEXT_SHELF")

    def handle_navigating_to_next_shelf(self):
        """Handle NAVIGATING_TO_NEXT_SHELF state"""
        self.get_logger().info(f"NAVIGATING_TO_NEXT: QR result: {self.latest_qr_result}")
        
        if self.latest_qr_result:
            shelf_id, angle = self.parse_qr_string(self.latest_qr_result)
            if angle is not None:
                # Set the navigation angle from the QR code
                self.current_nav_angle = math.radians(angle)
                
                # Set flag to use shelf center for the first exploration goal
                self.first_exploration_after_qr = True
                
                if shelf_id is not None:
                    self.detected_shelf_IDs.add(shelf_id)
                
                self.current_state = RobotState.EXPLORING_FOR_SHELF
                self.latest_qr_result = None  # Reset for next shelf
                
                if self.last_known_shelf_pose:
                    shelf_x = self.last_known_shelf_pose.pose.position.x
                    shelf_y = self.last_known_shelf_pose.pose.position.y
                    self.get_logger().info(f"Navigation angle {angle}° set from shelf center ({shelf_x:.2f}, {shelf_y:.2f})")
                    self.get_logger().info("Next exploration will start from this shelf center, then continue from robot position")
                
                self.get_logger().info(f"Transitioning to EXPLORING at angle {angle}° from previous shelf center")
            else:
                self.get_logger().warn("Could not parse angle from QR result")
        else:
            self.get_logger().warn("No QR result available for navigation")

    def reset_qr_scanning_state(self):
        """Reset all QR scanning state variables"""
        self.qr_scan_requested = False
        self.navigation_to_qr_complete = False
        self.qr_scan_start_time = None
        self.qr_scan_attempts = 0
        self.last_qr_attempt_time = None

def main(args=None):
    rclpy.init(args=args)
    controller = WarehouseController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()