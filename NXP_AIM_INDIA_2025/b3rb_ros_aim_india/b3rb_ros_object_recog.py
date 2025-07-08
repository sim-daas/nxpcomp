# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import WarehouseShelf

import cv2
import numpy as np
from pyzxing import BarCodeReader

from sensor_msgs.msg import Image
import threading
import time

QOS_PROFILE_DEFAULT = 10


class QRCodeRecognizer(Node):
	def __init__(self):
		super().__init__('qr_code_recognizer')
		self.subscription_camera = self.create_subscription(
			Image,
			'/camera/image_raw',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)
		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			'/shelf_data',
			QOS_PROFILE_DEFAULT)
		self.qr_found = False  # Only publish once per cycle
		self.reader = BarCodeReader()
		self.lock = threading.Lock()

	def scan_qr_code(self, img):
		"""Scan for QR codes using pyzxing and return the first decoded string, or None."""
		import tempfile
		with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
			cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
			results = self.reader.decode(tmp.name)
			if results and isinstance(results, list) and len(results) > 0:
				result = results[0]
				if 'parsed' in result and result['parsed']:
					return result['parsed']
				elif 'raw' in result and result['raw']:
					return result['raw']
				elif 'data' in result and result['data']:
					return result['data']
		return None

	def camera_image_callback(self, msg):
		with self.lock:
			if self.qr_found:
				return
			if msg.encoding == 'rgb8':
				img = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
			elif msg.encoding == 'bgr8':
				img = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			else:
				self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
				return
			data = self.scan_qr_code(img)
			if data:
				# Ensure data is a string, not bytes
				if isinstance(data, bytes):
					data = data.decode('utf-8')
				self.get_logger().info(f"QR Code detected: {data}")
				shelf_msg = WarehouseShelf()
				shelf_msg.qr_decoded = data
				self.publisher_shelf_data.publish(shelf_msg)
				self.qr_found = True
				# Start a timer to reset qr_found after 10 seconds
				threading.Thread(target=self._reset_qr_found_after_delay, daemon=True).start()

	def _reset_qr_found_after_delay(self):
		time.sleep(10)
		with self.lock:
			self.qr_found = False


def main(args=None):
	rclpy.init(args=args)
	node = QRCodeRecognizer()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()