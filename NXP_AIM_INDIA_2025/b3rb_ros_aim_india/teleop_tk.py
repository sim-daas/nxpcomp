import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import tkinter as tk

# Key mapping for Ackermann: W/S = forward/reverse, A/D = left/right
KEY_BINDINGS = {
    'w': (1.0, 0.0),   # forward
    's': (-1.0, 0.0),  # reverse
    'a': (0.0, 1.0),   # left
    'd': (0.0, -1.0),  # right
}

class TeleopTk(Node):
    def __init__(self):
        super().__init__('teleop_tk')
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', 10)
        self.pressed_keys = set()
        self.speed = 0.0
        self.turn = 0.0

        self.root = tk.Tk()
        self.root.title("Ackermann Teleop (WASD)")
        self.label = tk.Label(self.root, text="Use WASD keys to drive, X to stop", font=("Helvetica", 16))
        self.label.pack(padx=10, pady=10)
        self.root.bind('<KeyPress>', self.keydown)
        self.root.bind('<KeyRelease>', self.keyup)
        self.status = tk.Label(self.root, text="Speed: 0.0, Turn: 0.0", font=("Helvetica", 14))
        self.status.pack(padx=10, pady=10)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def keydown(self, event):
        key = event.keysym.lower()
        if key == 'x':
            self.pressed_keys.clear()
            self.speed, self.turn = 0.0, 0.0
            self.publish_joy()
            self.status.config(text=f"Speed: {self.speed}, Turn: {self.turn}")
            return
        if key in KEY_BINDINGS:
            self.pressed_keys.add(key)
            self.update_motion()

    def keyup(self, event):
        key = event.keysym.lower()
        if key in KEY_BINDINGS:
            self.pressed_keys.discard(key)
            self.update_motion()
        # Immediately send zero command if no keys are pressed
        if not self.pressed_keys:
            self.speed, self.turn = 0.0, 0.0
            self.publish_joy()
            self.status.config(text=f"Speed: {self.speed}, Turn: {self.turn}")

    def update_motion(self):
        speed, turn = 0.0, 0.0
        for key in self.pressed_keys:
            s, t = KEY_BINDINGS[key]
            speed += s
            turn += t
        # Clamp to [-1, 1]
        self.speed = max(-1.0, min(1.0, speed))
        self.turn = max(-1.0, min(1.0, turn))
        self.publish_joy()
        self.status.config(text=f"Speed: {self.speed}, Turn: {self.turn}")

    def publish_joy(self):
        msg = Joy()
        # Only manual mode (button 0), arming (button 7)
        msg.buttons = [0]*8
        msg.buttons[0] = 1  # manual mode
        msg.buttons[7] = 1  # arm
        msg.axes = [0.0, self.speed, 0.0, self.turn]
        self.publisher_joy.publish(msg)

    def on_close(self):
        self.destroy_node()
        self.root.destroy()

def main(args=None):
    rclpy.init(args=args)
    node = TeleopTk()
    try:
        node.root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
