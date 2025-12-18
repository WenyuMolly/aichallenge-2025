#!/usr/bin/env python3
import time
import numpy as np
from threading import Lock
from collections import deque
import cv2
from PIL import Image
from io import BytesIO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Image as Image
from autoware_auto_control_msgs.msg import AckermannControlCommand
from cv_bridge import CvBridge

from tiny_lidar_net_controller_core import TinyLidarNetCore


class TinyLidarNetNode(Node):
    """ROS 2 Node for TinyLidarNet autonomous driving control.

    This node subscribes to LaserScan messages, processes them using the
    TinyLidarNetCore logic, and publishes AckermannControlCommand messages.
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.model_type', 'lidar')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('model.resnet_out_dim', 256)
        self.declare_parameter('camera.image_topic', '/sensing/camera/image_raw')
        self.declare_parameter('camera.image_size', 224)
        self.declare_parameter('camera.img_mean', [0.485, 0.456, 0.406])
        self.declare_parameter('camera.img_std', [0.229, 0.224, 0.225])
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('debug', False)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        architecture = self.get_parameter('model.architecture').value
        model_type = self.get_parameter('model.model_type').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        resnet_out_dim = self.get_parameter('model.resnet_out_dim').value
        image_topic = self.get_parameter('camera.image_topic').value
        image_size = self.get_parameter('camera.image_size').value
        img_mean = tuple(self.get_parameter('camera.img_mean').value)
        img_std = tuple(self.get_parameter('camera.img_std').value)
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value
        self.model_type = model_type

        try:
            self.core = TinyLidarNetCore(
                input_dim=input_dim,
                output_dim=output_dim,
                architecture=architecture,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                max_range=max_range,
                model_type=model_type,
                resnet_out_dim=resnet_out_dim,
                image_size=image_size,
                img_mean=img_mean,
                img_std=img_std,
            )
            self.get_logger().info(
                f"Core initialized. Model: {model_type}, Arch: {architecture}, MaxRange: {max_range}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise e

        # --- Image preprocessing setup (multimodal only) ---
        self.cv_bridge = CvBridge()
        self.img_size = image_size
        self.img_mean = np.array(img_mean, dtype=np.float32)
        self.img_std = np.array(img_std, dtype=np.float32)
        self.latest_img_features = None
        self.img_lock = Lock()

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        
        if self.model_type == 'multimodal':
            self.sub_image = self.create_subscription(
                Image, image_topic, self.image_callback, qos
            )
            self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/awsim/control_cmd", 1
        )

        self.get_logger().info("TinyLidarNetNode is ready.")

    def scan_callback(self, msg: LaserScan):
        """Callback for LaserScan subscription.

        Processes the scan data via the core logic and publishes a control command.

        Args:
            msg (LaserScan): The incoming ROS 2 LaserScan message.
        """
        start_time = time.monotonic()
        # 1. Convert ROS message to Numpy
        # We pass the raw array; the core logic handles NaN/Inf and normalization.
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 2. Process via Core Logic
        img_features = None
        if self.model_type == 'multimodal':
            with self.img_lock:
                # Assign the latest image features to each scan
                img_features = self.latest_img_features if self.latest_img_features is not None else np.zeros(self.core.resnet_out_dim, dtype=np.float32)

        accel, steer = self.core.process(ranges, img_features=img_features)

        # 3. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = float(accel)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)

        # 4. Debug Logging
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Logs internal performance metrics at fixed intervals."""
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | "
                    f"Max: {max_time:.2f}ms"
                )
                self.inference_times.clear()
            
            self.last_log_time = now

    def image_callback(self, msg: Image):
        """Callback for Image subscription (multimodal mode only).

        Converts ROS image to numpy, extracts features (placeholder),
        and stores them for use in the next scan inference.

        Args:
            msg (Image): The incoming ROS 2 Image message.
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Resize to expected size
            cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
            
            # Normalize (convert to float, normalize by mean/std)
            img_array = np.float32(cv_image) / 255.0
            img_array = (img_array - self.img_mean) / self.img_std
            
            # Flatten or convert to feature representation
            # For now, flatten the image to match expected multimodal input
            # In a real scenario, this would come from a ResNet feature extractor
            img_features = img_array.flatten()[:self.core.resnet_out_dim]
            
            # Pad with zeros if needed
            if len(img_features) < self.core.resnet_out_dim:
                img_features = np.pad(img_features, (0, self.core.resnet_out_dim - len(img_features)))
            
            with self.img_lock:
                self.latest_img_features = img_features.astype(np.float32)
                # self.get_logger().info("Updated latest image features for multimodal inference.")
        
        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
