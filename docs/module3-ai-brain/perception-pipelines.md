---
title: Perception Pipelines for Robotics
sidebar_label: Perception Pipelines
---

# Perception Pipelines for Robotics

## Introduction to Robotics Perception

Perception is the ability of a robot to interpret and understand its environment through various sensors. In robotics, perception systems process data from cameras, LIDAR, IMU, and other sensors to enable tasks like object detection, localization, mapping, and navigation.

### Key Perception Tasks

- **Object Detection**: Identifying and localizing objects in the environment
- **Semantic Segmentation**: Pixel-level classification of scene elements
- **Instance Segmentation**: Distinguishing individual instances of objects
- **Depth Estimation**: Estimating distance to objects in the scene
- **Pose Estimation**: Determining position and orientation of objects

## Building a Basic Perception Pipeline

Here's a foundational perception pipeline that processes RGB and depth data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Load pre-trained object detection model
        self.detection_model = YOLO('yolov8n.pt')  # You can use other models too

        # Subscribe to camera topics
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Publisher for detection results
        self.detection_pub = self.create_publisher(
            # In practice, you'd use a custom message type
            Image,
            '/perception/detections',
            10
        )

        # Store depth image for processing
        self.latest_depth = None

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run object detection
        results = self.detection_model(cv_image)

        # Process detection results
        annotated_frame = results[0].plot()  # Draw bounding boxes

        # If we have depth information, add 3D information to detections
        if self.latest_depth is not None:
            # Process depth data with detections
            self.process_depth_with_detections(results[0], self.latest_depth)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.detection_pub.publish(annotated_msg)

    def depth_callback(self, msg):
        # Store depth image for use with RGB processing
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def process_depth_with_detections(self, detection_result, depth_image):
        """Process depth information with 2D detections to get 3D information"""
        boxes = detection_result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Get center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get depth at center of bounding box (with some averaging)
                depth_region = depth_image[max(0, center_y-10):center_y+10,
                                         max(0, center_x-10):center_x+10]

                if depth_region.size > 0:
                    avg_depth = np.nanmedian(depth_region[depth_region > 0])
                    if not np.isnan(avg_depth):
                        self.get_logger().info(
                            f'Detected {detection_result.names[int(box.cls)]} '
                            f'at depth: {avg_depth:.2f}m'
                        )

def main():
    rclpy.init()
    perception_node = PerceptionPipeline()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()
```

## Synthetic Data Generation with Isaac Sim

Isaac Sim excels at generating synthetic training data. Here's how to set up synthetic data collection:

```python
import omni
from omni.isaac.core import World
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2
import json
from pathlib import Path

class SyntheticDataCollector:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for different data types
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "depth").mkdir(exist_ok=True)

        self.data_index = 0

    def collect_data_frame(self, rgb_image, depth_image, semantic_image, camera_info):
        """Collect a single frame of synthetic data with annotations"""

        # Save RGB image
        rgb_filename = f"frame_{self.data_index:06d}.png"
        rgb_path = self.output_dir / "images" / rgb_filename
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # Save depth image
        depth_filename = f"depth_{self.data_index:06d}.png"
        depth_path = self.output_dir / "depth" / depth_filename
        cv2.imwrite(str(depth_path), (depth_image * 1000).astype(np.uint16))  # Scale for 16-bit storage

        # Create and save annotations
        annotations = {
            "filename": rgb_filename,
            "width": rgb_image.shape[1],
            "height": rgb_image.shape[0],
            "camera_info": {
                "fx": float(camera_info[0, 0]),
                "fy": float(camera_info[1, 1]),
                "cx": float(camera_info[0, 2]),
                "cy": float(camera_info[1, 2])
            },
            "objects": []  # This would be populated with object annotations from Isaac Sim
        }

        # Save annotations
        annotation_filename = f"annotations_{self.data_index:06d}.json"
        annotation_path = self.output_dir / "labels" / annotation_filename
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        self.data_index += 1
        print(f"Collected frame {self.data_index-1}")

# Example usage in Isaac Sim
def setup_synthetic_data_collection():
    """Setup synthetic data collection in Isaac Sim"""

    # This would typically be called from within an Isaac Sim extension
    collector = SyntheticDataCollector()

    # Set up Isaac Sim to generate various scenarios
    # (lighting changes, object variations, etc.)

    return collector
```

## Advanced Perception with Isaac ROS

Isaac ROS provides optimized perception pipelines. Here's an example using Isaac ROS packages:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np

class IsaacROSPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_perception')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Isaac ROS typically uses these topics:
        # - Left and right camera images for stereo processing
        # - Pre-processed detections from Isaac ROS nodes
        self.left_image_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect_color',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/camera/right/image_rect_color',
            self.right_image_callback,
            10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_ros/detections',
            self.detections_callback,
            10
        )

        # Publishers for processed data
        self.processed_detections_pub = self.create_publisher(
            Detection2DArray,
            '/perception/processed_detections',
            10
        )

        # Initialize stereo processing if needed
        self.left_image = None
        self.right_image = None

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process stereo pair if both images are available
        if self.left_image is not None:
            self.process_stereo_pair()

    def process_stereo_pair(self):
        """Process stereo images to generate depth information"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert to depth (this is simplified - real calibration needed)
        baseline = 0.1  # Baseline in meters
        focal_length = 616.12  # From camera calibration
        depth = (baseline * focal_length) / (disparity + 1e-6)

        self.get_logger().info(f"Generated depth map with range: {depth.min():.2f} - {depth.max():.2f}m")

    def detections_callback(self, msg):
        """Process detections from Isaac ROS pipeline"""
        # Process the detections array
        for detection in msg.detections:
            # Extract detection information
            label = detection.results[0].id if detection.results else "unknown"
            confidence = detection.results[0].score if detection.results else 0.0

            # Get bounding box
            bbox = detection.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y

            self.get_logger().info(
                f"Detected {label} with confidence {confidence:.2f} "
                f"at ({center_x:.1f}, {center_y:.1f})"
            )

        # Publish processed detections
        self.processed_detections_pub.publish(msg)
```

## Training Perception Models

Here's an example of how to train a custom perception model using synthetic data:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load all annotation files
        self.annotation_files = list((self.data_dir / "labels").glob("*.json"))

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        # Load annotation
        annotation_file = self.annotation_files[idx]
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        # Load corresponding image
        image_filename = annotation['filename']
        image_path = self.data_dir / "images" / image_filename
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For simplicity, return image and filename
        # In practice, you'd extract bounding boxes, class labels, etc.
        return image, annotation

# Training function
def train_perception_model():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = SyntheticDataset("synthetic_data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define model (using a pre-trained one for transfer learning)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Fine-tune on synthetic data
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Simplified - actual object detection loss is more complex

    model.train()
    for epoch in range(10):  # Number of training epochs
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # Simplified target
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_perception_model.pth')
```

## Best Practices for Perception Systems

- **Data Quality**: Ensure synthetic data closely matches real-world conditions
- **Model Validation**: Test perception models on real hardware regularly
- **Computational Efficiency**: Optimize models for real-time performance
- **Robustness**: Train with diverse scenarios and domain randomization
- **Safety**: Implement fallback mechanisms when perception fails
- **Calibration**: Regularly calibrate sensors for accurate measurements

## Next Steps

In the next chapter, we'll explore navigation systems that use the perception capabilities we've developed to enable autonomous robot movement and path planning.