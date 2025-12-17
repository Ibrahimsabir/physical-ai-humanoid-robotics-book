---
title: Simulation and AI Brain Integration
sidebar_label: Simulation + AI Brain Integration
---

# Simulation and AI Brain Integration

## Introduction to Simulation and AI Brain Integration

The integration between simulation environments and AI-based robot brains represents a crucial advancement in robotics development. This integration enables robots to learn, plan, and execute complex behaviors in virtual environments before transferring these capabilities to real hardware. The combination of physics-accurate simulation with AI-powered perception, navigation, and manipulation systems creates a powerful development pipeline for autonomous robots.

### Key Integration Points

- **Perception Pipeline**: AI models trained in simulation with synthetic data
- **Navigation Systems**: AI planning algorithms validated in simulated environments
- **Manipulation Learning**: Deep learning models for grasping and manipulation
- **Sensor Fusion**: AI algorithms processing simulated sensor data
- **Behavior Trees**: AI-driven decision making in virtual worlds
- **Reinforcement Learning**: Training AI agents in simulated environments

## Architecture Overview

The integration architecture combines NVIDIA Isaac Sim for advanced simulation with AI processing capabilities:

```
┌─────────────────────────────────────────────────────────┐
│                    AI Application Layer                 │
│         (Perception, Planning, Control, Learning)       │
├─────────────────────────────────────────────────────────┤
│                   AI Processing Layer                   │
│    (Neural Networks, ML Models, Planning Algorithms)    │
├─────────────────────────────────────────────────────────┤
│                   Simulation Interface Layer            │
│    (Isaac Sim API, Sensor Simulation, Physics Engine)   │
├─────────────────────────────────────────────────────────┤
│                   Physics Simulation Layer              │
│      (Gazebo/Isaac Sim, Sensor Models, Environment)     │
└─────────────────────────────────────────────────────────┘
```

## Setting Up Isaac Sim Integration

Here's how to set up the integration between simulation and AI processing:

```python
import omni
import carb
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformBroadcaster
import threading
import time

class IsaacSimIntegrationNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_integration')

        # Initialize ROS 2 components
        self.bridge = CvBridge()

        # Publishers for AI processing
        self.rgb_pub = self.create_publisher(ImageMsg, '/isaac_sim/rgb', 10)
        self.depth_pub = self.create_publisher(ImageMsg, '/isaac_sim/depth', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/isaac_sim/camera_info', 10)
        self.ai_result_pub = self.create_publisher(String, '/ai/result', 10)

        # Subscribers for AI commands
        self.ai_command_sub = self.create_subscription(
            String,
            '/ai/command',
            self.ai_command_callback,
            10
        )

        # Initialize Isaac Sim components (simulated)
        self.isaac_sim_initialized = self.initialize_isaac_sim()

        # AI model placeholders
        self.perception_model = None
        self.navigation_model = None
        self.manipulation_model = None

        # Simulation state
        self.simulation_running = False
        self.current_frame = None
        self.current_depth = None

        # Timer for simulation updates
        self.sim_timer = self.create_timer(0.033, self.simulation_update)  # ~30 FPS

        self.get_logger().info('Isaac Sim Integration Node initialized')

    def initialize_isaac_sim(self):
        """Initialize Isaac Sim components"""
        try:
            # In a real setup, this would connect to Isaac Sim
            # For this example, we'll simulate the connection
            self.get_logger().info('Simulated Isaac Sim initialization complete')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Isaac Sim: {e}')
            return False

    def simulation_update(self):
        """Update simulation and publish sensor data"""
        if not self.isaac_sim_initialized:
            return

        # Simulate getting sensor data from Isaac Sim
        rgb_image, depth_image, camera_info = self.get_simulated_sensor_data()

        if rgb_image is not None:
            # Publish RGB image
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'camera_rgb_optical_frame'
            self.rgb_pub.publish(rgb_msg)

            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            depth_msg.header.stamp = rgb_msg.header.stamp
            depth_msg.header.frame_id = 'camera_depth_optical_frame'
            self.depth_pub.publish(depth_msg)

            # Publish camera info
            camera_info.header.stamp = rgb_msg.header.stamp
            camera_info.header.frame_id = 'camera_rgb_optical_frame'
            self.camera_info_pub.publish(camera_info)

            # Store current frame for AI processing
            self.current_frame = rgb_image
            self.current_depth = depth_image

    def get_simulated_sensor_data(self):
        """Simulate getting sensor data from Isaac Sim"""
        # In a real implementation, this would get data from Isaac Sim
        # For this example, we'll create simulated data
        height, width = 480, 640

        # Create a simulated RGB image
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Add some simulated objects to make it more realistic
        cv2.rectangle(rgb_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(rgb_image, (300, 300), 50, (0, 255, 0), -1)  # Green circle

        # Create a simulated depth image
        depth_image = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)

        # Create camera info
        camera_info = CameraInfo()
        camera_info.width = width
        camera_info.height = height
        camera_info.k = [554.256, 0.0, 320.0, 0.0, 554.256, 240.0, 0.0, 0.0, 1.0]  # Example intrinsics
        camera_info.p = [554.256, 0.0, 320.0, 0.0, 0.0, 554.256, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        return rgb_image, depth_image, camera_info

    def ai_command_callback(self, msg):
        """Handle AI commands"""
        try:
            command_data = msg.data
            self.get_logger().info(f'Received AI command: {command_data}')

            # Process the command based on type
            if 'perception' in command_data.lower():
                self.run_perception_pipeline()
            elif 'navigation' in command_data.lower():
                self.run_navigation_pipeline()
            elif 'manipulation' in command_data.lower():
                self.run_manipulation_pipeline()
            else:
                self.get_logger().warn(f'Unknown AI command: {command_data}')

        except Exception as e:
            self.get_logger().error(f'Error processing AI command: {e}')

    def run_perception_pipeline(self):
        """Run perception pipeline on current simulation data"""
        if self.current_frame is None:
            self.get_logger().warn('No current frame for perception')
            return

        try:
            # Process the image with AI perception
            results = self.process_perception(self.current_frame)

            # Publish results
            result_msg = String()
            result_msg.data = str(results)
            self.ai_result_pub.publish(result_msg)

            self.get_logger().info(f'Perception results: {results}')

        except Exception as e:
            self.get_logger().error(f'Perception pipeline error: {e}')

    def run_navigation_pipeline(self):
        """Run navigation pipeline in simulation"""
        try:
            # In simulation, we can plan and execute navigation
            # This would involve path planning, obstacle avoidance, etc.
            self.get_logger().info('Running navigation pipeline in simulation')

            # Example: plan a path to a target location
            # In Isaac Sim, this would interact with the navigation stack
            target_pose = Pose()
            target_pose.position.x = 2.0
            target_pose.position.y = 1.0
            target_pose.position.z = 0.0
            target_pose.orientation.w = 1.0

            # Publish navigation command
            result_msg = String()
            result_msg.data = f'Navigating to: ({target_pose.position.x}, {target_pose.position.y})'
            self.ai_result_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Navigation pipeline error: {e}')

    def run_manipulation_pipeline(self):
        """Run manipulation pipeline in simulation"""
        try:
            self.get_logger().info('Running manipulation pipeline in simulation')

            # Example: identify and grasp an object
            if self.current_frame is not None:
                # Process current frame to identify graspable objects
                grasp_targets = self.identify_grasp_targets(self.current_frame)

                result_msg = String()
                result_msg.data = f'Found {len(grasp_targets)} grasp targets'
                self.ai_result_pub.publish(result_msg)

                self.get_logger().info(f'Found {len(grasp_targets)} grasp targets')

        except Exception as e:
            self.get_logger().error(f'Manipulation pipeline error: {e}')

    def process_perception(self, image):
        """Process image with AI perception pipeline"""
        # This is a simplified example - in reality, this would use actual AI models
        # For now, we'll simulate object detection

        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Simulate AI perception results
        objects = [
            {'class': 'box', 'confidence': 0.92, 'bbox': [100, 100, 200, 200]},
            {'class': 'cylinder', 'confidence': 0.87, 'bbox': [300, 300, 350, 350]}
        ]

        return objects

    def identify_grasp_targets(self, image):
        """Identify potential grasp targets in the image"""
        # This would use AI models to identify graspable objects
        # For simulation, we'll return mock targets

        targets = [
            {'position': [0.5, 0.2, 0.1], 'type': 'cylinder', 'grasp_point': [0.5, 0.2, 0.15]},
            {'position': [0.8, -0.1, 0.1], 'type': 'box', 'grasp_point': [0.8, -0.1, 0.15]}
        ]

        return targets

def main():
    rclpy.init()
    node = IsaacSimIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## AI Perception Integration

Here's how to integrate AI perception models with simulation data:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import String
from cv_bridge import CvBridge
import json

class AI PerceptionIntegrator(Node):
    def __init__(self):
        super().__init__('ai_perception_integrator')

        # Initialize components
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            ImageMsg,
            '/isaac_sim/rgb',
            self.image_callback,
            10
        )

        self.perception_result_pub = self.create_publisher(
            String,
            '/ai/perception_result',
            10
        )

        # Initialize AI models
        self.object_detection_model = self.initialize_object_detection_model()
        self.segmentation_model = self.initialize_segmentation_model()
        self.depth_estimation_model = self.initialize_depth_model()

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.get_logger().info('AI Perception Integrator initialized')

    def initialize_object_detection_model(self):
        """Initialize object detection model (simulated)"""
        # In a real implementation, this would load a trained model
        # For simulation, we'll return a mock model
        self.get_logger().info('Object detection model initialized')
        return "mock_object_detection_model"

    def initialize_segmentation_model(self):
        """Initialize segmentation model (simulated)"""
        # In a real implementation, this would load a segmentation model
        self.get_logger().info('Segmentation model initialized')
        return "mock_segmentation_model"

    def initialize_depth_model(self):
        """Initialize depth estimation model (simulated)"""
        # In a real implementation, this would load a depth estimation model
        self.get_logger().info('Depth estimation model initialized')
        return "mock_depth_model"

    def image_callback(self, msg):
        """Process incoming image from simulation"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run perception pipeline
            results = self.run_perception_pipeline(cv_image)

            # Publish results
            result_msg = String()
            result_msg.data = json.dumps(results)
            self.perception_result_pub.publish(result_msg)

            self.get_logger().info(f'Perception results: {len(results.get("objects", []))} objects detected')

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def run_perception_pipeline(self, image):
        """Run complete perception pipeline"""
        results = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'objects': [],
            'segmentation': [],
            'depth_map': None
        }

        # 1. Object Detection
        objects = self.run_object_detection(image)
        results['objects'] = objects

        # 2. Semantic Segmentation
        segmentation = self.run_segmentation(image)
        results['segmentation'] = segmentation

        # 3. Depth Estimation (if depth image is available)
        # This would typically use a separate depth channel

        return results

    def run_object_detection(self, image):
        """Run object detection on image"""
        # Convert image for model input
        input_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)

        # In a real implementation, this would run the actual model
        # For simulation, we'll return mock detections
        height, width = image.shape[:2]

        objects = [
            {
                'class': 'robot',
                'confidence': 0.95,
                'bbox': [int(0.1*width), int(0.2*height), int(0.3*width), int(0.4*height)],
                'center': [int(0.2*width), int(0.3*height)]
            },
            {
                'class': 'obstacle',
                'confidence': 0.88,
                'bbox': [int(0.5*width), int(0.6*height), int(0.7*width), int(0.8*height)],
                'center': [int(0.6*width), int(0.7*height)]
            }
        ]

        return objects

    def run_segmentation(self, image):
        """Run semantic segmentation on image"""
        # In a real implementation, this would run a segmentation model
        # For simulation, we'll return mock segmentation
        height, width = image.shape[:2]

        # Create a mock segmentation map
        segmentation_map = np.zeros((height, width), dtype=np.uint8)

        # Add some mock segments
        cv2.rectangle(segmentation_map, (100, 100), (200, 200), 1, -1)  # Segment 1
        cv2.circle(segmentation_map, (300, 300), 50, 2, -1)  # Segment 2

        segments = [
            {'id': 1, 'class': 'floor', 'pixel_count': np.sum(segmentation_map == 1)},
            {'id': 2, 'class': 'object', 'pixel_count': np.sum(segmentation_map == 2)}
        ]

        return segments

    def process_simulation_scene(self, scene_description):
        """Process scene description from simulation"""
        # This would integrate with Isaac Sim's scene understanding
        # For now, we'll simulate processing a scene description
        self.get_logger().info(f'Processing scene: {scene_description}')

        # In a real implementation, this would:
        # 1. Parse the scene description from Isaac Sim
        # 2. Identify objects and their properties
        # 3. Create semantic maps of the environment
        # 4. Update the robot's world model

        processed_scene = {
            'objects': scene_description.get('objects', []),
            'surfaces': scene_description.get('surfaces', []),
            'navigable_areas': scene_description.get('navigable_areas', []),
            'interaction_points': scene_description.get('interaction_points', [])
        }

        return processed_scene
```

## Navigation and Path Planning Integration

Here's how to integrate AI-based navigation with simulation:

```python
import numpy as np
import cv2
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
import json
import heapq
from typing import List, Tuple

class AINavigationIntegrator(Node):
    def __init__(self):
        super().__init__('ai_navigation_integrator')

        # Publishers and subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.cmd_vel_pub = self.create_publisher(String, '/navigation/command', 10)

        # Navigation components
        self.occupancy_grid = None
        self.current_pose = None
        self.goal_pose = None
        self.path = []

        # AI navigation parameters
        self.navigation_params = {
            'planner_type': 'dijkstra',  # or 'astar', 'rrt', etc.
            'inflation_radius': 0.5,
            'cost_scaling_factor': 3.0,
            'update_frequency': 5.0  # Hz
        }

        self.get_logger().info('AI Navigation Integrator initialized')

    def map_callback(self, msg):
        """Process occupancy grid from simulation or mapping"""
        self.occupancy_grid = msg
        self.get_logger().debug(f'Received map: {msg.info.width}x{msg.info.height}')

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Process laser scan for real-time obstacle detection
        # This supplements the static map from simulation
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            if min_distance < 0.5:  # 50cm threshold
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.goal_pose = msg.pose
        self.get_logger().info(f'Received navigation goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        # Plan path to goal
        if self.occupancy_grid is not None:
            self.plan_path_to_goal()

    def plan_path_to_goal(self):
        """Plan path to goal using AI navigation algorithm"""
        if self.occupancy_grid is None or self.goal_pose is None:
            return

        # Convert goal to grid coordinates
        goal_grid = self.world_to_grid(
            self.goal_pose.position.x,
            self.goal_pose.position.y
        )

        # Convert current pose to grid coordinates (assuming we have it)
        current_grid = self.world_to_grid(0, 0)  # Placeholder - in real system this would be actual pose

        # Plan path using AI algorithm
        path = self.plan_path_astar(current_grid, goal_grid)

        if path:
            # Convert path back to world coordinates
            world_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                point = Point()
                point.x = world_x
                point.y = world_y
                point.z = 0.0
                world_path.append(point)

            # Publish path
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for point in world_path:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = 'map'
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose.position = point
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)

            self.path_pub.publish(path_msg)
            self.path = world_path

            self.get_logger().info(f'Planned path with {len(world_path)} waypoints')

    def plan_path_astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* path planning algorithm"""
        if self.occupancy_grid is None:
            return []

        grid = np.array(self.occupancy_grid.data).reshape(
            self.occupancy_grid.info.height,
            self.occupancy_grid.info.width
        )

        # Define movement directions (8-connected)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Check if a position is valid (not occupied)
        def is_valid(pos):
            x, y = pos
            if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
                return False
            # Check if cell is occupied (value > 50 means occupied)
            return grid[y, x] < 50

        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not is_valid(neighbor):
                    continue

                # Calculate tentative g_score
                movement_cost = 1.0 if abs(dx) + abs(dy) == 1 else 1.414  # 4-connected vs 8-connected
                tentative_g_score = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        if self.occupancy_grid is None:
            return (0, 0)

        grid_x = int((x - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
        grid_y = int((y - self.occupancy_grid.info.origin.position.y) / self.occupancy_grid.info.resolution)

        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        if self.occupancy_grid is None:
            return (0, 0)

        world_x = grid_x * self.occupancy_grid.info.resolution + self.occupancy_grid.info.origin.position.x
        world_y = grid_y * self.occupancy_grid.info.resolution + self.occupancy_grid.info.origin.position.y

        return (world_x, world_y)

    def execute_navigation(self):
        """Execute navigation along planned path"""
        if not self.path:
            return

        # In a real implementation, this would:
        # 1. Follow the planned path using local planners
        # 2. Handle dynamic obstacles detected by sensors
        # 3. Adjust path as needed based on new information
        # 4. Coordinate with simulation environment

        self.get_logger().info('Executing navigation along planned path')

        # Publish navigation command
        cmd_msg = String()
        cmd_msg.data = json.dumps({
            'command': 'follow_path',
            'path': self.path,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        })
        self.cmd_vel_pub.publish(cmd_msg)
```

## Manipulation and Grasping Integration

Here's how to integrate AI-based manipulation with simulation:

```python
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
import json
import math

class AIManipulationIntegrator(Node):
    def __init__(self):
        super().__init__('ai_manipulation_integrator')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.object_pose_sub = self.create_subscription(
            String,
            '/ai/perception_result',
            self.object_pose_callback,
            10
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )

        # Manipulation components
        self.current_joint_states = {}
        self.object_poses = []
        self.arm_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joints = ['left_gripper_joint', 'right_gripper_joint']

        # Robot kinematic parameters (example for a 6-DOF arm)
        self.kinematics_params = {
            'link_lengths': [0.1, 0.2, 0.2, 0.1, 0.1, 0.05],  # Link lengths in meters
            'joint_limits': {
                'joint1': (-3.14, 3.14),
                'joint2': (-1.57, 1.57),
                'joint3': (-3.14, 3.14),
                'joint4': (-3.14, 3.14),
                'joint5': (-1.57, 1.57),
                'joint6': (-3.14, 3.14)
            }
        }

        self.get_logger().info('AI Manipulation Integrator initialized')

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_states[name] = msg.position[i]

    def object_pose_callback(self, msg):
        """Process object poses from perception system"""
        try:
            data = json.loads(msg.data)
            if 'objects' in data:
                self.object_poses = []
                for obj in data['objects']:
                    if obj['class'] in ['object', 'graspable', 'box', 'cylinder']:
                        self.object_poses.append({
                            'class': obj['class'],
                            'position': obj.get('center', [0, 0, 0]),
                            'bbox': obj.get('bbox', [0, 0, 0, 0]),
                            'confidence': obj.get('confidence', 0.0)
                        })
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in object pose message')

    def plan_grasp(self, object_position, object_class):
        """Plan grasp for a detected object"""
        # Calculate grasp pose based on object position and type
        grasp_pose = self.calculate_grasp_pose(object_position, object_class)

        if grasp_pose:
            # Plan trajectory to reach grasp pose
            trajectory = self.plan_reach_trajectory(grasp_pose)
            return trajectory

        return None

    def calculate_grasp_pose(self, object_position, object_class):
        """Calculate appropriate grasp pose for object"""
        obj_x, obj_y, obj_z = object_position

        # Different grasp strategies based on object type
        if object_class in ['cylinder', 'bottle']:
            # Top grasp for cylindrical objects
            grasp_pose = Pose()
            grasp_pose.position.x = obj_x
            grasp_pose.position.y = obj_y
            grasp_pose.position.z = obj_z + 0.1  # 10cm above object center
            grasp_pose.orientation = self.calculate_approach_orientation('top')
        elif object_class in ['box', 'cube']:
            # Side grasp for box-shaped objects
            grasp_pose = Pose()
            grasp_pose.position.x = obj_x + 0.05  # 5cm offset from center
            grasp_pose.position.y = obj_y
            grasp_pose.position.z = obj_z
            grasp_pose.orientation = self.calculate_approach_orientation('side')
        else:
            # Default grasp
            grasp_pose = Pose()
            grasp_pose.position.x = obj_x
            grasp_pose.position.y = obj_y
            grasp_pose.position.z = obj_z + 0.05
            grasp_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        return grasp_pose

    def calculate_approach_orientation(self, approach_type):
        """Calculate appropriate approach orientation"""
        if approach_type == 'top':
            # Looking down (Z-axis pointing down)
            return Quaternion(x=0.707, y=0.0, z=0.0, w=0.707)
        elif approach_type == 'side':
            # Looking horizontally
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        else:
            # Default orientation
            return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    def plan_reach_trajectory(self, target_pose):
        """Plan joint trajectory to reach target pose"""
        # This would use inverse kinematics to solve for joint angles
        # For this example, we'll simulate the process

        # Get current end-effector pose (simplified)
        current_joints = [self.current_joint_states.get(joint, 0.0) for joint in self.arm_joints]

        # Calculate target joint angles using inverse kinematics
        # In a real system, this would call an IK solver
        target_joints = self.inverse_kinematics(target_pose, current_joints)

        if target_joints:
            # Create trajectory message
            trajectory = JointTrajectory()
            trajectory.joint_names = self.arm_joints

            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = target_joints
            point.velocities = [0.0] * len(target_joints)  # Start and end at rest
            point.accelerations = [0.0] * len(target_joints)

            # Set execution time (1 second)
            duration = Duration()
            duration.sec = 1
            duration.nanosec = 0
            point.time_from_start = duration

            trajectory.points.append(point)
            return trajectory

        return None

    def inverse_kinematics(self, target_pose, current_joints):
        """Solve inverse kinematics for target pose"""
        # This is a simplified example - in reality, this would use a proper IK solver
        # For demonstration, we'll return a simple approximation

        # In a real implementation, this would:
        # 1. Use an IK solver (like KDL, MoveIt, or custom implementation)
        # 2. Consider joint limits and robot constraints
        # 3. Optimize for multiple possible solutions

        # For this example, return slightly modified current joints
        # In reality, this would calculate proper joint angles for the target pose
        return [j + 0.1 for j in current_joints]  # Simple offset for demo

    def execute_grasp(self, object_idx):
        """Execute grasp on specified object"""
        if object_idx >= len(self.object_poses):
            self.get_logger().error(f'Invalid object index: {object_idx}')
            return

        obj = self.object_poses[object_idx]
        self.get_logger().info(f'Planning grasp for {obj["class"]} at {obj["position"]}')

        # Plan the grasp trajectory
        trajectory = self.plan_grasp(obj['position'], obj['class'])

        if trajectory:
            # Execute the reaching motion
            self.trajectory_pub.publish(trajectory)

            # After reaching, close the gripper
            self.close_gripper()

            self.get_logger().info(f'Grasp planned and executed for {obj["class"]}')
        else:
            self.get_logger().error(f'Failed to plan grasp for {obj["class"]}')

    def close_gripper(self):
        """Close the robot gripper"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints

        point = JointTrajectoryPoint()
        # Close gripper (adjust values based on your gripper design)
        point.positions = [0.02, 0.02]  # Closed position
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]

        duration = Duration()
        duration.sec = 1
        duration.nanosec = 0
        point.time_from_start = duration

        trajectory.points.append(point)
        self.gripper_pub.publish(trajectory)

    def open_gripper(self):
        """Open the robot gripper"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints

        point = JointTrajectoryPoint()
        # Open gripper
        point.positions = [0.08, 0.08]  # Open position
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]

        duration = Duration()
        duration.sec = 1
        duration.nanosec = 0
        point.time_from_start = duration

        trajectory.points.append(point)
        self.gripper_pub.publish(trajectory)
```

## Complete Integration Example

Here's a complete example showing how all components work together:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import threading
import time

class CompleteIntegrationNode(Node):
    def __init__(self):
        super().__init__('complete_integration')

        # Publishers
        self.integration_status_pub = self.create_publisher(
            String,
            '/integration/status',
            10
        )

        # Initialize all integration components
        self.isaac_sim_node = IsaacSimIntegrationNode()
        self.perception_node = AI PerceptionIntegrator()
        self.navigation_node = AINavigationIntegrator()
        self.manipulation_node = AIManipulationIntegrator()

        # Integration state
        self.integration_state = {
            'isaac_sim_connected': False,
            'perception_ready': False,
            'navigation_ready': False,
            'manipulation_ready': False,
            'overall_status': 'initializing'
        }

        # Timer for integration monitoring
        self.integration_timer = self.create_timer(1.0, self.integration_monitor)

        self.get_logger().info('Complete Integration System initialized')

    def integration_monitor(self):
        """Monitor integration status and publish updates"""
        # Update integration status based on component readiness
        self.integration_state['isaac_sim_connected'] = self.isaac_sim_node.isaac_sim_initialized
        self.integration_state['perception_ready'] = self.perception_node.object_detection_model is not None
        self.integration_state['navigation_ready'] = self.navigation_node.occupancy_grid is not None
        self.integration_state['manipulation_ready'] = len(self.manipulation_node.current_joint_states) > 0

        # Determine overall status
        ready_components = sum([
            self.integration_state['isaac_sim_connected'],
            self.integration_state['perception_ready'],
            self.integration_state['navigation_ready'],
            self.integration_state['manipulation_ready']
        ])

        if ready_components == 4:
            self.integration_state['overall_status'] = 'fully_integrated'
        elif ready_components >= 2:
            self.integration_state['overall_status'] = 'partially_integrated'
        else:
            self.integration_state['overall_status'] = 'initializing'

        # Publish integration status
        status_msg = String()
        status_msg.data = json.dumps(self.integration_state)
        self.integration_status_pub.publish(status_msg)

        if self.integration_state['overall_status'] == 'fully_integrated':
            self.get_logger().info('All systems integrated and ready!')
        else:
            self.get_logger().info(f'Integration status: {self.integration_state["overall_status"]} '
                                 f'({ready_components}/4 components ready)')

def main():
    rclpy.init()
    integration_node = CompleteIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Configuration for Integration

Here's the configuration file for the integrated system:

```yaml
# config/isaac_sim_ai_integration.yaml
# Configuration for Isaac Sim and AI brain integration

isaac_sim_integration:
  ros__parameters:
    use_sim_time: true
    simulation_speed: 1.0  # Real-time simulation
    rendering_enabled: true
    physics_update_rate: 60.0  # Hz

ai_perception:
  ros__parameters:
    detection_threshold: 0.5
    max_detection_range: 10.0  # meters
    confidence_threshold: 0.7
    object_classes: ["robot", "obstacle", "graspable", "person"]
    preprocessing:
      image_size: [640, 480]
      normalization_mean: [0.485, 0.456, 0.406]
      normalization_std: [0.229, 0.224, 0.225]

ai_navigation:
  ros__parameters:
    planner_type: "astar"
    inflation_radius: 0.5  # meters
    cost_scaling_factor: 3.0
    update_frequency: 5.0  # Hz
    min_distance_to_obstacle: 0.3  # meters
    max_planning_time: 5.0  # seconds

ai_manipulation:
  ros__parameters:
    grasp_approach_distance: 0.1  # meters
    grasp_lift_height: 0.05  # meters
    gripper_force_limit: 50.0  # Newtons
    joint_velocity_limit: 1.0  # rad/s
    collision_check_enabled: true

integration_monitor:
  ros__parameters:
    status_publish_rate: 1.0  # Hz
    timeout_threshold: 5.0  # seconds
    critical_components: ["isaac_sim", "perception", "navigation", "manipulation"]
```

## Best Practices for Integration

### 1. Data Pipeline Consistency

Ensure that data flows consistently between simulation and AI systems:

```python
# Use consistent data formats and timestamps across all components
def standardize_sensor_data(self, raw_data, sensor_type):
    """Standardize sensor data format across simulation and AI systems"""
    standardized = {
        'timestamp': self.get_clock().now().nanoseconds / 1e9,
        'sensor_type': sensor_type,
        'data': raw_data,
        'frame_id': self.get_parameter('frame_id').value
    }
    return standardized
```

### 2. Error Handling and Recovery

Implement robust error handling for simulation-AI integration:

```python
def handle_integration_error(self, error_type, error_message):
    """Handle errors in the integration pipeline"""
    self.get_logger().error(f'Integration error ({error_type}): {error_message}')

    # Implement recovery strategies based on error type
    if error_type == 'perception_failure':
        # Fall back to alternative perception methods
        pass
    elif error_type == 'navigation_failure':
        # Stop navigation and request new plan
        pass
    elif error_type == 'simulation_disconnect':
        # Switch to safe mode or emergency procedures
        pass
```

### 3. Performance Monitoring

Monitor the performance of the integrated system:

```python
def monitor_integration_performance(self):
    """Monitor performance metrics for the integrated system"""
    metrics = {
        'perception_latency': self.calculate_perception_latency(),
        'navigation_frequency': self.get_navigation_frequency(),
        'simulation_fidelity': self.assess_simulation_fidelity(),
        'ai_processing_load': self.get_ai_processing_load()
    }

    # Log metrics and trigger alerts if thresholds are exceeded
    if metrics['perception_latency'] > 0.1:  # 100ms threshold
        self.get_logger().warn('High perception latency detected')
```

## Troubleshooting Integration Issues

### Common Problems and Solutions

1. **Synchronization Issues**: Ensure all components use the same time source
2. **Data Format Mismatches**: Standardize message formats across components
3. **Performance Bottlenecks**: Monitor and optimize AI model inference
4. **Calibration Problems**: Verify sensor and simulation calibration
5. **Communication Failures**: Check network and ROS 2 communication

## Next Steps

This integration between simulation and AI brain systems enables the development of sophisticated autonomous robots. In the next module, we'll explore how to integrate Vision-Language-Action (VLA) systems that combine these perception, navigation, and manipulation capabilities with natural language understanding and high-level task planning.