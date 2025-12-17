---
title: Manipulation Systems with MoveIt2
sidebar_label: Manipulation Systems
---

# Manipulation Systems with MoveIt2

## Introduction to Robot Manipulation

Manipulation is the capability of a robot to interact with objects in its environment using end-effectors (grippers, tools, etc.). MoveIt2 is the standard motion planning framework for ROS 2, providing comprehensive tools for motion planning, inverse kinematics, collision checking, and trajectory execution.

### Key Manipulation Components

- **Motion Planning**: Finding collision-free paths for robot arms
- **Inverse Kinematics**: Calculating joint angles to achieve desired end-effector poses
- **Collision Checking**: Ensuring planned motions don't collide with obstacles
- **Trajectory Execution**: Sending planned motions to robot controllers
- **Grasping**: Planning and executing object grasping behaviors

## Installing and Setting Up MoveIt2

First, install the MoveIt2 packages:

```bash
# Install MoveIt2 packages
sudo apt update
sudo apt install ros-humble-moveit ros-humble-moveit-visual-tools ros-humble-moveit-resources*

# Source the installation
source /opt/ros/humble/setup.bash
```

## Creating a MoveIt2 Configuration Package

MoveIt2 requires a configuration package for your robot. Here's the structure:

```
your_robot_moveit_config/
├── config/
│   ├── controllers.yaml
│   ├── fake_controllers.yaml
│   ├── joint_limits.yaml
│   ├── kinematics.yaml
│   ├── moveit_controllers.yaml
│   ├── moveit_py.yaml
│   ├── ompl_planning.yaml
│   └── pilz_cartesian_limits.yaml
├── launch/
│   ├── move_group.launch.py
│   ├── moveit_rviz.launch.py
│   ├── static_virtual_joint_tfs.launch.py
│   └── warehouse_db.launch.py
├── CMakeLists.txt
└── package.xml
```

## MoveIt2 Configuration Files

Create the main configuration file (`config/kinematics.yaml`):

```yaml
manipulator:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.005
  kinematics_solver_attempts: 3
```

Create the planning configuration (`config/ompl_planning.yaml`):

```yaml
planner_configs:
  SBLkConfigDefault:
    type: geometric::SBL
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
  ESTkConfigDefault:
    type: geometric::EST
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
  LBKPIECEkConfigDefault:
    type: geometric::LBKPIECE
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    border_fraction: 0.9  # Fraction of time focused on boarder (0.0, 1.0]
    min_valid_path_fraction: 0.5  # Accept partially valid moves above fraction
  BKPIECEkConfigDefault:
    type: geometric::BKPIECE
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    border_fraction: 0.9  # Fraction of time focused on boarder (0.0, 1.0]
    failed_expansion_score_factor: 0.5  # When extending motion fails, scale score by factor
    min_valid_path_fraction: 0.5  # Accept partially valid moves above fraction
  KPIECEkConfigDefault:
    type: geometric::KPIECE
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.5  # When close to goal select goal, with this probability
    border_fraction: 0.9  # Fraction of time focused on boarder (0.0, 1.0]
    failed_expansion_score_factor: 0.5  # When extending motion fails, scale score by factor
    min_valid_path_fraction: 0.5  # Accept partially valid moves above fraction
  RRTkConfigDefault:
    type: geometric::RRT
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
  RRTConnectkConfigDefault:
    type: geometric::RRTConnect
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
  RRTstarkConfigDefault:
    type: geometric::RRTstar
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
    delay_collision_checking: 1  # Stop collision checking as soon as C-free parent found
  TRRTkConfigDefault:
    type: geometric::TRRT
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
    max_states_failed: 10  # When to start increasing temp
    temp_change_factor: 2.0  # How much to increase or decrease temp
    min_temperature: 10e-10  # Lower limit of temperature
    init_temperature: 10e-6  # Starting temperature
    frountier_threshold: 0.0  # Dist new state to nearest neighbor to disqualify as frontier
    frountierNodeRatio: 0.1  # 1/10, or 1 nonfrontier for every 10 frontier
    k_constant: 0.0  # Value used to normalize expresssion
  PRMkConfigDefault:
    type: geometric::PRM
    max_nearest_neighbors: 10  # Maximum distance between two nodes
  PRMstarkConfigDefault:
    type: geometric::PRMstar
  FMTkConfigDefault:
    type: geometric::FMT
    num_samples: 1000  # Number of states that the planner should sample
    radius_multiplier: 1.1  # Multiplier used for the nearest neighbors search radius
    nearest_k: 1  # Use Knearest strategy if 1, otherwise rgraph strategy
    cache_cc: 1  # Use collision checking cache
    heuristics: 0  # Activate cost to go heuristics
    extended_fmt: 1  # Activate the extended FMT*: adding new samples if planner does not finish successfully
  BFMTkConfigDefault:
    type: geometric::BFMT
    num_samples: 1000  # Number of states that the planner should sample
    radius_multiplier: 1.0  # Multiplier used for the nearest neighbors search radius
    nearest_k: 1  # Use Knearest strategy if 1, otherwise rgraph strategy
    balanced: 0  # Exploration strategy: balanced true expands one tree from start and one from goal, false picks the smaller tree for expansion
    optimality: 1  # Terminate also when the best possible path is found
    heuristics: 1  # Activate cost to go heuristics
    cache_cc: 1  # Use collision checking cache
    extended_fmt: 1  # Activate the extended FMT*: adding new samples if planner does not finish successfully
  PDSTkConfigDefault:
    type: geometric::PDST
  STRIDEkConfigDefault:
    type: geometric::STRIDE
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
    use_projected_distance: 0  # Whether nearest neighbors are computed based on distances in a projection of the state rather distances in the state space itself
    degree: 16  # Maximum degree of a node in the geometric graph
    max_close_samples: 20  # Maximum number of close samples to retain for building the geometric graph
    max_elite_samples: 20  # Maximum number of elite samples to retain for building the geometric graph
  BiTRRTkConfigDefault:
    type: geometric::BiTRRT
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    temp_change_factor: 0.1  # How much to increase or decrease temp
    init_temperature: 100  # Starting temperature
    frountier_threshold: 0.0  # Dist new state to nearest neighbor to disqualify as frontier
    frountier_node_ratio: 0.1  # 1/10, or 1 nonfrontier for every 10 frontier
    cost_threshold: 1e300  # Optimization objective cost threshold
  LBTRRTkConfigDefault:
    type: geometric::LBTRRT
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
    epsilon: 0.4  # Optimality approximation factor
  BiESTkConfigDefault:
    type: geometric::BiEST
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
  ProjESTkConfigDefault:
    type: geometric::ProjEST
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
    goal_bias: 0.05  # When close to goal select goal, with this probability
  LazyPRMkConfigDefault:
    type: geometric::LazyPRM
    range: 0.0  # Max motion added to tree. If 0.0, default to 0.2 * state space extent
  LazyPRMstarkConfigDefault:
    type: geometric::LazyPRMstar
  SPARSkConfigDefault:
    type: geometric::SPARS
    stretch_factor: 3.0  # roadmap spanner stretch factor
    sparse_delta_fraction: 0.25  # delta fraction for connection distance
    dense_delta_fraction: 0.001  # delta fraction for dense connection
    max_failures: 1000  # maximum consecutive failure limit
  SPARStwoConfigDefault:
    type: geometric::SPARStwo
    stretch_factor: 3.0  # roadmap spanner stretch factor
    sparse_delta_fraction: 0.25  # delta fraction for connection distance
    dense_delta_fraction: 0.001  # delta fraction for dense connection
    max_failures: 5000  # maximum consecutive failure limit

manipulator:
  planner_configs:
    - SBLkConfigDefault
    - ESTkConfigDefault
    - LBKPIECEkConfigDefault
    - BKPIECEkConfigDefault
    - KPIECEkConfigDefault
    - RRTkConfigDefault
    - RRTConnectkConfigDefault
    - RRTstarkConfigDefault
    - TRRTkConfigDefault
    - PRMkConfigDefault
    - PRMstarkConfigDefault
    - FMTkConfigDefault
    - BFMTkConfigDefault
    - PDSTkConfigDefault
    - STRIDEkConfigDefault
    - BiTRRTkConfigDefault
    - LBTRRTkConfigDefault
    - BiESTkConfigDefault
    - ProjESTkConfigDefault
    - LazyPRMkConfigDefault
    - LazyPRMstarkConfigDefault
    - SPARSkConfigDefault
    - SPARStwoConfigDefault
```

## Creating a Basic Manipulation Node

Here's a basic manipulation node using MoveIt2's Python interface:

```python
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from geometry_msgs.msg import Pose, Point
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
import numpy as np

class ManipulationSystem(Node):
    def __init__(self):
        super().__init__('manipulation_system')

        # Create MoveIt2 interface
        self.planning_scene_interface = None
        self.move_group_interface = None

        # Publisher for collision objects
        self.collision_object_pub = self.create_publisher(
            CollisionObject,
            '/collision_object',
            10
        )

        # Service client for planning scene
        self.get_planning_scene_client = self.create_client(
            GetPlanningScene,
            '/get_planning_scene'
        )

        # Timer to initialize interfaces
        self.timer = self.create_timer(1.0, self.initialize_interfaces)

    def initialize_interfaces(self):
        """Initialize MoveIt2 interfaces"""
        # This would typically involve initializing the MoveIt2 Python interface
        # For now, we'll just log that initialization is happening
        self.get_logger().info('Initializing manipulation interfaces...')
        self.timer.cancel()  # Stop the timer once initialized

    def add_object_to_scene(self, name, pose, dimensions):
        """Add an object to the planning scene"""
        collision_object = CollisionObject()
        collision_object.header.frame_id = 'base_link'
        collision_object.id = name

        # Define the primitive shape (box)
        primitive = SolidPrimitive()
        primitive.type = primitive.BOX
        primitive.dimensions = dimensions  # [width, depth, height]

        # Define the pose
        collision_object.pose = pose

        # Add the primitive to the collision object
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)

        # Set operation to ADD
        collision_object.operation = CollisionObject.ADD

        # Publish the collision object
        self.collision_object_pub.publish(collision_object)
        self.get_logger().info(f'Added object {name} to planning scene')

    def plan_to_pose(self, target_pose):
        """Plan motion to a target pose"""
        # This would use MoveIt2's motion planning capabilities
        # For now, we'll just log the action
        self.get_logger().info(f'Planning to pose: {target_pose}')

    def execute_plan(self, plan):
        """Execute a motion plan"""
        # This would execute the planned motion
        self.get_logger().info('Executing motion plan')

def main():
    rclpy.init()
    manipulator = ManipulationSystem()

    # Example: Add an object to the scene
    # Create a pose for the object
    object_pose = Pose()
    object_pose.position.x = 0.5
    object_pose.position.y = 0.0
    object_pose.position.z = 0.1
    object_pose.orientation.w = 1.0

    # Add the object to the scene
    manipulator.add_object_to_scene(
        'table',
        object_pose,
        [1.0, 0.8, 0.2]  # width, depth, height
    )

    try:
        rclpy.spin(manipulator)
    except KeyboardInterrupt:
        pass
    finally:
        manipulator.destroy_node()
        rclpy.shutdown()
```

## Advanced Manipulation with Perception Integration

Here's an example of how to integrate perception with manipulation:

```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Pose, Point, Vector3
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np

class PerceptionManipulationIntegrator(Node):
    def __init__(self):
        super().__init__('perception_manipulation_integrator')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to perception outputs
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/perception/detections',
            self.detections_callback,
            10
        )

        # Publisher for collision objects
        self.collision_object_pub = self.create_publisher(
            CollisionObject,
            '/collision_object',
            10
        )

        # Store detected objects
        self.detected_objects = {}

    def detections_callback(self, msg):
        """Process detection results to identify graspable objects"""
        for detection in msg.detections:
            # Extract object information from detection
            bbox = detection.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y

            # In a real system, you'd convert pixel coordinates to world coordinates
            # using camera calibration and robot pose
            world_point = self.pixel_to_world(center_x, center_y)

            if world_point:
                # Determine object type and add to planning scene
                obj_type = detection.results[0].id if detection.results else "unknown"
                confidence = detection.results[0].score if detection.results else 0.0

                if confidence > 0.7:  # Only process high-confidence detections
                    self.add_object_to_scene(obj_type, world_point)

    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates (simplified)"""
        # This is a simplified version
        # In practice, you'd need camera calibration and robot pose
        # to do proper coordinate transformation
        world_point = Point()
        world_point.x = pixel_x / 100.0  # Simplified conversion
        world_point.y = pixel_y / 100.0
        world_point.z = 0.1  # Assume object is at 0.1m height
        return world_point

    def add_object_to_scene(self, obj_type, position):
        """Add detected object to MoveIt2 planning scene"""
        collision_object = CollisionObject()
        collision_object.header.frame_id = 'base_link'
        collision_object.id = f'{obj_type}_{len(self.detected_objects)}'

        # Create a box primitive for the object
        primitive = SolidPrimitive()
        if 'bottle' in obj_type.lower() or 'cup' in obj_type.lower':
            # Use cylinder for bottle/cup
            primitive.type = primitive.CYLINDER
            primitive.dimensions = [0.05, 0.15]  # radius, height
        else:
            # Use box for other objects
            primitive.type = primitive.BOX
            primitive.dimensions = [0.05, 0.05, 0.05]  # width, depth, height

        # Set object pose
        pose = Pose()
        pose.position = position
        pose.orientation.w = 1.0

        # Add to collision object
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        # Publish to planning scene
        self.collision_object_pub.publish(collision_object)
        self.get_logger().info(f'Added {obj_type} to planning scene at ({position.x}, {position.y}, {position.z})')

        # Store reference to object
        self.detected_objects[collision_object.id] = {
            'type': obj_type,
            'position': position,
            'pose': pose
        }

    def find_grasp_poses(self, object_id):
        """Find potential grasp poses for an object"""
        if object_id not in self.detected_objects:
            return []

        obj_info = self.detected_objects[object_id]
        obj_pose = obj_info['pose']

        # Generate grasp poses around the object
        grasp_poses = []

        # Approach from different directions
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            grasp_pose = Pose()
            grasp_pose.position.x = obj_pose.position.x + 0.15 * np.cos(angle)
            grasp_pose.position.y = obj_pose.position.y + 0.15 * np.sin(angle)
            grasp_pose.position.z = obj_pose.position.z + 0.1

            # Orient gripper to approach from the side
            from math import sin, cos
            grasp_pose.orientation.x = 0.0
            grasp_pose.orientation.y = 0.707  # Point gripper down
            grasp_pose.orientation.z = 0.0
            grasp_pose.orientation.w = 0.707

            grasp_poses.append(grasp_pose)

        return grasp_poses

def main():
    rclpy.init()
    integrator = PerceptionManipulationIntegrator()

    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        pass
    finally:
        integrator.destroy_node()
        rclpy.shutdown()
```

## Creating a Grasping Node

Here's a more complete grasping implementation:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.msg import Grasp, GripperTranslation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
import numpy as np

class GraspingSystem(Node):
    def __init__(self):
        super().__init__('grasping_system')

        # Publisher for grasp commands
        self.grasp_command_pub = self.create_publisher(
            String,
            '/grasp/command',
            10
        )

        # Publisher for gripper commands
        self.gripper_command_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )

    def create_grasp(self, grasp_pose):
        """Create a grasp message for MoveIt2"""
        grasp = Grasp()

        # Set pre-grasp posture (gripper open)
        grasp.pre_grasp_posture = self.create_gripper_trajectory(0.08)  # Open gripper

        # Set grasp posture (gripper closed)
        grasp.grasp_posture = self.create_gripper_trajectory(0.0)  # Close gripper

        # Set grasp pose
        grasp.grasp_pose = grasp_pose

        # Set approach and retreat
        grasp.pre_grasp_approach = self.create_gripper_translation(0.1, 0.1, [1, 0, 0])
        grasp.post_grasp_retreat = self.create_gripper_translation(0.1, 0.1, [-1, 0, 0])

        # Set grasp quality (how good this grasp is)
        grasp.grasp_quality = 1.0

        return grasp

    def create_gripper_trajectory(self, position):
        """Create a joint trajectory for gripper control"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['left_gripper_finger_joint', 'right_gripper_finger_joint']

        point = JointTrajectoryPoint()
        point.positions = [position, position]  # Both fingers move together
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)
        return trajectory

    def create_gripper_translation(self, min_distance, desired_distance, direction_vector):
        """Create a gripper translation for approach/retreat"""
        translation = GripperTranslation()
        translation.direction.vector.x = direction_vector[0]
        translation.direction.vector.y = direction_vector[1]
        translation.direction.vector.z = direction_vector[2]
        translation.direction.header.frame_id = 'base_link'
        translation.min_distance = min_distance
        translation.desired_distance = desired_distance
        return translation

    def execute_grasp(self, grasp_pose):
        """Execute a grasp at the given pose"""
        self.get_logger().info('Executing grasp...')

        # First, move gripper to pre-grasp position
        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header.frame_id = 'base_link'
        pre_grasp_pose.pose = grasp_pose

        # Add offset for pre-grasp position
        pre_grasp_pose.pose.position.z += 0.1  # 10cm above grasp point

        # Send pre-grasp command (this would use MoveIt2 to plan and execute)
        # For now, we'll just log the action
        self.get_logger().info(f'Moving to pre-grasp position: ({pre_grasp_pose.pose.position.x}, {pre_grasp_pose.pose.position.y}, {pre_grasp_pose.pose.position.z})')

        # Open gripper
        open_gripper_cmd = self.create_gripper_trajectory(0.08)
        self.gripper_command_pub.publish(open_gripper_cmd)
        self.get_logger().info('Opening gripper')

        # Wait briefly
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))

        # Move to grasp position
        self.get_logger().info(f'Moving to grasp position: ({grasp_pose.position.x}, {grasp_pose.position.y}, {grasp_pose.position.z})')

        # Close gripper
        close_gripper_cmd = self.create_gripper_trajectory(0.02)  # Close around object
        self.gripper_command_pub.publish(close_gripper_cmd)
        self.get_logger().info('Closing gripper')

        # Lift object
        lift_pose = grasp_pose
        lift_pose.position.z += 0.1  # Lift 10cm
        self.get_logger().info(f'Lifting object to: ({lift_pose.position.x}, {lift_pose.position.y}, {lift_pose.position.z})')

        self.get_logger().info('Grasp completed successfully')

def main():
    rclpy.init()
    grasper = GraspingSystem()

    # Example: Execute a grasp at a specific position
    grasp_pose = PoseStamped().pose
    grasp_pose.position.x = 0.5
    grasp_pose.position.y = 0.0
    grasp_pose.position.z = 0.1
    grasp_pose.orientation.w = 1.0

    # Execute the grasp after a delay
    def delayed_grasp():
        grasper.execute_grasp(grasp_pose)

    timer = grasper.create_timer(2.0, delayed_grasp)

    try:
        rclpy.spin(grasper)
    except KeyboardInterrupt:
        pass
    finally:
        grasper.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Manipulation Systems

- **Safety**: Always implement safety checks and limits for manipulation
- **Planning**: Use collision-aware motion planning for safe manipulation
- **Calibration**: Ensure proper calibration of cameras and robot kinematics
- **Grasp Planning**: Plan grasps considering object shape and stability
- **Force Control**: Implement force/torque control for delicate manipulation
- **Testing**: Test manipulation in simulation before real-world deployment

## Next Steps

In the next module, we'll explore Vision-Language-Action (VLA) systems that connect natural language understanding to robot action execution, building on all the capabilities we've developed so far.