---
title: Navigation Systems with Nav2
sidebar_label: Navigation Systems
---

# Navigation Systems with Nav2

## Introduction to Robot Navigation

Navigation is a fundamental capability for autonomous robots, enabling them to move from one location to another while avoiding obstacles. The Navigation2 (Nav2) stack is the standard navigation framework for ROS 2, providing a complete solution for robot navigation including path planning, obstacle avoidance, and localization.

### Key Navigation Components

- **Localization**: Determining the robot's position in a known map
- **Mapping**: Creating a representation of the environment (SLAM)
- **Path Planning**: Finding an optimal path from start to goal
- **Path Execution**: Following the planned path while avoiding obstacles
- **Recovery**: Handling navigation failures and getting back on track

## Nav2 Architecture Overview

Nav2 uses a behavior tree architecture to coordinate navigation tasks:

```
Navigation Behavior Tree:
├── Compute path to pose
├── Smooth path
├── Follow path
│   ├── Compute velocity commands
│   ├── Is goal reached?
│   └── Is path valid?
├── Get path when needed
├── Is task canceling?
└── Goal checker
```

## Installing and Setting Up Nav2

First, install the Nav2 packages:

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-turtlebot3* ros-humble-omni-navigation*

# Source the installation
source /opt/ros/humble/setup.bash
```

## Creating a Navigation Configuration

Create a navigation configuration file (`nav2_params.yaml`):

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_updated_goal_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      primary_controller: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      rotation_shim:
        plugin: "nav2_controller::SimpleProgressChecker"
        desired_linear_vel: 0.5
        lookahead_time: 1.5
        rotate_to_heading_angular_vel: 1.0
        max_angular_accel: 3.2
        goal_tolerance: 0.25

nav2_regulated_pure_pursuit_controller:
  ros__parameters:
    use_sim_time: True
    desired_linear_vel: 0.5
    max_linear_accel: 2.5
    max_linear_decel: 5.0
    desired_angular_vel: 1.0
    max_angular_accel: 3.2
    robot_base_frame: base_link
    rotate_to_heading_angular_vel: 1.0
    max_angular_vel: 1.5
    min_angular_vel: 0.0
    kp_angle: 2.0
    lookahead_time: 1.5
    min_lookahead_distance: 0.3
    max_lookahead_distance: 3.0
    use_velocity_scaled_lookahead_distance: false
    lookahead_distance: 0.6
    transform_tolerance: 0.1
    use_local_costmap: true
    short_circuit_trajectory_validation: true
    cost_scaling_dist: 1.0
    cost_scaling_gain: 1.0
    inflation_cost_scaling_factor: 3.0
    replan_frequency: 0.0
    use_dwa: false
    use_regulated_linear_velocity: true
    use_regulated_angular_velocity: false
    regulated_linear_scaling_min_radius: 0.9
    regulated_linear_scaling_min_speed: 0.25
    angular_dist_threshold: 0.785
    forward_sampling_distance: 0.5
    backward_sampling_distance: 0.2
    max_curvature_scaling_factor: 1.0
    curvature_lookahead_distance: 1.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
    backup:
      plugin: "nav2_behaviors::BackUp"
    wait:
      plugin: "nav2_behaviors::Wait"
    global_frame: odom
    robot_base_frame: base_link
    transform_tolerance: 0.1
    use_sim_time: true
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2

robot_state_publisher:
  ros__parameters:
    use_sim_time: True
```

## Creating a Navigation Launch File

Create a launch file (`navigation_launch.py`) to start the navigation system:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get paths
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    pkg_robot_navigation = get_package_share_directory('your_robot_navigation')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    autostart = LaunchConfiguration('autostart')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_robot_navigation, 'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    declare_bt_xml_file = DeclareLaunchArgument(
        'bt_xml_file',
        default_value=os.path.join(
            get_package_share_directory('nav2_bt_navigator'),
            'behavior_trees',
            'navigate_w_replanning_and_recovery.xml'
        ),
        description='Full path to the behavior tree xml file to use'
    )

    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    # Navigation launch file
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'bt_xml_file': bt_xml_file,
            'autostart': autostart
        }.items()
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_params_file,
        declare_bt_xml_file,
        declare_autostart,
        navigation_launch
    ])
```

## Creating a Complete Navigation System Node

Here's a complete navigation system that integrates with perception data:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.action import ActionClient
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import math

class NavigationSystem(Node):
    def __init__(self):
        super().__init__('navigation_system')

        # Create action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publisher for navigation status
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)

        # Timer for navigation tasks
        self.nav_timer = self.create_timer(1.0, self.navigation_callback)

        # Navigation state
        self.navigation_active = False
        self.goal_sent = False

    def scan_callback(self, msg):
        """Process LIDAR data to detect obstacles"""
        # Find minimum distance in the front 60 degrees
        front_scan_start = len(msg.ranges) // 2 - 30
        front_scan_end = len(msg.ranges) // 2 + 30

        if front_scan_start < 0:
            front_scan_start = 0
        if front_scan_end > len(msg.ranges):
            front_scan_end = len(msg.ranges)

        front_distances = msg.ranges[front_scan_start:front_scan_end]
        min_distance = min([r for r in front_distances if r > msg.range_min and r < msg.range_max], default=float('inf'))

        # Check for obstacles
        if min_distance < 0.5 and self.navigation_active:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, stopping navigation')
            # In a real system, you might cancel the current navigation goal here

    def send_navigation_goal(self, x, y, theta=0.0):
        """Send a navigation goal to the Nav2 system"""
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        import math
        from geometry_msgs.msg import Quaternion
        sin_theta = math.sin(theta / 2.0)
        cos_theta = math.cos(theta / 2.0)
        goal_msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=sin_theta,
            w=cos_theta
        )

        # Send goal
        self._send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.navigation_active = True
        self.goal_sent = True

        self.get_logger().info(f'Sent navigation goal: ({x}, {y}, {theta})')
        return True

    def goal_response_callback(self, future):
        """Handle response from navigation server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.navigation_active = False
            self.goal_sent = False
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'Navigation finished with status: {status}')

        self.navigation_active = False
        self.goal_sent = False

        status_msg = String()
        if status == 3:  # SUCCEEDED
            status_msg.data = "Navigation succeeded"
        else:
            status_msg.data = f"Navigation failed with status: {status}"
        self.status_pub.publish(status_msg)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        # Process feedback if needed
        self.get_logger().debug(f'Navigation progress: {feedback.current_pose}')

    def navigation_callback(self):
        """Timer callback for navigation tasks"""
        # This could be used to implement navigation strategies
        # For example, periodically check if we need to replan
        if self.navigation_active:
            self.get_logger().debug('Navigation in progress...')

def main():
    rclpy.init()
    nav_system = NavigationSystem()

    # Example: Send a navigation goal
    # nav_system.send_navigation_goal(1.0, 1.0, 0.0)  # Go to (1,1) with 0 rotation

    try:
        rclpy.spin(nav_system)
    except KeyboardInterrupt:
        pass
    finally:
        nav_system.destroy_node()
        rclpy.shutdown()
```

## Integration with Perception Data

Here's how to integrate perception data with navigation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class PerceptionNavigationIntegrator(Node):
    def __init__(self):
        super().__init__('perception_navigation_integrator')

        # Subscribe to perception outputs
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/perception/detections',
            self.detections_callback,
            10
        )

        # Subscribe to laser scan for additional obstacle detection
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publisher for dynamic obstacles
        self.obstacle_pub = self.create_publisher(
            MarkerArray,
            '/navigation/dynamic_obstacles',
            10
        )

        # Store detected obstacles
        self.dynamic_obstacles = []

    def detections_callback(self, msg):
        """Process detection results to identify dynamic obstacles"""
        new_obstacles = []

        for detection in msg.detections:
            # For each detection, we need to convert to world coordinates
            # This requires camera pose and calibration
            bbox = detection.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y

            # In a real system, you'd convert pixel coordinates to world coordinates
            # using camera calibration and robot pose
            world_point = self.pixel_to_world(center_x, center_y)

            if world_point:
                # Create obstacle marker
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "dynamic_obstacles"
                marker.id = len(new_obstacles)
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD

                marker.pose.position.x = world_point.x
                marker.pose.position.y = world_point.y
                marker.pose.position.z = 0.5  # Assume obstacle height

                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.5  # Cylinder diameter
                marker.scale.y = 0.5
                marker.scale.z = 1.0  # Cylinder height

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.7  # Semi-transparent

                new_obstacles.append(marker)

        # Update stored obstacles
        self.dynamic_obstacles = new_obstacles

        # Publish obstacle markers
        obstacle_array = MarkerArray()
        obstacle_array.markers = self.dynamic_obstacles
        self.obstacle_pub.publish(obstacle_array)

    def scan_callback(self, msg):
        """Process laser scan data to detect additional obstacles"""
        # Process LIDAR data to identify obstacles not detected by vision
        # This could be used to update costmaps or navigation behavior
        pass

    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates (simplified)"""
        # This is a simplified version
        # In practice, you'd need camera calibration and robot pose
        # to do proper coordinate transformation
        world_point = Point()
        world_point.x = pixel_x / 100.0  # Simplified conversion
        world_point.y = pixel_y / 100.0
        world_point.z = 0.0
        return world_point

def main():
    rclpy.init()
    integrator = PerceptionNavigationIntegrator()

    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        pass
    finally:
        integrator.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Navigation Systems

- **Map Quality**: Use high-quality maps for reliable navigation
- **Sensor Fusion**: Combine multiple sensor types for robust obstacle detection
- **Parameter Tuning**: Adjust navigation parameters based on robot dynamics
- **Safety**: Implement proper safety checks and emergency stops
- **Testing**: Test navigation in simulation before real-world deployment
- **Recovery**: Implement robust recovery behaviors for navigation failures

## Next Steps

In the next chapter, we'll explore manipulation systems that allow robots to interact with objects in their environment, building on the navigation and perception capabilities we've developed.