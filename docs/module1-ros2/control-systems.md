---
title: ROS 2 Control Systems
sidebar_label: Control Systems
---

# ROS 2 Control Systems

## Introduction to ros2_control

The ros2_control framework provides a hardware abstraction layer that allows ROS 2 to interface with various types of robot hardware. It provides a standardized way to control robot joints and sensors regardless of the underlying hardware implementation.

### Key Components

- **Hardware Interface**: Abstraction layer between ROS and hardware
- **Controllers**: Software components that command hardware
- **Controller Manager**: Orchestrates controller lifecycle
- **Robot State Publisher**: Publishes robot state and transforms

## Setting Up ros2_control for Your Robot

First, we need to add ros2_control interfaces to our URDF:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Robot links and joints as defined previously -->

  <!-- ros2_control interface -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="torso_to_head">
      <command_interface name="position">
        <param name="min">-0.5</param>
        <param name="max">0.5</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>
```

## Creating Controller Configuration

Create a controller configuration file (`controllers.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

joint_trajectory_controller:
  ros__parameters:
    joints:
      - torso_to_head
    interface_names:
      - position
    state_publish_rate: 50.0
    action_monitor_rate: 20.0
    allow_partial_joints_goal: false
    open_loop_control: true
    allow_integration_in_goal_trajectories: true
    constraints:
      stopped_velocity_tolerance: 0.01
      torso_to_head:
        trajectory: 0.05
        goal: 0.01
```

## Implementing a Joint Controller Node

Here's an example of how to command joint positions:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Create publisher for trajectory commands
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Create timer to send commands
        self.timer = self.create_timer(2.0, self.send_command)
        self.i = 0

    def send_command(self):
        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = ['torso_to_head']

        # Create trajectory point
        point = JointTrajectoryPoint()
        if self.i % 2 == 0:
            point.positions = [0.3]  # Move to position 0.3
        else:
            point.positions = [-0.3]  # Move to position -0.3

        point.time_from_start = Duration(sec=1)
        trajectory.points = [point]

        # Publish trajectory
        self.publisher.publish(trajectory)
        self.get_logger().info(f'Published trajectory command: {point.positions[0]}')
        self.i += 1

def main():
    rclpy.init()
    controller = JointController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

## Launch File for Control System

Create a launch file (`control_system.launch.py`):

```python
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get URDF path
    urdf_path = os.path.join(
        get_package_share_directory('your_robot_description'),
        'urdf',
        'humanoid_robot.urdf'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path])
        }]
    )

    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[os.path.join(
            get_package_share_directory('your_robot_control'),
            'config',
            'controllers.yaml'
        )]
    )

    # Joint state broadcaster spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster']
    )

    # Joint trajectory controller spawner
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller']
    )

    return LaunchDescription([
        robot_state_publisher,
        controller_manager,
        RegisterEventHandler(
            OnProcessStart(
                target_action=controller_manager,
                on_start=[
                    joint_state_broadcaster_spawner,
                    joint_trajectory_controller_spawner
                ]
            )
        )
    ])
```

## Position, Velocity, and Effort Control

### Position Control
Position control commands specific joint angles. It's the most common control mode for manipulation tasks.

### Velocity Control
Velocity control commands joint velocities. Useful for smooth motion and when precise timing is needed.

### Effort Control
Effort control commands joint torques/forces. Useful for compliant motion and force control tasks.

## Best Practices

- Always use safety limits in controller configurations
- Test control commands in simulation before hardware deployment
- Use appropriate control rates for your robot's dynamics
- Implement proper error handling and monitoring
- Validate controller configurations before deployment

## Next Steps

In the next module, we'll explore how to create physics-based simulation environments for testing your robot control systems in a safe, repeatable environment.