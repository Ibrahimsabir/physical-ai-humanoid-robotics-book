---
title: Simulation Setup with Gazebo
sidebar_label: Simulation Setup
---

# Simulation Setup with Gazebo

## Introduction to Gazebo Simulation

Gazebo is a physics-based simulation environment that provides realistic rendering, physics, and sensor simulation for robotics applications. It's essential for testing robot behaviors in a safe, repeatable environment before deploying to real hardware.

### Key Features

- Physics simulation with realistic dynamics
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Plugin system for custom behaviors
- Integration with ROS 2 through gazebo_ros2_control

## Installing and Setting Up Gazebo

First, ensure you have Gazebo installed along with the ROS 2 integration packages:

```bash
# Install Gazebo Garden (or Harmonic for ROS 2 Humble)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos
```

## Creating a Gazebo World File

Create a basic world file (`simple_world.world`) to define your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a model from Gazebo's model database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Your robot will be spawned here -->
  </world>
</sdf>
```

## Configuring Gazebo for Your Robot

To use your robot model in Gazebo, you need to make sure it has the proper ros2_control interface. Here's an updated URDF with Gazebo plugins:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Robot links and joints as defined previously -->

  <!-- ros2_control interface for Gazebo -->
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

  <!-- Gazebo plugins -->
  <gazebo reference="torso">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Gazebo plugin for ros2_control -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find your_robot_description)/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

## Launch File for Gazebo Simulation

Create a launch file (`gazebo_simulation.launch.py`) to start Gazebo with your robot:

```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot_description = get_package_share_directory('your_robot_description')

    # Get URDF
    robot_description_path = os.path.join(
        pkg_robot_description,
        'urdf',
        'humanoid_robot.urdf'
    )

    # Get world file
    world_path = os.path.join(
        pkg_robot_description,
        'worlds',
        'simple_world.world'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', robot_description_path])
        }]
    )

    # Spawn entity node to add robot to Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'  # Start above ground
        ],
        output='screen'
    )

    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[os.path.join(
            pkg_robot_description,
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

    # Event handler to start controllers after spawn
    controller_spawner_event = RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_entity,
            on_start=[
                joint_state_broadcaster_spawner,
                joint_trajectory_controller_spawner
            ]
        )
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        controller_spawner_event
    ])
```

## Running the Simulation

To start the simulation:

```bash
# Terminal 1: Start Gazebo simulation
ros2 launch your_robot_gazebo gazebo_simulation.launch.py

# Terminal 2: Command the robot
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "..."
```

## Physics Configuration

Tune physics parameters for realistic simulation:

```xml
<physics name="ode" default="0" type="ode">
  <!-- Time step for physics updates -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor (1.0 = real-time, >1.0 = faster than real-time) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Best Practices

- Start with simple models and gradually add complexity
- Use appropriate physics parameters for your robot's dynamics
- Validate simulation behavior against physical reality when possible
- Use simulation for testing before hardware deployment
- Monitor simulation performance and adjust parameters as needed

## Next Steps

In the next chapter, we'll explore how to integrate sensors into your simulation environment and process simulated sensor data.