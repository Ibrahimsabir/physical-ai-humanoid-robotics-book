---
title: System Integration for Autonomous Humanoid Robots
sidebar_label: System Integration
---

# System Integration for Autonomous Humanoid Robots

## Introduction to System Integration

System integration is the process of combining all individual robot subsystems into a cohesive, functional autonomous humanoid robot. This involves connecting the robotic nervous system (ROS 2), digital twin (simulation), AI brain (perception/navigation/manipulation), and Vision-Language-Action (VLA) systems into a unified platform that can execute complex tasks autonomously.

### Key Integration Challenges

- **Data Flow**: Ensuring seamless data flow between subsystems
- **Timing**: Coordinating real-time and non-real-time operations
- **Resource Management**: Managing computational and power resources
- **Error Handling**: Implementing robust error recovery across subsystems
- **Performance**: Optimizing system performance across all components

## Architecture Overview

The integrated system architecture follows a layered approach:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                 │
│                (Voice, GUI, Mobile App)                 │
├─────────────────────────────────────────────────────────┤
│                   Command Processing Layer              │
│           (NLP, Context Awareness, Task Planning)       │
├─────────────────────────────────────────────────────────┤
│                    Coordination Layer                   │
│        (Behavior Trees, State Machines, Orchestration)  │
├─────────────────────────────────────────────────────────┤
│                   Capability Layer                      │
│    (Navigation, Manipulation, Perception, Speech)       │
├─────────────────────────────────────────────────────────┤
│                   Control Layer                         │
│         (Motion Control, Joint Control, Safety)         │
├─────────────────────────────────────────────────────────┤
│                   Hardware Layer                        │
│        (Sensors, Actuators, Processors, Communication)  │
└─────────────────────────────────────────────────────────┘
```

## Creating the Integration Launch File

Here's a comprehensive launch file that brings together all subsystems:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch

def generate_launch_description():
    # Get package directories
    pkg_robot_bringup = get_package_share_directory('your_robot_bringup')
    pkg_robot_description = get_package_share_directory('your_robot_description')
    pkg_robot_navigation = get_package_share_directory('your_robot_navigation')
    pkg_robot_manipulation = get_package_share_directory('your_robot_manipulation')
    pkg_robot_perception = get_package_share_directory('your_robot_perception')
    pkg_robot_vla = get_package_share_directory('your_robot_vla')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_namespace = LaunchConfiguration('robot_namespace', default='')

    # 1. Robot State Publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_description, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 2. Navigation System
    navigation_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_navigation, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 3. Manipulation System
    manipulation_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_manipulation, 'launch', 'manipulation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 4. Perception System
    perception_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_perception, 'launch', 'perception_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 5. VLA (Vision-Language-Action) System
    vla_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_robot_vla, 'launch', 'vla_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 6. System Integration Node
    system_integrator = Node(
        package='your_robot_system',
        executable='system_integrator',
        name='system_integrator',
        parameters=[
            os.path.join(pkg_robot_bringup, 'config', 'system_integration.yaml')
        ],
        remappings=[
            ('/command_input', '/vla/command'),
            ('/action_output', '/system/actions'),
        ]
    )

    # 7. Context Manager
    context_manager = Node(
        package='your_robot_vla',
        executable='context_manager',
        name='context_manager',
        parameters=[
            os.path.join(pkg_robot_bringup, 'config', 'context_manager.yaml')
        ]
    )

    # 8. Task Planner
    task_planner = Node(
        package='your_robot_vla',
        executable='action_planner',
        name='action_planner',
        parameters=[
            os.path.join(pkg_robot_bringup, 'config', 'action_planner.yaml')
        ]
    )

    # 9. Safety Monitor
    safety_monitor = Node(
        package='your_robot_system',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[
            os.path.join(pkg_robot_bringup, 'config', 'safety_monitor.yaml')
        ]
    )

    # Event handler to start dependent systems after robot state publisher
    nav_spawner_event = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[navigation_system]
        )
    )

    manipulation_spawner_event = RegisterEventHandler(
        OnProcessStart(
            target_action=navigation_system,
            on_start=[manipulation_system]
        )
    )

    perception_spawner_event = RegisterEventHandler(
        OnProcessStart(
            target_action=manipulation_system,
            on_start=[perception_system]
        )
    )

    vla_spawner_event = RegisterEventHandler(
        OnProcessStart(
            target_action=perception_system,
            on_start=[vla_system]
        )
    )

    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='',
            description='Robot namespace'
        ),

        # System Components
        robot_state_publisher,
        system_integrator,
        context_manager,
        task_planner,
        safety_monitor,

        # Event handlers for ordered startup
        nav_spawner_event,
        manipulation_spawner_event,
        perception_spawner_event,
        vla_spawner_event,
    ])
```

## System Integration Node Implementation

Here's the main system integration node that coordinates all subsystems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
import json
from typing import Dict, Any, Optional
from enum import Enum
import threading
import time

class SystemState(Enum):
    IDLE = "idle"
    PROCESSING_COMMAND = "processing_command"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    PERCEIVING = "perceiving"
    ERROR = "error"
    SAFETY_STOP = "safety_stop"

class SystemIntegratorNode(Node):
    def __init__(self):
        super().__init__('system_integrator')

        # State management
        self.current_state = SystemState.IDLE
        self.previous_state = SystemState.IDLE
        self.system_status = {
            'navigation_ready': False,
            'manipulation_ready': False,
            'perception_ready': False,
            'vla_ready': False,
            'safety_ok': True
        }

        # Action clients for major subsystems
        self.nav_client = ActionClient(self, MoveGroup, 'move_group')
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        # Subscribers for system status
        self.command_sub = self.create_subscription(
            String,
            '/integrated_commands',
            self.command_callback,
            10
        )

        self.nav_status_sub = self.create_subscription(
            String,
            '/navigation/status',
            self.nav_status_callback,
            10
        )

        self.manip_status_sub = self.create_subscription(
            String,
            '/manipulation/status',
            self.manip_status_callback,
            10
        )

        self.perception_status_sub = self.create_subscription(
            String,
            '/perception/status',
            self.perception_status_callback,
            10
        )

        self.vla_status_sub = self.create_subscription(
            String,
            '/vla/status',
            self.vla_status_callback,
            10
        )

        self.safety_sub = self.create_subscription(
            Bool,
            '/safety/emergency_stop',
            self.safety_callback,
            10
        )

        # Publishers for system commands
        self.status_pub = self.create_publisher(String, '/system/status', 10)
        self.nav_goal_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(0.1, self.system_monitor)

        # Command queue for handling multiple commands
        self.command_queue = []
        self.command_queue_lock = threading.Lock()

        # Initialize system
        self.initialize_system()

        self.get_logger().info('System Integrator node initialized')

    def initialize_system(self):
        """Initialize the integrated system"""
        self.get_logger().info('Initializing integrated system...')

        # Check if all subsystems are ready
        self.check_subsystem_readiness()

        # Wait for action servers to be available
        self.get_logger().info('Waiting for action servers...')
        nav_ready = self.nav_client.wait_for_server(timeout_sec=10.0)
        arm_ready = self.arm_client.wait_for_server(timeout_sec=10.0)
        gripper_ready = self.gripper_client.wait_for_server(timeout_sec=10.0)

        self.system_status['navigation_ready'] = nav_ready
        self.system_status['manipulation_ready'] = arm_ready and gripper_ready

        if nav_ready and arm_ready and gripper_ready:
            self.get_logger().info('All action servers ready')
            self.current_state = SystemState.IDLE
        else:
            self.get_logger().error('Some action servers not available')
            self.current_state = SystemState.ERROR

        # Publish initial status
        self.publish_system_status()

    def command_callback(self, msg: String):
        """Process integrated commands"""
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('type', 'unknown')

            with self.command_queue_lock:
                self.command_queue.append(command_data)

            self.get_logger().info(f'Queued command: {command_type}')

            # Process commands if system is ready
            if self.current_state == SystemState.IDLE and self.system_status['safety_ok']:
                self.process_next_command()

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid command JSON: {msg.data}')

    def process_next_command(self):
        """Process the next command in the queue"""
        with self.command_queue_lock:
            if not self.command_queue:
                return

            command = self.command_queue.pop(0)

        command_type = command.get('type', 'unknown')

        if command_type == 'navigate':
            self.execute_navigation_command(command)
        elif command_type == 'manipulate':
            self.execute_manipulation_command(command)
        elif command_type == 'perceive':
            self.execute_perception_command(command)
        elif command_type == 'complex_task':
            self.execute_complex_task_command(command)
        else:
            self.get_logger().warn(f'Unknown command type: {command_type}')
            self.current_state = SystemState.IDLE

    def execute_navigation_command(self, command: Dict[str, Any]):
        """Execute navigation command"""
        self.current_state = SystemState.NAVIGATING

        target_pose = command.get('target_pose', {})
        x = target_pose.get('x', 0.0)
        y = target_pose.get('y', 0.0)
        theta = target_pose.get('theta', 0.0)

        # Create and send navigation goal
        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        goal_pose.position.z = 0.0

        import math
        goal_pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.orientation.w = math.cos(theta / 2.0)

        self.nav_goal_pub.publish(goal_pose)
        self.get_logger().info(f'Navigating to ({x}, {y})')

    def execute_manipulation_command(self, command: Dict[str, Any]):
        """Execute manipulation command"""
        self.current_state = SystemState.MANIPULATING

        action = command.get('action', 'unknown')
        object_name = command.get('object', 'unknown')

        if action == 'pick_up':
            self.execute_pickup(object_name)
        elif action == 'place':
            target_location = command.get('target_location', 'default')
            self.execute_place(object_name, target_location)
        elif action == 'gripper':
            gripper_action = command.get('gripper_action', 'unknown')
            self.execute_gripper_action(gripper_action)

    def execute_perception_command(self, command: Dict[str, Any]):
        """Execute perception command"""
        self.current_state = SystemState.PERCEIVING

        task = command.get('task', 'unknown')
        self.get_logger().info(f'Executing perception task: {task}')

        # In a real system, this would trigger perception pipelines
        # For now, just log and return to idle

    def execute_complex_task_command(self, command: Dict[str, Any]):
        """Execute complex multi-step task"""
        task_name = command.get('task_name', 'unknown')
        steps = command.get('steps', [])

        self.get_logger().info(f'Executing complex task: {task_name} with {len(steps)} steps')

        # For complex tasks, we might want to use a behavior tree or state machine
        # For now, we'll execute steps sequentially
        self.execute_sequential_steps(steps)

    def execute_sequential_steps(self, steps: list):
        """Execute steps sequentially"""
        for step in steps:
            step_type = step.get('type', 'unknown')

            if step_type == 'navigate':
                self.execute_navigation_command(step)
            elif step_type == 'manipulate':
                self.execute_manipulation_command(step)
            elif step_type == 'perceive':
                self.execute_perception_command(step)

            # Wait for completion before next step
            # In a real system, this would be more sophisticated
            time.sleep(2)  # Simulate step completion time

    def execute_pickup(self, object_name: str):
        """Execute object pickup"""
        self.get_logger().info(f'Picking up {object_name}')
        # Implementation would involve perception, planning, and execution

    def execute_place(self, object_name: str, location: str):
        """Execute object placement"""
        self.get_logger().info(f'Placing {object_name} at {location}')
        # Implementation would involve navigation and manipulation

    def execute_gripper_action(self, action: str):
        """Execute gripper action"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['left_gripper_finger_joint', 'right_gripper_finger_joint']

        point = JointTrajectoryPoint()
        if action == 'open':
            point.positions = [0.08, 0.08]  # Open position
        elif action == 'close':
            point.positions = [0.02, 0.02]  # Closed position (not fully closed to grip objects)
        else:
            return

        point.time_from_start.sec = 1
        trajectory.points.append(point)

        self.gripper_pub.publish(trajectory)

    def nav_status_callback(self, msg: String):
        """Handle navigation status updates"""
        try:
            status_data = json.loads(msg.data)
            status = status_data.get('status', 'unknown')

            if status == 'completed' and self.current_state == SystemState.NAVIGATING:
                self.current_state = SystemState.IDLE
                self.process_next_command()  # Process next command if available
            elif status == 'failed':
                self.current_state = SystemState.ERROR
                self.get_logger().error('Navigation failed')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid navigation status JSON: {msg.data}')

    def manip_status_callback(self, msg: String):
        """Handle manipulation status updates"""
        # Similar to navigation status but for manipulation
        if self.current_state == SystemState.MANIPULATING:
            # Check if manipulation is complete
            if 'completed' in msg.data.lower():
                self.current_state = SystemState.IDLE
                self.process_next_command()

    def perception_status_callback(self, msg: String):
        """Handle perception status updates"""
        if self.current_state == SystemState.PERCEIVING:
            # Check if perception is complete
            if 'completed' in msg.data.lower():
                self.current_state = SystemState.IDLE
                self.process_next_command()

    def vla_status_callback(self, msg: String):
        """Handle VLA system status updates"""
        # Update VLA readiness status
        if 'ready' in msg.data.lower():
            self.system_status['vla_ready'] = True
        elif 'not ready' in msg.data.lower():
            self.system_status['vla_ready'] = False

    def safety_callback(self, msg: Bool):
        """Handle safety system updates"""
        if not msg.data:  # Emergency stop activated
            self.current_state = SystemState.SAFETY_STOP
            self.system_status['safety_ok'] = False
            self.get_logger().error('Emergency stop activated!')
        else:  # Safety system OK
            self.system_status['safety_ok'] = True
            if self.current_state == SystemState.SAFETY_STOP:
                self.current_state = SystemState.IDLE

    def check_subsystem_readiness(self):
        """Check if all subsystems are ready"""
        # This would involve checking the status of all subsystems
        # For now, we'll just log the check
        self.get_logger().debug('Checking subsystem readiness...')

    def system_monitor(self):
        """Monitor system status and health"""
        # Publish current system status
        self.publish_system_status()

        # Check for system health issues
        if not self.system_status['safety_ok']:
            if self.current_state != SystemState.SAFETY_STOP:
                self.current_state = SystemState.SAFETY_STOP

        # Log system state periodically
        if self.get_clock().now().nanoseconds / 1e9 % 10 < 0.1:  # Every 10 seconds
            self.get_logger().info(f'System state: {self.current_state.value}, '
                                 f'Safety: {self.system_status["safety_ok"]}, '
                                 f'Queue size: {len(self.command_queue)}')

    def publish_system_status(self):
        """Publish system status"""
        status_msg = String()
        status_data = {
            'state': self.current_state.value,
            'previous_state': self.previous_state.value,
            'subsystem_status': self.system_status,
            'command_queue_size': len(self.command_queue),
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

def main():
    rclpy.init()
    integrator = SystemIntegratorNode()

    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        pass
    finally:
        integrator.destroy_node()
        rclpy.shutdown()
```

## Safety and Error Handling

Here's a safety monitor node that ensures system safety:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import LaserScan, BatteryState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import json
from typing import Dict, Any

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/safety/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety/status', 10)
        self.cmd_vel_stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.battery_sub = self.create_subscription(
            BatteryState,
            '/battery_state',
            self.battery_callback,
            10
        )

        self.system_status_sub = self.create_subscription(
            String,
            '/system/status',
            self.system_status_callback,
            10
        )

        # Safety parameters
        self.safety_params = {
            'min_obstacle_distance': 0.3,  # meters
            'min_battery_level': 10.0,     # percent
            'max_velocity': 0.5,           # m/s
            'check_frequency': 10.0        # Hz
        }

        # System state
        self.emergency_stop_active = False
        self.battery_level = 100.0
        self.system_healthy = True

        # Timer for safety checks
        self.safety_timer = self.create_timer(1.0/self.safety_params['check_frequency'], self.safety_check)

        self.get_logger().info('Safety Monitor initialized')

    def scan_callback(self, msg):
        """Check for obstacles"""
        if self.emergency_stop_active:
            return

        # Find minimum distance in scan
        if msg.ranges:
            min_distance = min([r for r in msg.ranges if msg.range_min <= r <= msg.range_max], default=float('inf'))

            if min_distance < self.safety_params['min_obstacle_distance']:
                self.trigger_emergency_stop(f'Obstacle too close: {min_distance:.2f}m < {self.safety_params["min_obstacle_distance"]}m')
                return

    def battery_callback(self, msg):
        """Monitor battery level"""
        self.battery_level = msg.percentage * 100.0

        if self.battery_level < self.safety_params['min_battery_level']:
            self.get_logger().warn(f'Low battery: {self.battery_level:.1f}%')
            # For low battery, we might not need emergency stop, just warning
            # unless it's critically low

    def system_status_callback(self, msg):
        """Monitor system status for errors"""
        try:
            status_data = json.loads(msg.data)
            state = status_data.get('state', 'unknown')

            # Check for error states
            if state == 'error':
                self.trigger_emergency_stop(f'System error detected: {state}')
                return

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid system status JSON: {msg.data}')

    def safety_check(self):
        """Perform regular safety checks"""
        if self.emergency_stop_active:
            # Keep publishing stop commands while emergency stop is active
            self.publish_stop_command()
            return

        # Additional safety checks can go here
        # For example, checking joint limits, temperature, etc.

        # Publish safety status
        safety_status = {
            'emergency_stop': self.emergency_stop_active,
            'battery_level': self.battery_level,
            'system_healthy': self.system_healthy,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

        status_msg = String()
        status_msg.data = json.dumps(safety_status)
        self.safety_status_pub.publish(status_msg)

    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.system_healthy = False

            # Publish emergency stop
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)

            # Also send stop command to base
            self.publish_stop_command()

            self.get_logger().error(f'EMERGENCY STOP TRIGGERED: {reason}')

    def publish_stop_command(self):
        """Publish stop command to base"""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        self.cmd_vel_stop_pub.publish(stop_cmd)

    def reset_emergency_stop(self):
        """Reset emergency stop (only if safe to do so)"""
        # This would typically require manual intervention or specific safe conditions
        self.emergency_stop_active = False
        self.system_healthy = True

        stop_msg = Bool()
        stop_msg.data = False
        self.emergency_stop_pub.publish(stop_msg)

        self.get_logger().info('Emergency stop reset')

def main():
    rclpy.init()
    safety_monitor = SafetyMonitorNode()

    try:
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        safety_monitor.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

Here's a system performance monitor to track and optimize system performance:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
import time
from typing import Dict, Any

class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        self.perf_pub = self.create_publisher(String, '/system/performance', 10)

        # Timer for performance monitoring
        self.perf_timer = self.create_timer(1.0, self.monitor_performance)

        self.get_logger().info('Performance Monitor initialized')

    def monitor_performance(self):
        """Monitor system performance"""
        performance_data = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'network_io': self.get_network_io(),
            'topics_status': self.get_topic_status()
        }

        perf_msg = String()
        perf_msg.data = json.dumps(performance_data)
        self.perf_pub.publish(perf_msg)

        # Log warnings for performance issues
        if performance_data['cpu_percent'] > 80:
            self.get_logger().warn(f'High CPU usage: {performance_data["cpu_percent"]}%')
        if performance_data['memory_percent'] > 80:
            self.get_logger().warn(f'High memory usage: {performance_data["memory_percent"]}%')

    def get_network_io(self) -> Dict[str, float]:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except:
            return {'error': 'Unable to get network stats'}

    def get_topic_status(self) -> Dict[str, Any]:
        """Get ROS topic status"""
        try:
            # Get list of topics
            topic_names_and_types = self.get_topic_names_and_types()
            return {
                'topic_count': len(topic_names_and_types),
                'topics': [name for name, _ in topic_names_and_types]
            }
        except:
            return {'error': 'Unable to get topic status'}

def main():
    rclpy.init()
    perf_monitor = PerformanceMonitorNode()

    try:
        rclpy.spin(perf_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        perf_monitor.destroy_node()
        rclpy.shutdown()
```

## Best Practices for System Integration

- **Modular Design**: Keep subsystems loosely coupled but well-integrated
- **Standardized Interfaces**: Use consistent message types and APIs across subsystems
- **Error Propagation**: Ensure errors are properly propagated and handled
- **Performance Monitoring**: Continuously monitor system performance
- **Safety First**: Implement multiple layers of safety checks
- **Graceful Degradation**: Systems should degrade gracefully when components fail
- **Testing**: Thoroughly test integration points between subsystems

## Next Steps

In the next chapter, we'll complete the capstone project by implementing the final autonomous humanoid demonstration that showcases all the integrated capabilities.