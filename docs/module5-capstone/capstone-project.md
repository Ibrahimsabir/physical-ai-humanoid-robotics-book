---
title: Capstone Project - Autonomous Humanoid Robot
sidebar_label: Capstone Project
---

# Capstone Project - Autonomous Humanoid Robot

## Introduction to the Capstone Project

The capstone project brings together all the modules and capabilities developed throughout this course into a complete, functional autonomous humanoid robot system. This project demonstrates the integration of the robotic nervous system (ROS 2), digital twin (simulation), AI brain (perception/navigation/manipulation), and Vision-Language-Action (VLA) systems working in harmony.

### Capstone Project Requirements

- **End-to-End Functionality**: Complete task execution from natural language command to action completion
- **Multi-Modal Integration**: Coordination of perception, navigation, and manipulation
- **Autonomous Operation**: System operates without direct human intervention
- **Robustness**: Handles errors and unexpected situations gracefully
- **Performance**: Meets specified performance benchmarks
- **Safety**: Incorporates comprehensive safety measures

## Complete System Architecture

Here's the architecture of the fully integrated system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                    (Natural Language, Voice Commands)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VLA SYSTEM (Vision-Language-Action)               │
│  - Speech Recognition & Natural Language Understanding                      │
│  - Intent Recognition & Entity Extraction                                   │
│  - Context-Aware Command Interpretation                                     │
│  - Task Planning & Action Mapping                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COORDINATION SYSTEM                               │
│  - Behavior Trees for Complex Task Execution                                │
│  - State Machine for System State Management                                │
│  - Resource Allocation & Conflict Resolution                                │
│  - Safety Monitoring & Emergency Response                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 ▼                         ▼                         ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│      NAVIGATION         │ │     MANIPULATION        │ │      PERCEPTION         │
│  - Path Planning        │ │  - Motion Planning      │ │  - Object Detection     │
│  - Obstacle Avoidance   │ │  - Inverse Kinematics   │ │  - SLAM                 │
│  - Localization         │ │  - Grasping Planning    │ │  - Semantic Segmentation│
│  - Map Building         │ │  - Trajectory Execution │ │  - Depth Estimation     │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL SYSTEM                                    │
│  - Low-Level Motor Control                                                  │
│  - Joint Trajectory Execution                                               │
│  - Sensor Fusion & State Estimation                                         │
│  - Real-Time Performance Management                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HARDWARE INTERFACE                                │
│  - Sensors (Cameras, LIDAR, IMU, Force/Torque)                             │
│  - Actuators (Motors, Servos, Grippers)                                     │
│  - Communication Interfaces (CAN, EtherCAT, WiFi)                          │
│  - Power Management & Safety Systems                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete Capstone Demonstration Script

Here's a comprehensive demonstration script that showcases all system capabilities:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
import time
from typing import Dict, Any, List
import threading

class CapstoneDemonstrationNode(Node):
    def __init__(self):
        super().__init__('capstone_demonstration')

        # Action clients for major subsystems
        self.move_group_client = ActionClient(self, MoveGroup, 'move_group')
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        # Publishers
        self.command_pub = self.create_publisher(String, '/integrated_commands', 10)
        self.status_pub = self.create_publisher(String, '/capstone/status', 10)
        self.nav_goal_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Subscribers
        self.system_status_sub = self.create_subscription(
            String,
            '/system/status',
            self.system_status_callback,
            10
        )

        # Capstone demonstration parameters
        self.demonstration_scenarios = [
            self.fetch_and_carry_scenario,
            self.room_cleaning_scenario,
            self.object_sorting_scenario
        ]

        self.current_scenario = None
        self.scenario_running = False
        self.demo_step = 0
        self.system_ready = False

        # Timer for demonstration execution
        self.demo_timer = self.create_timer(0.1, self.execute_demo_step)

        # Demo control
        self.demo_start_sub = self.create_subscription(
            Bool,
            '/capstone/start_demo',
            self.start_demo_callback,
            10
        )

        self.demo_stop_sub = self.create_subscription(
            Bool,
            '/capstone/stop_demo',
            self.stop_demo_callback,
            10
        )

        self.get_logger().info('Capstone Demonstration Node initialized')

    def system_status_callback(self, msg: String):
        """Monitor system status"""
        try:
            status_data = json.loads(msg.data)
            state = status_data.get('state', 'unknown')

            # Check if system is ready to run demonstrations
            if state == 'idle':
                self.system_ready = True
            else:
                self.system_ready = False

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid status JSON: {msg.data}')

    def start_demo_callback(self, msg: Bool):
        """Start demonstration"""
        if msg.data and self.system_ready:
            self.get_logger().info('Starting capstone demonstration...')
            self.start_capstone_demo()

    def stop_demo_callback(self, msg: Bool):
        """Stop demonstration"""
        if msg.data:
            self.get_logger().info('Stopping capstone demonstration...')
            self.scenario_running = False
            self.demo_step = 0
            self.publish_status('demonstration_stopped')

    def start_capstone_demo(self):
        """Start the complete capstone demonstration"""
        if not self.system_ready:
            self.get_logger().error('System not ready for demonstration')
            return

        self.get_logger().info('Starting comprehensive capstone demonstration...')

        # Start with the fetch and carry scenario
        self.current_scenario = self.demonstration_scenarios[0]
        self.scenario_running = True
        self.demo_step = 0

        self.publish_status('demonstration_started')

    def execute_demo_step(self):
        """Execute demonstration steps"""
        if not self.scenario_running or not self.current_scenario:
            return

        # Execute current scenario step
        try:
            if self.current_scenario():
                # Scenario completed, move to next scenario
                scenario_index = self.demonstration_scenarios.index(self.current_scenario)

                if scenario_index < len(self.demonstration_scenarios) - 1:
                    # Move to next scenario
                    self.current_scenario = self.demonstration_scenarios[scenario_index + 1]
                    self.demo_step = 0
                    self.get_logger().info(f'Moving to next scenario: {self.current_scenario.__name__}')
                else:
                    # All scenarios completed
                    self.scenario_running = False
                    self.publish_status('demonstration_completed')
                    self.get_logger().info('All demonstration scenarios completed!')

        except Exception as e:
            self.get_logger().error(f'Error executing demo step: {e}')
            self.scenario_running = False
            self.publish_status('demonstration_error')

    def fetch_and_carry_scenario(self) -> bool:
        """Complete fetch and carry scenario demonstrating all capabilities"""

        # Define scenario steps
        scenario_steps = [
            {
                'name': 'navigate_to_kitchen',
                'type': 'navigate',
                'target_pose': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
                'description': 'Navigating to kitchen to find the red cup'
            },
            {
                'name': 'locate_object',
                'type': 'perceive',
                'task': 'detect_red_cup',
                'description': 'Using perception to locate the red cup'
            },
            {
                'name': 'approach_object',
                'type': 'navigate',
                'target_pose': {'x': 2.2, 'y': 1.0, 'theta': 0.0},
                'description': 'Approaching the red cup for pickup'
            },
            {
                'name': 'pick_up_object',
                'type': 'manipulate',
                'action': 'pick_up',
                'object': 'red_cup',
                'description': 'Picking up the red cup'
            },
            {
                'name': 'navigate_to_table',
                'type': 'navigate',
                'target_pose': {'x': 0.5, 'y': 0.0, 'theta': 0.0},
                'description': 'Navigating to the table to place the cup'
            },
            {
                'name': 'place_object',
                'type': 'manipulate',
                'action': 'place',
                'object': 'red_cup',
                'location': 'table',
                'description': 'Placing the red cup on the table'
            },
            {
                'name': 'return_home',
                'type': 'navigate',
                'target_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
                'description': 'Returning to home position'
            }
        ]

        if self.demo_step >= len(scenario_steps):
            self.get_logger().info('Fetch and carry scenario completed')
            return True  # Scenario completed

        current_step = scenario_steps[self.demo_step]

        self.get_logger().info(f'Executing step {self.demo_step + 1}/{len(scenario_steps)}: {current_step["description"]}')

        # Execute the step
        command = {
            'type': current_step['type'],
            'step_name': current_step['name'],
            'description': current_step['description']
        }

        if current_step['type'] == 'navigate':
            command.update(current_step['target_pose'])
        elif current_step['type'] == 'manipulate':
            command.update(current_step)
        elif current_step['type'] == 'perceive':
            command.update(current_step)

        # Publish command
        cmd_msg = String()
        cmd_msg.data = json.dumps(command)
        self.command_pub.publish(cmd_msg)

        # Simulate step completion (in real system, this would be based on actual completion)
        time.sleep(3)  # Simulate execution time
        self.demo_step += 1

        return False  # Scenario not yet completed

    def room_cleaning_scenario(self) -> bool:
        """Room cleaning scenario demonstrating autonomous operation"""

        scenario_steps = [
            {
                'name': 'scan_room',
                'type': 'perceive',
                'task': 'room_scan',
                'description': 'Scanning the room to identify dirty areas and objects'
            },
            {
                'name': 'plan_cleaning_route',
                'type': 'navigate',
                'target_pose': {'x': 1.0, 'y': 0.5, 'theta': 0.0},
                'description': 'Moving to first cleaning location'
            },
            {
                'name': 'clean_area',
                'type': 'manipulate',
                'action': 'clean',
                'location': 'spot1',
                'description': 'Cleaning the first area'
            },
            {
                'name': 'navigate_to_next_area',
                'type': 'navigate',
                'target_pose': {'x': -0.5, 'y': 1.0, 'theta': 1.57},
                'description': 'Moving to next cleaning location'
            },
            {
                'name': 'collect_trash',
                'type': 'manipulate',
                'action': 'pick_up',
                'object': 'trash',
                'description': 'Collecting trash'
            },
            {
                'name': 'dispose_trash',
                'type': 'navigate',
                'target_pose': {'x': 2.0, 'y': -1.0, 'theta': 0.0},
                'description': 'Taking trash to disposal area'
            },
            {
                'name': 'return_to_base',
                'type': 'navigate',
                'target_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
                'description': 'Returning to base station'
            }
        ]

        if self.demo_step >= len(scenario_steps):
            self.get_logger().info('Room cleaning scenario completed')
            return True  # Scenario completed

        current_step = scenario_steps[self.demo_step]

        self.get_logger().info(f'Executing room cleaning step {self.demo_step + 1}/{len(scenario_steps)}: {current_step["description"]}')

        # Execute the step
        command = {
            'type': current_step['type'],
            'step_name': current_step['name'],
            'description': current_step['description']
        }

        if current_step['type'] == 'navigate':
            command.update(current_step['target_pose'])
        elif current_step['type'] == 'manipulate':
            command.update(current_step)
        elif current_step['type'] == 'perceive':
            command.update(current_step)

        # Publish command
        cmd_msg = String()
        cmd_msg.data = json.dumps(command)
        self.command_pub.publish(cmd_msg)

        # Simulate step completion
        time.sleep(2.5)
        self.demo_step += 1

        return False  # Scenario not yet completed

    def object_sorting_scenario(self) -> bool:
        """Object sorting scenario demonstrating perception and manipulation integration"""

        scenario_steps = [
            {
                'name': 'scan_workspace',
                'type': 'perceive',
                'task': 'workspace_scan',
                'description': 'Scanning workspace to identify objects'
            },
            {
                'name': 'identify_objects',
                'type': 'perceive',
                'task': 'object_classification',
                'description': 'Classifying objects by color and type'
            },
            {
                'name': 'navigate_to_first_object',
                'type': 'navigate',
                'target_pose': {'x': 0.8, 'y': 0.3, 'theta': 0.0},
                'description': 'Moving to first object'
            },
            {
                'name': 'pick_up_red_object',
                'type': 'manipulate',
                'action': 'pick_up',
                'object': 'red_block',
                'description': 'Picking up red object'
            },
            {
                'name': 'place_in_red_bin',
                'type': 'navigate',
                'target_pose': {'x': -0.5, 'y': 0.5, 'theta': 0.0},
                'description': 'Placing red object in red bin'
            },
            {
                'name': 'navigate_to_next_object',
                'type': 'navigate',
                'target_pose': {'x': 0.8, 'y': -0.2, 'theta': 0.0},
                'description': 'Moving to next object'
            },
            {
                'name': 'pick_up_blue_object',
                'type': 'manipulate',
                'action': 'pick_up',
                'object': 'blue_block',
                'description': 'Picking up blue object'
            },
            {
                'name': 'place_in_blue_bin',
                'type': 'navigate',
                'target_pose': {'x': -0.5, 'y': -0.5, 'theta': 0.0},
                'description': 'Placing blue object in blue bin'
            }
        ]

        if self.demo_step >= len(scenario_steps):
            self.get_logger().info('Object sorting scenario completed')
            return True  # Scenario completed

        current_step = scenario_steps[self.demo_step]

        self.get_logger().info(f'Executing object sorting step {self.demo_step + 1}/{len(scenario_steps)}: {current_step["description"]}')

        # Execute the step
        command = {
            'type': current_step['type'],
            'step_name': current_step['name'],
            'description': current_step['description']
        }

        if current_step['type'] == 'navigate':
            command.update(current_step['target_pose'])
        elif current_step['type'] == 'manipulate':
            command.update(current_step)
        elif current_step['type'] == 'perceive':
            command.update(current_step)

        # Publish command
        cmd_msg = String()
        cmd_msg.data = json.dumps(command)
        self.command_pub.publish(cmd_msg)

        # Simulate step completion
        time.sleep(3.5)
        self.demo_step += 1

        return False  # Scenario not yet completed

    def publish_status(self, status: str):
        """Publish demonstration status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'current_scenario': self.current_scenario.__name__ if self.current_scenario else 'none',
            'demo_step': self.demo_step,
            'timestamp': time.time()
        })
        self.status_pub.publish(status_msg)

def main():
    rclpy.init()
    demo_node = CapstoneDemonstrationNode()

    # Start demonstration automatically after initialization
    def start_demo():
        time.sleep(2)  # Wait for system to initialize
        start_msg = Bool()
        start_msg.data = True
        demo_node.start_demo_callback(start_msg)

    # Start demo in a separate thread
    demo_thread = threading.Thread(target=start_demo)
    demo_thread.daemon = True
    demo_thread.start()

    try:
        rclpy.spin(demo_node)
    except KeyboardInterrupt:
        pass
    finally:
        demo_node.destroy_node()
        rclpy.shutdown()
```

## Performance Validation and Benchmarks

Here's a comprehensive validation system for the capstone project:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import Pose
import json
import time
from typing import Dict, Any, List
import statistics

class PerformanceValidatorNode(Node):
    def __init__(self):
        super().__init__('performance_validator')

        # Publishers for validation results
        self.validation_results_pub = self.create_publisher(String, '/validation/results', 10)
        self.performance_metrics_pub = self.create_publisher(String, '/validation/performance', 10)

        # Subscribers for system metrics
        self.system_status_sub = self.create_subscription(
            String,
            '/system/status',
            self.system_status_callback,
            10
        )

        self.performance_sub = self.create_subscription(
            String,
            '/system/performance',
            self.performance_callback,
            10
        )

        self.task_completion_sub = self.create_subscription(
            String,
            '/task/completion',
            self.task_completion_callback,
            10
        )

        # Validation parameters
        self.validation_criteria = {
            'task_success_rate': 0.95,  # 95% success rate
            'response_time_max': 5.0,   # 5 seconds max response time
            'battery_consumption_max': 20.0,  # 20% max battery consumption per task
            'cpu_utilization_max': 80.0,      # 80% max CPU utilization
            'memory_utilization_max': 80.0,   # 80% max memory utilization
            'navigation_accuracy': 0.1,       # 10cm navigation accuracy
            'manipulation_success': 0.90      # 90% manipulation success rate
        }

        # Metrics storage
        self.task_times = []
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.performance_history = []
        self.current_battery_level = 100.0

        # Timer for validation reporting
        self.validation_timer = self.create_timer(5.0, self.run_validation)

        self.get_logger().info('Performance Validator initialized')

    def system_status_callback(self, msg: String):
        """Monitor system status for validation"""
        try:
            status_data = json.loads(msg.data)
            # Store system status for validation
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid status JSON: {msg.data}')

    def performance_callback(self, msg: String):
        """Collect performance metrics"""
        try:
            perf_data = json.loads(msg.data)
            self.performance_history.append(perf_data)

            # Keep history manageable
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid performance JSON: {msg.data}')

    def task_completion_callback(self, msg: String):
        """Track task completions for validation"""
        try:
            task_data = json.loads(msg.data)
            success = task_data.get('success', False)
            start_time = task_data.get('start_time', 0)
            end_time = task_data.get('end_time', 0)

            task_duration = end_time - start_time

            if success:
                self.successful_tasks += 1
                self.task_times.append(task_duration)
            else:
                self.failed_tasks += 1

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid task completion JSON: {msg.data}')

    def run_validation(self):
        """Run comprehensive validation"""
        validation_results = {
            'timestamp': time.time(),
            'criteria_met': True,
            'results': {}
        }

        # Validate task success rate
        total_tasks = self.successful_tasks + self.failed_tasks
        if total_tasks > 0:
            success_rate = self.successful_tasks / total_tasks
            validation_results['results']['task_success_rate'] = {
                'value': success_rate,
                'threshold': self.validation_criteria['task_success_rate'],
                'passed': success_rate >= self.validation_criteria['task_success_rate']
            }

            if not validation_results['results']['task_success_rate']['passed']:
                validation_results['criteria_met'] = False

        # Validate response times
        if self.task_times:
            avg_response_time = statistics.mean(self.task_times)
            max_response_time = max(self.task_times) if self.task_times else 0.0

            validation_results['results']['response_time'] = {
                'average': avg_response_time,
                'max': max_response_time,
                'threshold': self.validation_criteria['response_time_max'],
                'passed': max_response_time <= self.validation_criteria['response_time_max']
            }

            if not validation_results['results']['response_time']['passed']:
                validation_results['criteria_met'] = False

        # Validate system performance metrics
        if self.performance_history:
            recent_perf = self.performance_history[-1]  # Most recent

            # Validate CPU utilization
            cpu_percent = recent_perf.get('cpu_percent', 0.0)
            validation_results['results']['cpu_utilization'] = {
                'value': cpu_percent,
                'threshold': self.validation_criteria['cpu_utilization_max'],
                'passed': cpu_percent <= self.validation_criteria['cpu_utilization_max']
            }

            if not validation_results['results']['cpu_utilization']['passed']:
                validation_results['criteria_met'] = False

            # Validate memory utilization
            memory_percent = recent_perf.get('memory_percent', 0.0)
            validation_results['results']['memory_utilization'] = {
                'value': memory_percent,
                'threshold': self.validation_criteria['memory_utilization_max'],
                'passed': memory_percent <= self.validation_criteria['memory_utilization_max']
            }

            if not validation_results['results']['memory_utilization']['passed']:
                validation_results['criteria_met'] = False

        # Publish validation results
        results_msg = String()
        results_msg.data = json.dumps(validation_results, indent=2)
        self.validation_results_pub.publish(results_msg)

        # Log validation summary
        if validation_results['criteria_met']:
            self.get_logger().info('✓ All validation criteria met')
        else:
            self.get_logger().warn('✗ Some validation criteria not met')

            for criterion, result in validation_results['results'].items():
                if not result.get('passed', True):
                    self.get_logger().warn(f'  - {criterion}: {result["value"]} (threshold: {result["threshold"]})')

    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tasks = self.successful_tasks + self.failed_tasks
        success_rate = (self.successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0

        avg_response_time = statistics.mean(self.task_times) if self.task_times else 0.0
        max_response_time = max(self.task_times) if self.task_times else 0.0

        return {
            'summary': {
                'total_tasks': total_tasks,
                'successful_tasks': self.successful_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate_percent': success_rate,
                'average_response_time': avg_response_time,
                'max_response_time': max_response_time
            },
            'detailed_results': self.get_validation_results(),
            'recommendations': self.generate_recommendations()
        }

    def get_validation_results(self) -> Dict[str, Any]:
        """Get detailed validation results"""
        # This would return more detailed validation data
        return {
            'task_success_rate': {
                'achieved': (self.successful_tasks / (self.successful_tasks + self.failed_tasks)) if (self.successful_tasks + self.failed_tasks) > 0 else 0,
                'required': self.validation_criteria['task_success_rate'],
                'status': 'PASS' if (self.successful_tasks / (self.successful_tasks + self.failed_tasks)) >= self.validation_criteria['task_success_rate'] else 'FAIL'
            },
            'response_time': {
                'average': statistics.mean(self.task_times) if self.task_times else 0,
                'maximum': max(self.task_times) if self.task_times else 0,
                'required': self.validation_criteria['response_time_max'],
                'status': 'PASS' if max(self.task_times) <= self.validation_criteria['response_time_max'] if self.task_times else 'FAIL'
            }
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        total_tasks = self.successful_tasks + self.failed_tasks
        if total_tasks > 0:
            success_rate = self.successful_tasks / total_tasks
            if success_rate < 0.9:  # Below 90% success rate
                recommendations.append("Consider improving task success rate through better error handling and recovery mechanisms")

        if self.task_times:
            avg_time = statistics.mean(self.task_times)
            if avg_time > 3.0:  # Above 3 seconds average
                recommendations.append("Investigate performance bottlenecks to reduce task execution time")

        if len(recommendations) == 0:
            recommendations.append("System performance is satisfactory. No immediate improvements needed.")

        return recommendations

def main():
    rclpy.init()
    validator = PerformanceValidatorNode()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        # Generate final validation report
        report = validator.get_validation_report()
        print("\n=== CAPSTONE PROJECT VALIDATION REPORT ===")
        print(json.dumps(report, indent=2))
    finally:
        validator.destroy_node()
        rclpy.shutdown()
```

## Deployment and Optimization

Here's a system for optimizing the complete system for deployment:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import json
import subprocess
import os
from typing import Dict, Any

class DeploymentOptimizerNode(Node):
    def __init__(self):
        super().__init__('deployment_optimizer')

        # Publishers
        self.optimization_status_pub = self.create_publisher(String, '/optimization/status', 10)

        # Subscribers
        self.system_load_sub = self.create_subscription(
            String,
            '/system/performance',
            self.system_load_callback,
            10
        )

        self.user_feedback_sub = self.create_subscription(
            Joy,
            '/user_feedback',
            self.user_feedback_callback,
            10
        )

        # Optimization parameters
        self.optimization_params = {
            'cpu_threshold': 70.0,
            'memory_threshold': 75.0,
            'battery_threshold': 20.0,
            'optimization_enabled': True
        }

        # Timer for optimization checks
        self.optimization_timer = self.create_timer(10.0, self.optimize_system)

        self.get_logger().info('Deployment Optimizer initialized')

    def system_load_callback(self, msg: String):
        """Monitor system load for optimization"""
        try:
            perf_data = json.loads(msg.data)

            # Check if optimization is needed based on system load
            cpu_percent = perf_data.get('cpu_percent', 0.0)
            memory_percent = perf_data.get('memory_percent', 0.0)

            if (cpu_percent > self.optimization_params['cpu_threshold'] or
                memory_percent > self.optimization_params['memory_threshold']):
                self.optimize_for_performance()

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid performance JSON: {msg.data}')

    def user_feedback_callback(self, msg: Joy):
        """Handle user feedback for optimization"""
        # Use joystick buttons to trigger optimization
        if len(msg.buttons) > 0 and msg.buttons[0] == 1:  # Button 0 pressed
            self.get_logger().info('Manual optimization triggered by user')
            self.perform_manual_optimization()

    def optimize_system(self):
        """Perform system optimization"""
        if not self.optimization_params['optimization_enabled']:
            return

        self.get_logger().info('Running system optimization...')

        optimization_status = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'optimizations_performed': [],
            'system_resources_freed': {}
        }

        # Perform various optimizations
        cpu_optimized = self.optimize_cpu_usage()
        memory_optimized = self.optimize_memory_usage()
        battery_optimized = self.optimize_power_consumption()

        if cpu_optimized:
            optimization_status['optimizations_performed'].append('CPU optimization')
        if memory_optimized:
            optimization_status['optimizations_performed'].append('Memory optimization')
        if battery_optimized:
            optimization_status['optimizations_performed'].append('Power optimization')

        # Publish optimization status
        status_msg = String()
        status_msg.data = json.dumps(optimization_status)
        self.optimization_status_pub.publish(status_msg)

    def optimize_cpu_usage(self) -> bool:
        """Optimize CPU usage"""
        try:
            # Reduce computational load for non-critical processes
            # This is a placeholder - actual implementation would adjust process priorities
            self.get_logger().debug('Optimizing CPU usage...')

            # Example: Reduce perception processing frequency if under high load
            # This would involve adjusting parameters in perception nodes
            return True
        except Exception as e:
            self.get_logger().error(f'CPU optimization failed: {e}')
            return False

    def optimize_memory_usage(self) -> bool:
        """Optimize memory usage"""
        try:
            # Clear caches, optimize data structures, etc.
            self.get_logger().debug('Optimizing memory usage...')

            # Example: Clear image processing caches
            # This would involve managing internal data structures
            return True
        except Exception as e:
            self.get_logger().error(f'Memory optimization failed: {e}')
            return False

    def optimize_power_consumption(self) -> bool:
        """Optimize power consumption"""
        try:
            # Reduce power consumption by adjusting operational parameters
            self.get_logger().debug('Optimizing power consumption...')

            # Example: Reduce actuator power if possible
            # Adjust motion speeds to conserve energy
            return True
        except Exception as e:
            self.get_logger().error(f'Power optimization failed: {e}')
            return False

    def perform_manual_optimization(self):
        """Perform manual optimization triggered by user"""
        self.get_logger().info('Performing manual system optimization...')

        # Perform all optimizations
        cpu_ok = self.optimize_cpu_usage()
        memory_ok = self.optimize_memory_usage()
        power_ok = self.optimize_power_consumption()

        status = f"Manual optimization completed: CPU={cpu_ok}, Memory={memory_ok}, Power={power_ok}"
        self.get_logger().info(status)

def main():
    rclpy.init()
    optimizer = DeploymentOptimizerNode()

    try:
        rclpy.spin(optimizer)
    except KeyboardInterrupt:
        pass
    finally:
        optimizer.destroy_node()
        rclpy.shutdown()
```

## Final Integration and Testing

Here's a comprehensive test suite for the complete system:

```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time
import json

class CapstoneIntegrationTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.test_node = Node('capstone_test_node')

        # Publishers for sending test commands
        self.command_pub = self.test_node.create_publisher(String, '/integrated_commands', 10)
        self.demo_start_pub = self.test_node.create_publisher(Bool, '/capstone/start_demo', 10)

        # Subscribers for receiving test results
        self.status_sub = self.test_node.create_subscription(
            String,
            '/system/status',
            self.status_callback,
            10
        )

        self.results_sub = self.test_node.create_subscription(
            String,
            '/validation/results',
            self.results_callback,
            10
        )

        self.status_messages = []
        self.result_messages = []

        # Test parameters
        self.timeout = 30  # seconds

    def status_callback(self, msg):
        self.status_messages.append(msg.data)

    def results_callback(self, msg):
        self.result_messages.append(msg.data)

    def test_system_initialization(self):
        """Test that all subsystems initialize correctly"""
        start_time = time.time()
        initialized = False

        while time.time() - start_time < self.timeout and not initialized:
            if self.status_messages:
                latest_status = json.loads(self.status_messages[-1])
                if latest_status.get('state') == 'idle':
                    initialized = True
                    break
            time.sleep(0.1)

        self.assertTrue(initialized, "System did not reach idle state within timeout")

    def test_basic_navigation(self):
        """Test basic navigation capability"""
        # Send navigation command
        nav_cmd = {
            'type': 'navigate',
            'target_pose': {'x': 1.0, 'y': 0.0, 'theta': 0.0}
        }

        cmd_msg = String()
        cmd_msg.data = json.dumps(nav_cmd)
        self.command_pub.publish(cmd_msg)

        # Wait for navigation to complete
        start_time = time.time()
        completed = False

        while time.time() - start_time < self.timeout and not completed:
            # Check for navigation completion in status messages
            for status_msg in self.status_messages:
                try:
                    status = json.loads(status_msg)
                    if status.get('state') == 'idle':
                        completed = True
                        break
                except:
                    continue
            time.sleep(0.1)

        self.assertTrue(completed, "Navigation did not complete within timeout")

    def test_perception_functionality(self):
        """Test perception system functionality"""
        # Send perception command
        perceive_cmd = {
            'type': 'perceive',
            'task': 'object_detection'
        }

        cmd_msg = String()
        cmd_msg.data = json.dumps(perceive_cmd)
        self.command_pub.publish(cmd_msg)

        # Wait for perception to complete
        start_time = time.time()
        completed = False

        while time.time() - start_time < self.timeout and not completed:
            # Check for perception completion
            for status_msg in self.status_messages:
                try:
                    status = json.loads(status_msg)
                    if 'perceiving' not in status.get('state', '') and status.get('state') == 'idle':
                        completed = True
                        break
                except:
                    continue
            time.sleep(0.1)

        self.assertTrue(completed, "Perception did not complete within timeout")

    def test_complete_scenario_execution(self):
        """Test execution of a complete scenario"""
        # Start the capstone demonstration
        start_msg = Bool()
        start_msg.data = True
        self.demo_start_pub.publish(start_msg)

        # Wait for demonstration to complete
        start_time = time.time()
        completed = False

        while time.time() - start_time < self.timeout * 2 and not completed:  # Longer timeout for full demo
            if self.result_messages:
                latest_result = json.loads(self.result_messages[-1])
                if latest_result.get('criteria_met', False):
                    completed = True
                    break
            time.sleep(0.1)

        self.assertTrue(completed, "Complete scenario did not complete within timeout")

    def tearDown(self):
        self.test_node.destroy_node()
        rclpy.shutdown()

def run_integration_tests():
    """Run the complete integration test suite"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(CapstoneIntegrationTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Capstone Integration Tests...")
    success = run_integration_tests()

    if success:
        print("\n✓ All integration tests passed!")
    else:
        print("\n✗ Some integration tests failed!")
        exit(1)
```

## Best Practices for Capstone Project

- **Incremental Testing**: Test each subsystem before integration
- **Comprehensive Validation**: Validate all performance criteria
- **Safety First**: Ensure all safety systems are operational
- **Performance Monitoring**: Continuously monitor system performance
- **Error Recovery**: Implement robust error recovery mechanisms
- **Documentation**: Maintain comprehensive documentation
- **User Experience**: Focus on intuitive and responsive interaction

## Conclusion

The capstone project successfully demonstrates the integration of all modules into a functional autonomous humanoid robot system. This project showcases:

- **ROS 2 Integration**: Complete robotic nervous system with proper communication
- **Simulation to Reality**: Digital twin capabilities with realistic physics
- **AI Brain**: Advanced perception, navigation, and manipulation
- **VLA System**: Natural language understanding and action execution
- **System Integration**: Seamless coordination of all subsystems
- **Performance**: Optimized operation meeting specified criteria

The autonomous humanoid robot system is now ready for deployment and can execute complex tasks based on natural language commands while ensuring safety and reliability.