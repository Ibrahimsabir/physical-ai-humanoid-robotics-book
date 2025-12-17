---
title: Action Mapping for VLA Systems
sidebar_label: Action Mapping
---

# Action Mapping for VLA Systems

## Introduction to Action Mapping

Action mapping is the critical component that converts natural language commands into executable robot actions. This involves understanding the intent behind user commands and translating them into sequences of low-level robot behaviors. In Vision-Language-Action (VLA) systems, this mapping must consider the robot's perception of the environment and its available capabilities.

### Key Components of Action Mapping

- **Intent Recognition**: Understanding what the user wants to achieve
- **Action Planning**: Determining the sequence of actions needed
- **Context Management**: Maintaining state across multiple commands
- **Action Execution**: Converting high-level actions to low-level commands
- **Feedback Generation**: Providing status updates to the user

## Creating an Action Mapping System

Here's a comprehensive action mapping system that connects language understanding to robot execution:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import MoveGroupGoal
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
import re
from typing import Dict, List, Optional, Any

class ActionMappingNode(Node):
    def __init__(self):
        super().__init__('action_mapping')

        # Action clients for robot execution
        self.move_group_client = ActionClient(self, MoveGroup, 'move_group')
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        # Subscribers for language commands
        self.command_sub = self.create_subscription(
            String,
            '/language_commands',
            self.command_callback,
            10
        )

        # Publishers for robot actions
        self.nav_goal_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        self.status_pub = self.create_publisher(String, '/action_status', 10)

        # Context management
        self.context = {
            'current_location': 'home',
            'carrying_object': None,
            'last_action': None,
            'action_history': []
        }

        # Define action mappings
        self.action_definitions = {
            'navigate': {
                'required_params': ['target_location'],
                'execution_method': self.execute_navigation
            },
            'pick_up': {
                'required_params': ['object_name'],
                'execution_method': self.execute_pickup
            },
            'place': {
                'required_params': ['object_name', 'target_location'],
                'execution_method': self.execute_place
            },
            'drop': {
                'required_params': ['object_name'],
                'execution_method': self.execute_drop
            },
            'greet': {
                'required_params': [],
                'execution_method': self.execute_greet
            }
        }

        self.get_logger().info('Action Mapping node initialized')

    def command_callback(self, msg):
        """Process incoming language commands"""
        try:
            # Parse the command (could be JSON or simple string)
            command_data = self.parse_command(msg.data)

            if command_data:
                self.execute_command(command_data)
            else:
                self.publish_status(f"Could not parse command: {msg.data}")

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            self.publish_status(f"Error processing command: {str(e)}")

    def parse_command(self, command_str: str) -> Optional[Dict[str, Any]]:
        """Parse a command string into structured data"""
        # Simple parsing - in a real system, this would be more sophisticated
        command_str = command_str.lower().strip()

        # Try to identify action and parameters
        if 'go to' in command_str or 'navigate to' in command_str:
            # Extract location
            location_match = re.search(r'(?:go to|navigate to) (.+)', command_str)
            if location_match:
                return {
                    'action': 'navigate',
                    'target_location': location_match.group(1).strip(),
                    'confidence': 0.9
                }
        elif 'pick up' in command_str or 'grasp' in command_str:
            # Extract object
            obj_match = re.search(r'(?:pick up|grasp) (.+)', command_str)
            if obj_match:
                return {
                    'action': 'pick_up',
                    'object_name': obj_match.group(1).strip(),
                    'confidence': 0.9
                }
        elif 'place' in command_str or 'put' in command_str:
            # Extract object and location
            match = re.search(r'(?:place|put) (.+?) (?:on|at|in) (.+)', command_str)
            if match:
                return {
                    'action': 'place',
                    'object_name': match.group(1).strip(),
                    'target_location': match.group(2).strip(),
                    'confidence': 0.9
                }
        elif 'drop' in command_str:
            obj_match = re.search(r'drop (.+)', command_str)
            if obj_match:
                return {
                    'action': 'drop',
                    'object_name': obj_match.group(1).strip(),
                    'confidence': 0.9
                }
        elif 'hello' in command_str or 'hi' in command_str:
            return {
                'action': 'greet',
                'confidence': 0.9
            }

        # If no pattern matches, return None
        return None

    def execute_command(self, command_data: Dict[str, Any]):
        """Execute a parsed command"""
        action = command_data['action']
        confidence = command_data.get('confidence', 0.0)

        # Check if we're confident enough to execute
        if confidence < 0.7:
            self.publish_status(f"Command confidence too low: {confidence}")
            return

        # Check if action is defined
        if action not in self.action_definitions:
            self.publish_status(f"Unknown action: {action}")
            return

        # Get action definition
        action_def = self.action_definitions[action]

        # Check required parameters
        required_params = action_def['required_params']
        missing_params = []

        for param in required_params:
            if param not in command_data:
                missing_params.append(param)

        if missing_params:
            self.publish_status(f"Missing required parameters for {action}: {missing_params}")
            return

        # Update context
        self.context['last_action'] = action
        self.context['action_history'].append(command_data)

        # Execute the action
        execution_method = action_def['execution_method']
        try:
            result = execution_method(command_data)
            if result:
                self.publish_status(f"Successfully executed {action}")
            else:
                self.publish_status(f"Failed to execute {action}")
        except Exception as e:
            self.get_logger().error(f'Error executing {action}: {e}')
            self.publish_status(f"Error executing {action}: {str(e)}")

    def execute_navigation(self, command_data: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        target_location = command_data['target_location']

        # Map location name to coordinates
        location_map = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 2.0, 0.0),
            'bedroom': (-1.0, -1.0, 0.0),
            'office': (1.5, -0.5, 0.0),
            'home': (0.0, 0.0, 0.0)
        }

        if target_location in location_map:
            x, y, theta = location_map[target_location]

            # Create navigation goal
            goal_pose = Pose()
            goal_pose.position.x = x
            goal_pose.position.y = y
            goal_pose.position.z = 0.0

            # Convert theta to quaternion
            import math
            goal_pose.orientation.z = math.sin(theta / 2.0)
            goal_pose.orientation.w = math.cos(theta / 2.0)

            # Publish navigation goal
            self.nav_goal_pub.publish(goal_pose)

            # Update context
            self.context['current_location'] = target_location

            self.get_logger().info(f'Navigating to {target_location} at ({x}, {y})')
            return True
        else:
            self.get_logger().warn(f'Unknown location: {target_location}')
            return False

    def execute_pickup(self, command_data: Dict[str, Any]) -> bool:
        """Execute pickup action"""
        object_name = command_data['object_name']

        # Check if we're already carrying something
        if self.context['carrying_object']:
            self.get_logger().warn(f'Already carrying {self.context["carrying_object"]}')
            return False

        # In a real system, you would:
        # 1. Locate the object using perception
        # 2. Plan a grasp
        # 3. Execute the grasp

        # For simulation, just update context
        self.context['carrying_object'] = object_name

        self.get_logger().info(f'Picking up {object_name}')
        return True

    def execute_place(self, command_data: Dict[str, Any]) -> bool:
        """Execute place action"""
        object_name = command_data['object_name']
        target_location = command_data['target_location']

        # Check if we're carrying the right object
        if self.context['carrying_object'] != object_name:
            self.get_logger().warn(f'Not carrying {object_name}, currently carrying {self.context["carrying_object"]}')
            return False

        # In a real system, you would:
        # 1. Navigate to target location if needed
        # 2. Position robot appropriately
        # 3. Execute placing motion

        # Update context
        self.context['carrying_object'] = None

        self.get_logger().info(f'Placing {object_name} at {target_location}')
        return True

    def execute_drop(self, command_data: Dict[str, Any]) -> bool:
        """Execute drop action"""
        object_name = command_data['object_name']

        # Check if we're carrying the object
        if self.context['carrying_object'] != object_name:
            self.get_logger().warn(f'Not carrying {object_name}')
            return False

        # Update context
        self.context['carrying_object'] = None

        self.get_logger().info(f'Dropping {object_name}')
        return True

    def execute_greet(self, command_data: Dict[str, Any]) -> bool:
        """Execute greeting action"""
        self.get_logger().info('Executing greeting')
        self.publish_status('Hello! How can I help you today?')
        return True

    def publish_status(self, status: str):
        """Publish action status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Action status: {status}')

def main():
    rclpy.init()
    action_mapper = ActionMappingNode()

    try:
        rclpy.spin(action_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        action_mapper.destroy_node()
        rclpy.shutdown()
```

## Context Management System

Here's an advanced context management system that maintains conversation state:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
import time
from typing import Dict, Any, Optional

class ContextManagerNode(Node):
    def __init__(self):
        super().__init__('context_manager')

        # Subscribe to various inputs
        self.command_sub = self.create_subscription(
            String,
            '/user_commands',
            self.command_callback,
            10
        )

        self.context_sub = self.create_subscription(
            String,
            '/robot_state',
            self.state_callback,
            10
        )

        self.response_pub = self.create_publisher(String, '/contextual_response', 10)

        # Initialize context
        self.context = {
            'conversation_id': 0,
            'turn_number': 0,
            'user_id': 'default_user',
            'current_task': None,
            'task_steps': [],
            'completed_tasks': [],
            'robot_state': {
                'location': 'home',
                'carrying': None,
                'battery_level': 100,
                'gripper_state': 'open'
            },
            'object_locations': {},
            'user_preferences': {},
            'conversation_history': [],
            'entities_mentioned': set(),
            'last_interaction_time': time.time()
        }

        # Timer to clean up old context
        self.context_timer = self.create_timer(300.0, self.cleanup_context)  # 5 minutes

        self.get_logger().info('Context Manager node initialized')

    def command_callback(self, msg):
        """Process user command and update context"""
        command = msg.data
        timestamp = time.time()

        # Add to conversation history
        self.context['conversation_history'].append({
            'type': 'user_input',
            'content': command,
            'timestamp': timestamp,
            'turn': self.context['turn_number']
        })

        # Update turn number
        self.context['turn_number'] += 1
        self.context['last_interaction_time'] = timestamp

        # Extract entities from command
        entities = self.extract_entities(command)
        self.context['entities_mentioned'].update(entities)

        # Process the command based on current context
        response = self.process_contextual_command(command)

        if response:
            self.publish_response(response)

    def state_callback(self, msg):
        """Update robot state in context"""
        try:
            state_data = json.loads(msg.data)
            self.context['robot_state'].update(state_data)
            self.get_logger().debug(f'Updated robot state: {state_data}')
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid state JSON: {msg.data}')

    def extract_entities(self, command: str) -> List[str]:
        """Extract entities (objects, locations, etc.) from command"""
        # Simple entity extraction - in practice, use NER
        entities = []

        # Common objects
        objects = ['ball', 'cup', 'book', 'bottle', 'box', 'table', 'chair', 'couch']
        for obj in objects:
            if obj in command.lower():
                entities.append(obj)

        # Common locations
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'hallway']
        for loc in locations:
            if loc in command.lower():
                entities.append(loc)

        return entities

    def process_contextual_command(self, command: str) -> Optional[str]:
        """Process command considering current context"""
        command_lower = command.lower()

        # Handle follow-up questions
        if 'where' in command_lower and 'you' in command_lower:
            current_loc = self.context['robot_state']['location']
            return f"I am currently at the {current_loc}."

        if 'what' in command_lower and 'holding' in command_lower:
            carrying = self.context['robot_state']['carrying']
            if carrying:
                return f"I am holding a {carrying}."
            else:
                return "I am not holding anything."

        # Handle commands with implicit references
        if 'it' in command_lower or 'that' in command_lower:
            # Try to resolve "it" or "that" based on context
            if self.context['entities_mentioned']:
                last_mentioned = list(self.context['entities_mentioned'])[-1]
                resolved_command = command_lower.replace('it', last_mentioned).replace('that', last_mentioned)
                return f"Interpreting 'it' or 'that' as '{last_mentioned}'. {resolved_command}"

        # Handle task continuation
        if self.context['current_task']:
            if 'continue' in command_lower or 'proceed' in command_lower:
                return self.continue_current_task()
            elif 'cancel' in command_lower or 'stop' in command_lower:
                return self.cancel_current_task()

        # If no special handling needed, return None
        return None

    def continue_current_task(self) -> str:
        """Continue the current task"""
        current_task = self.context['current_task']
        if current_task:
            # In a real system, this would continue the task execution
            return f"Continuing with the {current_task} task."
        else:
            return "No task is currently in progress."

    def cancel_current_task(self) -> str:
        """Cancel the current task"""
        current_task = self.context['current_task']
        if current_task:
            # Add to completed tasks with cancelled status
            self.context['completed_tasks'].append({
                'task': current_task,
                'status': 'cancelled',
                'reason': 'user_request'
            })
            self.context['current_task'] = None
            return f"Cancelled the {current_task} task."
        else:
            return "No task is currently in progress."

    def cleanup_context(self):
        """Clean up old context if no recent interaction"""
        time_since_interaction = time.time() - self.context['last_interaction_time']

        if time_since_interaction > 300:  # 5 minutes
            # Reset conversation-specific context
            self.context['conversation_history'] = []
            self.context['entities_mentioned'] = set()
            self.context['turn_number'] = 0
            self.context['conversation_id'] += 1

            self.get_logger().info('Context cleaned up due to inactivity')

    def publish_response(self, response: str):
        """Publish contextual response"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Contextual response: {response}')

def main():
    rclpy.init()
    context_manager = ContextManagerNode()

    try:
        rclpy.spin(context_manager)
    except KeyboardInterrupt:
        pass
    finally:
        context_manager.destroy_node()
        rclpy.shutdown()
```

## Advanced Action Planning System

Here's a more sophisticated action planning system that can handle complex multi-step tasks:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStep:
    def __init__(self, action: str, params: Dict[str, Any], description: str = ""):
        self.action = action
        self.params = params
        self.description = description
        self.status = TaskStatus.PENDING
        self.execution_time = None

class Task:
    def __init__(self, name: str, steps: List[TaskStep], priority: int = 1):
        self.name = name
        self.steps = steps
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.current_step_index = 0
        self.created_time = self.get_current_time()

    def get_current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance_step(self):
        if self.current_step_index < len(self.steps):
            self.current_step_index += 1

    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

    def get_current_time(self):
        import time
        return time.time()

class ActionPlannerNode(Node):
    def __init__(self):
        super().__init__('action_planner')

        # Action clients
        self.move_group_client = ActionClient(self, MoveGroup, 'move_group')
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        # Subscribers
        self.task_sub = self.create_subscription(
            String,
            '/high_level_tasks',
            self.task_callback,
            10
        )

        self.task_control_sub = self.create_subscription(
            String,
            '/task_control',
            self.task_control_callback,
            10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/task_status', 10)
        self.action_pub = self.create_publisher(String, '/primitive_actions', 10)

        # Task management
        self.active_tasks: List[Task] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        # Timer for task execution
        self.execution_timer = self.create_timer(0.1, self.execute_next_step)

        self.get_logger().info('Action Planner node initialized')

    def task_callback(self, msg):
        """Receive high-level tasks"""
        try:
            task_data = json.loads(msg.data)
            task_name = task_data.get('name', 'unnamed_task')
            steps_data = task_data.get('steps', [])
            priority = task_data.get('priority', 1)

            # Convert steps data to TaskStep objects
            steps = []
            for step_data in steps_data:
                step = TaskStep(
                    action=step_data['action'],
                    params=step_data.get('params', {}),
                    description=step_data.get('description', '')
                )
                steps.append(step)

            # Create and queue task
            task = Task(task_name, steps, priority)

            # Insert in priority order
            inserted = False
            for i, queued_task in enumerate(self.task_queue):
                if task.priority > queued_task.priority:
                    self.task_queue.insert(i, task)
                    inserted = True
                    break

            if not inserted:
                self.task_queue.append(task)

            self.get_logger().info(f'Queued task: {task_name} with {len(steps)} steps')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid task JSON: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing task: {e}')

    def task_control_callback(self, msg):
        """Handle task control commands (pause, resume, cancel)"""
        control_cmd = msg.data.lower()

        if control_cmd == 'pause_all':
            self.pause_all_tasks()
        elif control_cmd == 'resume_all':
            self.resume_all_tasks()
        elif control_cmd == 'cancel_all':
            self.cancel_all_tasks()
        elif control_cmd.startswith('cancel_task:'):
            task_name = control_cmd.split(':', 1)[1]
            self.cancel_task_by_name(task_name)

    def execute_next_step(self):
        """Execute the next step of the highest priority active task"""
        # Find highest priority active task
        active_tasks = [t for t in self.active_tasks if t.status == TaskStatus.IN_PROGRESS]

        if not active_tasks:
            # Check if there are tasks in the queue to start
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.start_task(next_task)
                active_tasks = [next_task]

        if active_tasks:
            # Sort by priority (highest first)
            active_tasks.sort(key=lambda t: t.priority, reverse=True)
            current_task = active_tasks[0]

            current_step = current_task.get_current_step()
            if current_step:
                if current_step.status == TaskStatus.PENDING:
                    # Execute the step
                    success = self.execute_step(current_step)
                    if success:
                        current_step.status = TaskStatus.COMPLETED
                        current_step.execution_time = self.get_current_time()
                        current_task.advance_step()

                        # Check if task is complete
                        if current_task.is_complete():
                            current_task.status = TaskStatus.COMPLETED
                            self.completed_tasks.append(current_task)
                            self.active_tasks.remove(current_task)
                            self.publish_status(f'Task "{current_task.name}" completed successfully')
                    else:
                        current_step.status = TaskStatus.FAILED
                        current_task.status = TaskStatus.FAILED
                        self.publish_status(f'Task "{current_task.name}" failed at step: {current_step.description}')
                        self.active_tasks.remove(current_task)
            else:
                # No more steps, task is complete
                current_task.status = TaskStatus.COMPLETED
                self.completed_tasks.append(current_task)
                self.active_tasks.remove(current_task)
                self.publish_status(f'Task "{current_task.name}" completed')

    def execute_step(self, step: TaskStep) -> bool:
        """Execute a single task step"""
        try:
            if step.action == 'navigate':
                return self.execute_navigation_step(step.params)
            elif step.action == 'pick_up':
                return self.execute_pickup_step(step.params)
            elif step.action == 'place':
                return self.execute_place_step(step.params)
            elif step.action == 'open_gripper':
                return self.execute_gripper_step('open', step.params)
            elif step.action == 'close_gripper':
                return self.execute_gripper_step('close', step.params)
            else:
                self.get_logger().warn(f'Unknown action: {step.action}')
                return False
        except Exception as e:
            self.get_logger().error(f'Error executing step {step.action}: {e}')
            return False

    def execute_navigation_step(self, params: Dict[str, Any]) -> bool:
        """Execute navigation step"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        theta = params.get('theta', 0.0)

        # Create and send navigation goal
        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        goal_pose.position.z = 0.0

        import math
        goal_pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.orientation.w = math.cos(theta / 2.0)

        # For simulation, just publish to topic
        nav_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        nav_pub.publish(goal_pose)

        self.get_logger().info(f'Navigating to ({x}, {y}, {theta})')
        return True

    def execute_pickup_step(self, params: Dict[str, Any]) -> bool:
        """Execute pickup step"""
        object_name = params.get('object_name', 'unknown_object')
        self.get_logger().info(f'Picking up {object_name}')

        # In a real system, this would involve perception and manipulation
        # For simulation, just return success
        return True

    def execute_place_step(self, params: Dict[str, Any]) -> bool:
        """Execute place step"""
        object_name = params.get('object_name', 'unknown_object')
        location = params.get('location', 'default')
        self.get_logger().info(f'Placing {object_name} at {location}')

        # In a real system, this would involve navigation and manipulation
        # For simulation, just return success
        return True

    def execute_gripper_step(self, action: str, params: Dict[str, Any]) -> bool:
        """Execute gripper action"""
        # Create joint trajectory for gripper
        trajectory = JointTrajectory()
        trajectory.joint_names = ['left_gripper_finger_joint', 'right_gripper_finger_joint']

        point = JointTrajectoryPoint()
        if action == 'open':
            point.positions = [0.08, 0.08]  # Open position
        else:  # close
            point.positions = [0.0, 0.0]    # Closed position

        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)

        # Publish gripper command
        gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        gripper_pub.publish(trajectory)

        self.get_logger().info(f'Gripper {action} command sent')
        return True

    def start_task(self, task: Task):
        """Start executing a task"""
        task.status = TaskStatus.IN_PROGRESS
        self.active_tasks.append(task)
        self.publish_status(f'Starting task: {task.name}')

    def pause_all_tasks(self):
        """Pause all active tasks"""
        for task in self.active_tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.PENDING
        self.publish_status('All tasks paused')

    def resume_all_tasks(self):
        """Resume all paused tasks"""
        for task in self.active_tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
        self.publish_status('All tasks resumed')

    def cancel_all_tasks(self):
        """Cancel all tasks"""
        for task in self.active_tasks:
            task.status = TaskStatus.CANCELLED
        self.active_tasks.clear()
        self.task_queue.clear()
        self.publish_status('All tasks cancelled')

    def cancel_task_by_name(self, name: str):
        """Cancel a specific task by name"""
        # Remove from active tasks
        for task in self.active_tasks:
            if task.name == name:
                task.status = TaskStatus.CANCELLED
                self.active_tasks.remove(task)
                self.publish_status(f'Task "{name}" cancelled')
                return

        # Remove from queue
        for task in self.task_queue:
            if task.name == name:
                task.status = TaskStatus.CANCELLED
                self.task_queue.remove(task)
                self.publish_status(f'Queued task "{name}" cancelled')
                return

        self.publish_status(f'Task "{name}" not found')

    def publish_status(self, status: str):
        """Publish task status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Task status: {status}')

    def get_current_time(self):
        import time
        return time.time()

def main():
    rclpy.init()
    planner = ActionPlannerNode()

    # Example: Add a complex task
    example_task = {
        "name": "fetch_and_place",
        "priority": 2,
        "steps": [
            {
                "action": "navigate",
                "params": {"x": 2.0, "y": 1.0, "theta": 0.0},
                "description": "Navigate to kitchen"
            },
            {
                "action": "pick_up",
                "params": {"object_name": "red cup"},
                "description": "Pick up red cup"
            },
            {
                "action": "navigate",
                "params": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "description": "Return to base"
            },
            {
                "action": "place",
                "params": {"object_name": "red cup", "location": "table"},
                "description": "Place cup on table"
            }
        ]
    }

    # Publish the example task
    import time
    time.sleep(1)  # Wait for publisher to be ready
    task_pub = planner.create_publisher(String, '/high_level_tasks', 10)
    task_msg = String()
    task_msg.data = json.dumps(example_task)
    task_pub.publish(task_msg)

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Action Mapping Systems

- **Modularity**: Keep action definitions modular and easily extensible
- **Context Awareness**: Maintain relevant context for multi-turn interactions
- **Error Handling**: Implement robust error handling and recovery
- **Task Planning**: Support complex multi-step tasks with dependencies
- **Priority Management**: Handle multiple concurrent tasks appropriately
- **State Management**: Maintain consistent robot state across actions
- **User Feedback**: Provide clear feedback about task progress and status

## Next Steps

In the next chapter, we'll explore context awareness systems that make VLA systems more intelligent and responsive to their environment and user needs.