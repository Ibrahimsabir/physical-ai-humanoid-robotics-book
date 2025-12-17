---
title: Context Awareness in VLA Systems
sidebar_label: Context Awareness
---

# Context Awareness in VLA Systems

## Introduction to Context Awareness

Context awareness is the ability of a robot to understand and respond to its environment, state, and user needs. In Vision-Language-Action (VLA) systems, context awareness enables robots to interpret commands more accurately by considering the current situation, previous interactions, and environmental factors.

### Key Context Dimensions

- **Spatial Context**: Physical location and environment layout
- **Temporal Context**: Time of day, sequence of events, duration
- **Social Context**: Presence of people, their activities, social norms
- **Task Context**: Current and previous tasks, goals, and progress
- **Device Context**: Robot capabilities, battery level, available resources

## Implementing Context Awareness

Here's a comprehensive context awareness system for VLA robots:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16, Float32
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import LaserScan, BatteryState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Time
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class ContextAwarenessNode(Node):
    def __init__(self):
        super().__init__('context_awareness')

        # Publishers for context information
        self.context_pub = self.create_publisher(String, '/context_state', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/context_visualization', 10)

        # Subscribers for various context sources
        self.odom_sub = self.create_subscription(
            Pose,
            '/odom',
            self.odom_callback,
            10
        )

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

        self.user_proximity_sub = self.create_subscription(
            Float32,
            '/user_distance',
            self.user_proximity_callback,
            10
        )

        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )

        # Initialize context
        self.context = {
            'spatial': {
                'current_position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
                'room': 'unknown',
                'obstacles': [],
                'navigation_goals': [],
                'safe_zones': [],
                'forbidden_zones': []
            },
            'temporal': {
                'start_time': time.time(),
                'current_time': time.time(),
                'session_duration': 0,
                'time_of_day': 'unknown',
                'day_of_week': 'unknown'
            },
            'social': {
                'user_distance': float('inf'),
                'user_presence': False,
                'user_attention': 'unknown',
                'interaction_history': []
            },
            'task': {
                'current_task': None,
                'task_progress': 0.0,
                'task_history': [],
                'task_priority': 0
            },
            'device': {
                'battery_level': 100.0,
                'charging': False,
                'available_capabilities': [],
                'performance_metrics': {}
            },
            'environmental': {
                'lighting_condition': 'unknown',
                'noise_level': 0.0,
                'temperature': 20.0,
                'obstacle_density': 0.0
            }
        }

        # Timer for context updates
        self.context_timer = self.create_timer(1.0, self.update_context)
        self.viz_timer = self.create_timer(0.5, self.publish_visualization)

        # Context history for temporal reasoning
        self.context_history = []

        self.get_logger().info('Context Awareness node initialized')

    def odom_callback(self, msg: Pose):
        """Update spatial context from odometry"""
        self.context['spatial']['current_position'] = {
            'x': msg.position.x,
            'y': msg.position.y,
            'theta': self.quaternion_to_yaw(msg.orientation)
        }

        # Determine room based on position (simplified)
        x, y = msg.position.x, msg.position.y
        if abs(x) < 2.0 and abs(y) < 2.0:
            self.context['spatial']['room'] = 'center'
        elif x > 1.0:
            self.context['spatial']['room'] = 'kitchen'
        elif y > 1.0:
            self.context['spatial']['room'] = 'living_room'
        elif x < -1.0:
            self.context['spatial']['room'] = 'bedroom'
        else:
            self.context['spatial']['room'] = 'hallway'

    def scan_callback(self, msg: LaserScan):
        """Update obstacle context from laser scan"""
        obstacles = []
        min_distance = float('inf')

        for i, range_val in enumerate(msg.ranges):
            if msg.range_min <= range_val <= msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment
                # Convert polar to Cartesian
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                if range_val < 1.0:  # Consider as obstacle if closer than 1m
                    obstacles.append({
                        'x': x,
                        'y': y,
                        'distance': range_val,
                        'angle': angle
                    })

                if range_val < min_distance:
                    min_distance = range_val

        self.context['spatial']['obstacles'] = obstacles
        self.context['environmental']['obstacle_density'] = len(obstacles) / len(msg.ranges) if msg.ranges else 0.0

    def battery_callback(self, msg: BatteryState):
        """Update device context from battery state"""
        self.context['device']['battery_level'] = msg.percentage * 100.0
        self.context['device']['charging'] = msg.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING

    def user_proximity_callback(self, msg: Float32):
        """Update social context from user proximity"""
        distance = msg.data
        self.context['social']['user_distance'] = distance
        self.context['social']['user_presence'] = distance < 3.0  # User is present if within 3m

    def task_status_callback(self, msg: String):
        """Update task context from task status"""
        try:
            status_data = json.loads(msg.data)
            task_name = status_data.get('task_name', 'unknown')
            status = status_data.get('status', 'unknown')
            progress = status_data.get('progress', 0.0)

            self.context['task']['current_task'] = task_name
            self.context['task']['task_progress'] = progress

            # Add to history if task completed
            if status == 'completed':
                self.context['task']['task_history'].append({
                    'task_name': task_name,
                    'completion_time': time.time(),
                    'success': True
                })
        except json.JSONDecodeError:
            # Handle simple status messages
            self.get_logger().debug(f'Task status: {msg.data}')

    def update_context(self):
        """Update temporal and other context information"""
        current_time = time.time()
        self.context['temporal']['current_time'] = current_time
        self.context['temporal']['session_duration'] = current_time - self.context['temporal']['start_time']

        # Update time of day
        dt = datetime.fromtimestamp(current_time)
        hour = dt.hour
        if 6 <= hour < 12:
            self.context['temporal']['time_of_day'] = 'morning'
        elif 12 <= hour < 17:
            self.context['temporal']['time_of_day'] = 'afternoon'
        elif 17 <= hour < 22:
            self.context['temporal']['time_of_day'] = 'evening'
        else:
            self.context['temporal']['time_of_day'] = 'night'

        self.context['temporal']['day_of_week'] = dt.strftime('%A')

        # Update environmental context based on time and other factors
        if self.context['temporal']['time_of_day'] in ['night', 'morning']:
            self.context['environmental']['lighting_condition'] = 'dim'
        else:
            self.context['environmental']['lighting_condition'] = 'bright'

        # Add to history (keep last 100 entries)
        self.context_history.append((current_time, self.context.copy()))
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]

        # Publish updated context
        context_msg = String()
        context_msg.data = json.dumps(self.context, indent=2)
        self.context_pub.publish(context_msg)

    def publish_visualization(self):
        """Publish visualization markers for context"""
        marker_array = MarkerArray()
        id_counter = 0

        # Visualize obstacles
        for i, obstacle in enumerate(self.context['spatial']['obstacles'][:20]):  # Limit visualization
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = id_counter
            id_counter += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obstacle['x']
            marker.pose.position.y = obstacle['y']
            marker.pose.position.z = 0.25  # Half height of cylinder
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1  # Diameter
            marker.scale.y = 0.1
            marker.scale.z = 0.5  # Height

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        # Visualize user if present
        if self.context['social']['user_presence']:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "user"
            marker.id = id_counter
            id_counter += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Calculate user position based on robot position and distance
            robot_pos = self.context['spatial']['current_position']
            # This is a simplified calculation - in reality, you'd need angle too
            marker.pose.position.x = robot_pos['x'] + 1.0  # Assumed user position
            marker.pose.position.y = robot_pos['y']
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.viz_pub.publish(marker_array)

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_context_relevance(self, query: str) -> Dict[str, float]:
        """Determine which context dimensions are relevant to a query"""
        relevance_scores = {}

        # Spatial relevance
        spatial_keywords = ['go', 'move', 'navigate', 'to', 'at', 'near', 'by', 'location', 'room', 'kitchen', 'bedroom']
        spatial_score = sum(1 for keyword in spatial_keywords if keyword.lower() in query.lower())
        relevance_scores['spatial'] = min(spatial_score / len(spatial_keywords), 1.0)

        # Social relevance
        social_keywords = ['you', 'me', 'us', 'person', 'human', 'user', 'follow', 'come', 'with']
        social_score = sum(1 for keyword in social_keywords if keyword.lower() in query.lower())
        relevance_scores['social'] = min(social_score / len(social_keywords), 1.0)

        # Task relevance
        task_keywords = ['task', 'do', 'perform', 'complete', 'help', 'assist', 'work', 'job']
        task_score = sum(1 for keyword in task_keywords if keyword.lower() in query.lower())
        relevance_scores['task'] = min(task_score / len(task_keywords), 1.0)

        # Device relevance (battery, charging, etc.)
        device_keywords = ['battery', 'charge', 'power', 'energy', 'tired', 'low']
        device_score = sum(1 for keyword in device_keywords if keyword.lower() in query.lower())
        relevance_scores['device'] = min(device_score / len(device_keywords), 1.0)

        return relevance_scores

    def get_context_summary(self) -> str:
        """Generate a text summary of the current context"""
        summary = f"Context Summary:\n"
        summary += f"- Position: ({self.context['spatial']['current_position']['x']:.2f}, {self.context['spatial']['current_position']['y']:.2f}) in {self.context['spatial']['room']}\n"
        summary += f"- User: {'present' if self.context['social']['user_presence'] else 'absent'} (distance: {self.context['social']['user_distance']:.2f}m)\n"
        summary += f"- Battery: {self.context['device']['battery_level']:.1f}% ({'charging' if self.context['device']['charging'] else 'not charging'})\n"
        summary += f"- Current task: {self.context['task']['current_task'] or 'none'} ({self.context['task']['task_progress']:.1f}% complete)\n"
        summary += f"- Time: {self.context['temporal']['time_of_day']} ({self.context['temporal']['day_of_week']})\n"
        summary += f"- Obstacles detected: {len(self.context['spatial']['obstacles'])}\n"

        return summary

def main():
    rclpy.init()
    context_node = ContextAwarenessNode()

    try:
        rclpy.spin(context_node)
    except KeyboardInterrupt:
        pass
    finally:
        context_node.destroy_node()
        rclpy.shutdown()
```

## Context-Based Command Processing

Here's how to integrate context awareness with command processing:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from typing import Dict, Any

class ContextualCommandProcessor(Node):
    def __init__(self):
        super().__init__('contextual_command_processor')

        # Subscribe to commands and context
        self.command_sub = self.create_subscription(
            String,
            '/user_commands',
            self.command_callback,
            10
        )

        self.context_sub = self.create_subscription(
            String,
            '/context_state',
            self.context_callback,
            10
        )

        # Publisher for processed commands
        self.processed_command_pub = self.create_publisher(String, '/processed_commands', 10)

        # Store current context
        self.current_context = {}
        self.command_history = []

        self.get_logger().info('Contextual Command Processor initialized')

    def context_callback(self, msg: String):
        """Update current context"""
        try:
            self.current_context = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid context JSON: {msg.data}')

    def command_callback(self, msg: String):
        """Process command with context awareness"""
        command = msg.data
        timestamp = self.get_clock().now().nanoseconds / 1e9

        # Add to command history
        self.command_history.append({
            'command': command,
            'timestamp': timestamp,
            'context_snapshot': self.current_context.copy()
        })

        # Process command with context
        processed_command = self.process_with_context(command, self.current_context)

        if processed_command:
            # Publish processed command
            processed_msg = String()
            processed_msg.data = json.dumps(processed_command)
            self.processed_command_pub.publish(processed_msg)

            self.get_logger().info(f'Processed command: {processed_command}')

    def process_with_context(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process command considering current context"""
        command_lower = command.lower()

        # Resolve ambiguous references based on context
        resolved_command = self.resolve_ambiguous_references(command_lower, context)

        # Determine action based on context
        action = self.determine_contextual_action(resolved_command, context)

        # Add context-relevant parameters
        contextual_params = self.add_contextual_parameters(action, context)

        return {
            'original_command': command,
            'resolved_command': resolved_command,
            'action': action,
            'context': contextual_params,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        }

    def resolve_ambiguous_references(self, command: str, context: Dict[str, Any]) -> str:
        """Resolve ambiguous references like 'it', 'there', 'here' based on context"""
        resolved = command

        # Resolve 'here' based on current location
        if 'here' in resolved:
            room = context.get('spatial', {}).get('room', 'current location')
            resolved = resolved.replace('here', f'in the {room}')

        # Resolve 'there' based on last mentioned location or current context
        if 'there' in resolved:
            # In a real system, you'd have more sophisticated resolution
            # For now, we'll leave it as is or try to infer from context
            pass

        # Resolve 'it' based on last mentioned object or action
        if 'it' in resolved and 'task' in context:
            last_task = context['task'].get('current_task')
            if last_task:
                resolved = resolved.replace('it', last_task)

        return resolved

    def determine_contextual_action(self, command: str, context: Dict[str, Any]) -> str:
        """Determine appropriate action based on command and context"""
        # Check if user is present (social context)
        user_present = context.get('social', {}).get('user_presence', False)

        # Check battery level (device context)
        battery_level = context.get('device', {}).get('battery_level', 100.0)

        # Check current room (spatial context)
        current_room = context.get('spatial', {}).get('room', 'unknown')

        # Modify action based on context
        if not user_present and 'follow' in command:
            return 'wait_and_announce'  # Wait for user before following

        if battery_level < 20.0 and ('go to' in command or 'navigate' in command):
            return 'battery_check_first'  # Check battery before navigating far

        if current_room == 'bedroom' and 'loud' in command:
            return 'quiet_mode'  # Use quiet mode in bedroom

        # Default action determination
        if 'go to' in command or 'navigate to' in command:
            return 'navigate'
        elif 'pick up' in command or 'grasp' in command:
            return 'pick_up'
        elif 'place' in command or 'put' in command:
            return 'place'
        elif 'follow' in command:
            return 'follow'
        else:
            return 'unknown'

    def add_contextual_parameters(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add context-relevant parameters to the action"""
        params = {}

        if action == 'navigate':
            # Add safety parameters based on obstacle density
            obstacle_density = context.get('environmental', {}).get('obstacle_density', 0.0)
            params['safety_margin'] = 0.3 + (obstacle_density * 0.5)  # More obstacles = wider margin

            # Add speed based on lighting condition
            lighting = context.get('environmental', {}).get('lighting_condition', 'bright')
            params['max_speed'] = 0.5 if lighting == 'dim' else 1.0

        elif action == 'follow':
            # Add distance based on user proximity
            user_distance = context.get('social', {}).get('user_distance', 2.0)
            params['follow_distance'] = max(0.5, min(user_distance, 2.0))

        elif action == 'battery_check_first':
            params['battery_level'] = context.get('device', {}).get('battery_level', 100.0)
            params['critical_threshold'] = 15.0

        # Add temporal context
        time_of_day = context.get('temporal', {}).get('time_of_day', 'unknown')
        params['time_context'] = time_of_day

        # Add spatial context
        current_room = context.get('spatial', {}).get('room', 'unknown')
        params['current_location'] = current_room

        return params

def main():
    rclpy.init()
    processor = ContextualCommandProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
```

## Context Learning and Adaptation

Here's a system that learns from context to improve future interactions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple

class ContextLearnerNode(Node):
    def __init__(self):
        super().__init__('context_learner')

        # Subscribe to commands, context, and outcomes
        self.command_sub = self.create_subscription(
            String,
            '/processed_commands',
            self.command_callback,
            10
        )

        self.context_sub = self.create_subscription(
            String,
            '/context_state',
            self.context_callback,
            10
        )

        self.outcome_sub = self.create_subscription(
            String,
            '/action_outcomes',
            self.outcome_callback,
            10
        )

        # Store learned patterns
        self.context_patterns = defaultdict(list)  # Maps context patterns to successful actions
        self.action_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
        self.context_history = []

        # Load learned patterns if available
        self.load_learned_patterns()

        # Timer to save patterns periodically
        self.save_timer = self.create_timer(60.0, self.save_learned_patterns)  # Save every minute

        self.get_logger().info('Context Learner node initialized')

    def command_callback(self, msg: String):
        """Record command and context for learning"""
        try:
            command_data = json.loads(msg.data)
            context_snapshot = self.get_current_context_snapshot()

            if context_snapshot:
                # Store the command-context pair
                self.context_history.append({
                    'command': command_data,
                    'context': context_snapshot,
                    'timestamp': self.get_clock().now().nanoseconds / 1e9,
                    'outcome': None  # Will be filled when outcome is received
                })

                # Keep history manageable
                if len(self.context_history) > 1000:
                    self.context_history = self.context_history[-500:]

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid command JSON: {msg.data}')

    def context_callback(self, msg: String):
        """Store current context"""
        # Context is already stored by the context awareness node
        # This is just to ensure we have the latest context
        pass

    def outcome_callback(self, msg: String):
        """Record outcome to update learning"""
        try:
            outcome_data = json.loads(msg.data)
            success = outcome_data.get('success', False)
            action = outcome_data.get('action', 'unknown')

            # Update success rate for this action
            if success:
                self.action_success_rates[action]['success'] += 1
            self.action_success_rates[action]['total'] += 1

            # Try to match outcome with recent command-context
            for entry in reversed(self.context_history[-10:]):  # Look at last 10 entries
                if (entry['command']['action'] == action and
                    entry['outcome'] is None and
                    abs(entry['timestamp'] - outcome_data.get('timestamp', 0)) < 10):  # Within 10 seconds
                    entry['outcome'] = success
                    self.learn_from_interaction(entry)
                    break

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid outcome JSON: {msg.data}')

    def get_current_context_snapshot(self) -> Dict[str, Any]:
        """Get current context snapshot"""
        # In a real implementation, you'd get the current context
        # For now, we'll return a placeholder
        return {
            'spatial': {'room': 'unknown', 'obstacle_density': 0.0},
            'temporal': {'time_of_day': 'unknown'},
            'social': {'user_presence': False},
            'device': {'battery_level': 100.0}
        }

    def learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn from a successful or failed interaction"""
        if interaction['outcome'] is None:
            return

        context = interaction['context']
        command = interaction['command']
        success = interaction['outcome']
        action = command['action']

        # Create context signature for pattern matching
        context_signature = self.create_context_signature(context)

        # Update pattern learning
        pattern_key = f"{context_signature}_{action}"

        if success:
            # Add to successful patterns if not already there
            if action not in self.context_patterns[context_signature]:
                self.context_patterns[context_signature].append(action)
        else:
            # Remove from successful patterns if it failed
            if action in self.context_patterns[context_signature]:
                self.context_patterns[context_signature].remove(action)

        self.get_logger().debug(f'Learned from {action}: success={success}, context={context_signature}')

    def create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context matching"""
        # Simplified signature - in practice, this would be more sophisticated
        spatial_room = context.get('spatial', {}).get('room', 'unknown')
        time_of_day = context.get('temporal', {}).get('time_of_day', 'unknown')
        user_present = context.get('social', {}).get('user_presence', False)
        battery_level = context.get('device', {}).get('battery_level', 100.0)

        # Discretize continuous values
        battery_range = 'low' if battery_level < 30 else 'medium' if battery_level < 70 else 'high'

        return f"{spatial_room}_{time_of_day}_user-{user_present}_battery-{battery_range}"

    def get_advice_for_context(self, context: Dict[str, Any]) -> List[str]:
        """Get advice based on learned patterns for a given context"""
        context_signature = self.create_context_signature(context)
        suggested_actions = self.context_patterns[context_signature]

        # Sort by success rate
        success_rates = []
        for action in suggested_actions:
            stats = self.action_success_rates[action]
            if stats['total'] > 0:
                rate = stats['success'] / stats['total']
                success_rates.append((action, rate))

        success_rates.sort(key=lambda x: x[1], reverse=True)
        return [action for action, rate in success_rates]

    def load_learned_patterns(self):
        """Load learned patterns from file"""
        patterns_file = 'context_patterns.pkl'
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'rb') as f:
                    loaded = pickle.load(f)
                    self.context_patterns.update(loaded)
                self.get_logger().info(f'Loaded context patterns from {patterns_file}')
            except Exception as e:
                self.get_logger().error(f'Error loading context patterns: {e}')

    def save_learned_patterns(self):
        """Save learned patterns to file"""
        try:
            with open('context_patterns.pkl', 'wb') as f:
                pickle.dump(dict(self.context_patterns), f)
            self.get_logger().debug('Saved context patterns to file')
        except Exception as e:
            self.get_logger().error(f'Error saving context patterns: {e}')

def main():
    rclpy.init()
    learner = ContextLearnerNode()

    try:
        rclpy.spin(learner)
    except KeyboardInterrupt:
        pass
    finally:
        learner.save_learned_patterns()  # Save on shutdown
        learner.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Context Awareness

- **Modularity**: Keep different context dimensions separate but integrated
- **Efficiency**: Update context efficiently without overwhelming the system
- **Relevance**: Focus on context that actually affects robot behavior
- **Learning**: Implement adaptive systems that learn from experience
- **Privacy**: Respect user privacy when collecting context information
- **Robustness**: Handle missing or uncertain context gracefully
- **Scalability**: Design systems that can handle multiple context sources

## Next Steps

In the next module, we'll explore the capstone integration that brings together all the capabilities we've developed into a complete autonomous humanoid system.