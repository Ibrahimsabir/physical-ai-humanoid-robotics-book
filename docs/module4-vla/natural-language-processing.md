---
title: Natural Language Processing for Robotics
sidebar_label: Natural Language Processing
---

# Natural Language Processing for Robotics

## Introduction to Natural Language Processing in Robotics

Natural Language Processing (NLP) in robotics enables human-robot interaction through spoken or written commands. This technology allows robots to understand, interpret, and execute commands given in natural language, making them more accessible and intuitive to use.

### Key NLP Components for Robotics

- **Speech Recognition**: Converting speech to text
- **Natural Language Understanding**: Interpreting the meaning of commands
- **Intent Recognition**: Determining what action the user wants
- **Entity Extraction**: Identifying objects, locations, and parameters
- **Speech Synthesis**: Converting robot responses to speech (optional)

## Setting Up Speech Recognition with Whisper

Whisper is an open-source automatic speech recognition (ASR) system developed by OpenAI. Here's how to set it up for robotics applications:

```bash
# Install Whisper and related dependencies
pip install openai-whisper
pip install torch torchvision torchaudio
pip install sounddevice numpy
```

## Creating a Speech Recognition Node

Here's a ROS 2 node that captures audio and processes it with Whisper:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Load Whisper model (use 'tiny', 'base', 'small', 'medium', or 'large')
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded successfully')

        # Publisher for recognized text
        self.text_pub = self.create_publisher(String, '/speech_to_text', 10)

        # Audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.recording = False

        # Timer to start audio recording
        self.timer = self.create_timer(1.0, self.start_recording)

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input"""
        if status:
            self.get_logger().warning(f'Audio status: {status}')

        # Add audio data to queue
        audio_data = indata.copy()
        self.audio_queue.put(audio_data)

    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.recording = True
        self.timer.cancel()  # Stop the timer after starting

        # Start audio recording thread
        audio_thread = threading.Thread(target=self.record_audio)
        audio_thread.daemon = True
        audio_thread.start()

        self.get_logger().info('Started audio recording for speech recognition')

    def record_audio(self):
        """Record audio in a loop and process when enough data is collected"""
        accumulated_audio = np.array([], dtype=np.float32)

        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32'
        ):
            while self.recording:
                try:
                    # Get audio data from queue
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Accumulate audio data
                    accumulated_audio = np.concatenate([accumulated_audio, audio_chunk.flatten()])

                    # Process audio when we have enough (about 5 seconds of audio)
                    if len(accumulated_audio) >= self.sample_rate * 5:
                        # Process the accumulated audio
                        self.process_audio(accumulated_audio)

                        # Keep some overlap to avoid losing words at the boundary
                        overlap_samples = int(self.sample_rate * 0.5)  # 0.5 second overlap
                        accumulated_audio = accumulated_audio[-overlap_samples:]

                except queue.Empty:
                    continue

    def process_audio(self, audio_data):
        """Process audio data with Whisper"""
        try:
            # Convert audio to the format expected by Whisper
            audio_data = audio_data.astype(np.float32)

            # Run Whisper transcription
            result = self.model.transcribe(audio_data, fp16=False)
            text = result['text'].strip()

            if text:  # Only publish if there's text
                self.get_logger().info(f'Recognized: {text}')

                # Publish recognized text
                msg = String()
                msg.data = text
                self.text_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def destroy_node(self):
        """Clean up when node is destroyed"""
        self.recording = False
        super().destroy_node()

def main():
    rclpy.init()
    speech_node = SpeechRecognitionNode()

    try:
        rclpy.spin(speech_node)
    except KeyboardInterrupt:
        pass
    finally:
        speech_node.destroy_node()
        rclpy.shutdown()
```

## Natural Language Understanding Node

Here's a node that processes the recognized text and converts it to robot commands:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveGroupGoal
from std_msgs.msg import Bool
import re

class NaturalLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('natural_language_understanding')

        # Subscribe to speech recognition output
        self.speech_sub = self.create_subscription(
            String,
            '/speech_to_text',
            self.speech_callback,
            10
        )

        # Publishers for different robot commands
        self.navigation_pub = self.create_publisher(Pose, '/navigation/goal', 10)
        self.manipulation_pub = self.create_publisher(String, '/manipulation/command', 10)
        self.system_pub = self.create_publisher(String, '/system/command', 10)

        # Publisher for robot responses
        self.response_pub = self.create_publisher(String, '/robot_response', 10)

        self.get_logger().info('Natural Language Understanding node started')

    def speech_callback(self, msg):
        """Process incoming speech text"""
        text = msg.data.lower().strip()
        self.get_logger().info(f'Processing command: "{text}"')

        # Parse the command and determine intent
        intent, params = self.parse_command(text)

        if intent:
            self.execute_command(intent, params)
        else:
            self.respond(f"I didn't understand the command: {text}")

    def parse_command(self, text):
        """Parse natural language command and extract intent and parameters"""
        # Navigation commands
        navigation_patterns = [
            (r'move to (.+)', 'navigate'),
            (r'go to (.+)', 'navigate'),
            (r'go (.+)', 'navigate'),
            (r'go near (.+)', 'navigate'),
            (r'go by (.+)', 'navigate'),
        ]

        # Manipulation commands
        manipulation_patterns = [
            (r'pick up (.+)', 'pick_up'),
            (r'grasp (.+)', 'pick_up'),
            (r'grab (.+)', 'pick_up'),
            (r'lift (.+)', 'pick_up'),
            (r'put (.+) (.+)', 'place'),
            (r'place (.+) (.+)', 'place'),
            (r'drop (.+)', 'drop'),
        ]

        # System commands
        system_patterns = [
            (r'stop', 'stop'),
            (r'pause', 'stop'),
            (r'resume', 'resume'),
            (r'help', 'help'),
            (r'what can you do', 'help'),
            (r'hello', 'greet'),
            (r'hi', 'greet'),
        ]

        # Check navigation patterns
        for pattern, intent in navigation_patterns:
            match = re.search(pattern, text)
            if match:
                return intent, {'location': match.group(1)}

        # Check manipulation patterns
        for pattern, intent in manipulation_patterns:
            match = re.search(pattern, text)
            if match:
                if intent == 'place':
                    # Special handling for place command with location
                    parts = text.split()
                    if len(parts) >= 3:
                        object_name = parts[1]  # e.g., "ball" in "put ball on table"
                        location = ' '.join(parts[2:])
                        return intent, {'object': object_name, 'location': location}
                return intent, {'object': match.group(1)}

        # Check system patterns
        for pattern, intent in system_patterns:
            if re.search(pattern, text):
                return intent, {}

        # If no pattern matches, return None
        return None, {}

    def execute_command(self, intent, params):
        """Execute the parsed command"""
        if intent == 'navigate':
            self.execute_navigation(params)
        elif intent == 'pick_up':
            self.execute_manipulation_pick(params)
        elif intent == 'place':
            self.execute_manipulation_place(params)
        elif intent == 'drop':
            self.execute_manipulation_drop(params)
        elif intent == 'stop':
            self.execute_stop()
        elif intent == 'resume':
            self.execute_resume()
        elif intent == 'help':
            self.execute_help()
        elif intent == 'greet':
            self.execute_greet()
        else:
            self.respond(f"Unknown command intent: {intent}")

    def execute_navigation(self, params):
        """Execute navigation command"""
        location = params.get('location', '')
        self.get_logger().info(f'Navigating to {location}')

        # In a real system, you would convert the location name to coordinates
        # For now, we'll use a simple mapping
        location_map = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 2.0, 0.0),
            'bedroom': (-1.0, -1.0, 0.0),
            'table': (1.0, 0.0, 0.0),
            'couch': (0.0, 1.0, 0.0),
        }

        if location in location_map:
            x, y, theta = location_map[location]
            pose_msg = Pose()
            pose_msg.position.x = x
            pose_msg.position.y = y
            pose_msg.position.z = 0.0
            # Simple orientation (facing forward)
            pose_msg.orientation.z = 0.0
            pose_msg.orientation.w = 1.0

            self.navigation_pub.publish(pose_msg)
            self.respond(f"Moving to the {location}")
        else:
            self.respond(f"I don't know where {location} is. I can go to: kitchen, living room, bedroom, table, or couch.")

    def execute_manipulation_pick(self, params):
        """Execute pick up command"""
        obj = params.get('object', 'object')
        self.get_logger().info(f'Attempting to pick up {obj}')

        # Publish manipulation command
        cmd_msg = String()
        cmd_msg.data = f'pick_up:{obj}'
        self.manipulation_pub.publish(cmd_msg)
        self.respond(f"Attempting to pick up the {obj}")

    def execute_manipulation_place(self, params):
        """Execute place command"""
        obj = params.get('object', 'object')
        location = params.get('location', 'nearby')
        self.get_logger().info(f'Attempting to place {obj} at {location}')

        # Publish manipulation command
        cmd_msg = String()
        cmd_msg.data = f'place:{obj}:at:{location}'
        self.manipulation_pub.publish(cmd_msg)
        self.respond(f"Attempting to place the {obj} at {location}")

    def execute_manipulation_drop(self, params):
        """Execute drop command"""
        obj = params.get('object', 'object')
        self.get_logger().info(f'Dropping {obj}')

        # Publish manipulation command
        cmd_msg = String()
        cmd_msg.data = f'drop:{obj}'
        self.manipulation_pub.publish(cmd_msg)
        self.respond(f"Dropping the {obj}")

    def execute_stop(self):
        """Execute stop command"""
        cmd_msg = String()
        cmd_msg.data = 'stop'
        self.system_pub.publish(cmd_msg)
        self.respond("Stopping all operations")

    def execute_resume(self):
        """Execute resume command"""
        cmd_msg = String()
        cmd_msg.data = 'resume'
        self.system_pub.publish(cmd_msg)
        self.respond("Resuming operations")

    def execute_help(self):
        """Execute help command"""
        help_text = (
            "I can help you with navigation and manipulation tasks. "
            "You can ask me to: "
            "go to a location (like 'go to kitchen'), "
            "pick up objects (like 'pick up the red ball'), "
            "place objects (like 'place the ball on the table'), "
            "or stop/resume operations."
        )
        self.respond(help_text)

    def execute_greet(self):
        """Execute greeting"""
        self.respond("Hello! I'm your robot assistant. How can I help you today?")

    def respond(self, text):
        """Publish a response"""
        response_msg = String()
        response_msg.data = text
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Robot response: {text}')

def main():
    rclpy.init()
    nlu_node = NaturalLanguageUnderstandingNode()

    try:
        rclpy.spin(nlu_node)
    except KeyboardInterrupt:
        pass
    finally:
        nlu_node.destroy_node()
        rclpy.shutdown()
```

## Creating a Command Mapping System

Here's a more sophisticated system that maps natural language to specific robot actions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import json
import os

class CommandMapperNode(Node):
    def __init__(self):
        super().__init__('command_mapper')

        # Subscribe to natural language commands
        self.command_sub = self.create_subscription(
            String,
            '/parsed_commands',
            self.command_callback,
            10
        )

        # Publishers for different robot subsystems
        self.nav_pub = self.create_publisher(Pose, '/move_base_simple/goal', 10)
        self.arm_pub = self.create_publisher(JointState, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointState, '/gripper_controller/joint_trajectory', 10)

        # Load command mappings from configuration
        self.load_command_mappings()

        self.get_logger().info('Command Mapper node initialized')

    def load_command_mappings(self):
        """Load command mappings from configuration file"""
        # Default mappings - in practice, these would be loaded from a config file
        self.command_mappings = {
            'locations': {
                'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
                'living room': {'x': 0.0, 'y': 2.0, 'theta': 1.57},
                'bedroom': {'x': -1.0, 'y': -1.0, 'theta': 3.14},
                'office': {'x': 1.5, 'y': -0.5, 'theta': -1.57},
            },
            'objects': {
                'red ball': {'type': 'graspable', 'size': 'small', 'color': 'red'},
                'blue cup': {'type': 'graspable', 'size': 'medium', 'color': 'blue'},
                'book': {'type': 'graspable', 'size': 'medium', 'color': 'brown'},
            },
            'actions': {
                'move_to': 'navigation',
                'go_to': 'navigation',
                'navigate_to': 'navigation',
                'pick_up': 'manipulation',
                'grasp': 'manipulation',
                'place': 'manipulation',
                'drop': 'manipulation',
            }
        }

    def command_callback(self, msg):
        """Process a parsed command"""
        try:
            # Parse the command from JSON string
            command_data = json.loads(msg.data)
            action = command_data.get('action')
            target = command_data.get('target')
            location = command_data.get('location')

            self.get_logger().info(f'Processing command: {action} {target} at {location}')

            # Execute the appropriate action
            if action in ['move_to', 'go_to', 'navigate_to']:
                self.execute_navigation(target or location)
            elif action in ['pick_up', 'grasp']:
                self.execute_manipulation_pick(target)
            elif action == 'place':
                self.execute_manipulation_place(target, location)
            elif action == 'drop':
                self.execute_manipulation_drop(target)

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON command: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def execute_navigation(self, location_name):
        """Execute navigation to a named location"""
        if location_name in self.command_mappings['locations']:
            location = self.command_mappings['locations'][location_name]

            pose_msg = Pose()
            pose_msg.position.x = location['x']
            pose_msg.position.y = location['y']
            pose_msg.position.z = 0.0

            # Convert theta to quaternion
            import math
            theta = location['theta']
            pose_msg.orientation.z = math.sin(theta / 2.0)
            pose_msg.orientation.w = math.cos(theta / 2.0)

            self.nav_pub.publish(pose_msg)
            self.get_logger().info(f'Navigating to {location_name} at ({location["x"]}, {location["y"]})')
        else:
            self.get_logger().warn(f'Unknown location: {location_name}')

    def execute_manipulation_pick(self, object_name):
        """Execute pick up action for a named object"""
        if object_name in self.command_mappings['objects']:
            obj_info = self.command_mappings['objects'][object_name]
            self.get_logger().info(f'Attempting to pick up {object_name} ({obj_info["color"]} {obj_info["size"]})')

            # In a real system, this would involve:
            # 1. Finding the object using perception
            # 2. Planning a grasp
            # 3. Executing the grasp
            # For now, we'll just log the action
        else:
            self.get_logger().warn(f'Unknown object: {object_name}')

    def execute_manipulation_place(self, object_name, location_name):
        """Execute place action"""
        self.get_logger().info(f'Attempting to place {object_name} at {location_name}')
        # Similar to pick_up but for placing

    def execute_manipulation_drop(self, object_name):
        """Execute drop action"""
        self.get_logger().info(f'Dropping {object_name}')
        # Open gripper to drop object

def main():
    rclpy.init()
    mapper_node = CommandMapperNode()

    try:
        rclpy.spin(mapper_node)
    except KeyboardInterrupt:
        pass
    finally:
        mapper_node.destroy_node()
        rclpy.shutdown()
```

## Integration with Perception Systems

Here's how to integrate NLP with perception to understand spatial relationships:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
import re

class SpatialLanguageProcessor(Node):
    def __init__(self):
        super().__init__('spatial_language_processor')

        # Subscribe to language commands and perception data
        self.command_sub = self.create_subscription(
            String,
            '/spatial_commands',
            self.spatial_command_callback,
            10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/perception/detections',
            self.detections_callback,
            10
        )

        # Publisher for spatially-aware commands
        self.spatial_action_pub = self.create_publisher(String, '/spatial_actions', 10)

        # Store detected objects
        self.detected_objects = {}

    def detections_callback(self, msg):
        """Update detected objects from perception system"""
        self.detected_objects = {}

        for detection in msg.detections:
            if detection.results:
                obj_id = detection.results[0].id
                if obj_id not in self.detected_objects:
                    self.detected_objects[obj_id] = {
                        'bbox': detection.bbox,
                        'center': Point(
                            x=detection.bbox.center.x,
                            y=detection.bbox.center.y,
                            z=0.0
                        )
                    }

    def spatial_command_callback(self, msg):
        """Process spatial language commands"""
        command = msg.data.lower()

        # Look for spatial relationships in the command
        spatial_patterns = [
            (r'pick up the (.+?) (?:on|at|by|next to|near) the (.+)', 'spatial_pick'),
            (r'go (?:to|by|near) the (.+?) (?:by|next to|near) the (.+)', 'spatial_nav'),
            (r'put the (.+?) (?:on|at|by) the (.+?) (?:by|next to|near) the (.+)', 'spatial_place'),
        ]

        for pattern, action_type in spatial_patterns:
            match = re.search(pattern, command)
            if match:
                if action_type == 'spatial_pick':
                    target_obj = match.group(1)
                    reference_obj = match.group(2)
                    self.process_spatial_pick(target_obj, reference_obj)
                elif action_type == 'spatial_nav':
                    target_loc = match.group(1)
                    reference_obj = match.group(2)
                    self.process_spatial_navigation(target_loc, reference_obj)
                elif action_type == 'spatial_place':
                    obj_to_place = match.group(1)
                    target_loc = match.group(2)
                    reference_obj = match.group(3)
                    self.process_spatial_place(obj_to_place, target_loc, reference_obj)
                return

        # If no spatial pattern matched, process as regular command
        self.get_logger().info(f'No spatial pattern found in: {command}')

    def process_spatial_pick(self, target_obj, reference_obj):
        """Process command to pick up object near reference object"""
        if reference_obj in self.detected_objects:
            ref_pos = self.detected_objects[reference_obj]['center']
            self.get_logger().info(f'Picking up {target_obj} near {reference_obj} at ({ref_pos.x}, {ref_pos.y})')

            # In a real system, you'd find the target object near the reference
            # For now, we'll just publish a spatial action
            action_msg = String()
            action_msg.data = f'spatial_pick:{target_obj}:near:{reference_obj}'
            self.spatial_action_pub.publish(action_msg)
        else:
            self.get_logger().warn(f'Reference object {reference_obj} not detected')

    def process_spatial_navigation(self, target_loc, reference_obj):
        """Process command to navigate to location near reference object"""
        if reference_obj in self.detected_objects:
            ref_pos = self.detected_objects[reference_obj]['center']
            self.get_logger().info(f'Navigating to {target_loc} near {reference_obj}')

            action_msg = String()
            action_msg.data = f'spatial_nav:{target_loc}:near:{reference_obj}'
            self.spatial_action_pub.publish(action_msg)

    def process_spatial_place(self, obj_to_place, target_loc, reference_obj):
        """Process command to place object at location near reference object"""
        if reference_obj in self.detected_objects:
            self.get_logger().info(f'Placing {obj_to_place} at {target_loc} near {reference_obj}')

            action_msg = String()
            action_msg.data = f'spatial_place:{obj_to_place}:at:{target_loc}:near:{reference_obj}'
            self.spatial_action_pub.publish(action_msg)

def main():
    rclpy.init()
    spatial_node = SpatialLanguageProcessor()

    try:
        rclpy.spin(spatial_node)
    except KeyboardInterrupt:
        pass
    finally:
        spatial_node.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Natural Language Interfaces

- **Robustness**: Handle ambiguous or unclear commands gracefully
- **Feedback**: Provide clear feedback to users about command understanding
- **Context**: Maintain context for multi-turn conversations
- **Error Recovery**: Implement strategies for handling misrecognition
- **Privacy**: Consider privacy implications of always-listening systems
- **Localization**: Adapt to different languages and accents as needed

## Next Steps

In the next chapter, we'll explore how to map natural language commands to specific robot actions and create sophisticated command interpretation systems.