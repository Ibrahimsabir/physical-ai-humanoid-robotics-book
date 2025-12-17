---
title: ROS 2 Fundamentals
sidebar_label: ROS 2 Fundamentals
---

# ROS 2 Fundamentals

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Concepts

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Messages**: ROS data types used when publishing or subscribing to a Topic
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous communication for long-running tasks

## Setting Up Your ROS 2 Environment

Before we dive into creating nodes, you'll need to set up your ROS 2 environment. We'll be using ROS 2 Humble Hawksbill, which is an LTS version with long-term support.

```bash
# Source the ROS 2 installation
source /opt/ros/humble/setup.bash

# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Creating Your First Publisher Node

Let's create a simple publisher node that sends joint position data:

```python
# publisher_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Your First Subscriber Node

Now let's create a subscriber node that receives the joint position data:

```python
# subscriber_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Running the Publisher and Subscriber

To run your nodes:

```bash
# Terminal 1
ros2 run your_package_name minimal_publisher

# Terminal 2
ros2 run your_package_name minimal_subscriber
```

## Key Takeaways

- ROS 2 provides a distributed communication framework for robotics
- Nodes communicate through topics using publisher/subscriber pattern
- Services provide request/response communication
- Actions handle long-running tasks with feedback
- Always follow ROS 2 naming conventions for packages and nodes

## Next Steps

In the next chapter, we'll explore how to model robots using URDF (Unified Robot Description Format) and understand robot kinematics.