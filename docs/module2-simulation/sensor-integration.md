---
title: Sensor Integration in Simulation
sidebar_label: Sensor Integration
---

# Sensor Integration in Simulation

## Introduction to Sensor Simulation

Sensor simulation in Gazebo allows you to test perception algorithms and sensor processing pipelines in a controlled environment. Properly configured sensors provide realistic data that closely matches real-world sensor behavior, including noise and limitations.

### Common Sensor Types

- **Cameras**: RGB, depth, stereo vision
- **LIDAR**: 2D and 3D laser range finders
- **IMU**: Inertial measurement units
- **Force/Torque**: Joint and contact sensors
- **GPS**: Global positioning system

## Adding a Camera Sensor to Your Robot

Here's how to add a camera sensor to your URDF:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot_with_sensors">
  <!-- Robot links and joints as defined previously -->

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joint connecting head to camera -->
  <joint name="head_to_camera" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo camera plugin -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>image_raw</topic_name>
        <camera_info_topic_name>camera_info</camera_info_topic_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## Adding a LIDAR Sensor

Here's how to add a 2D LIDAR sensor:

```xml
<!-- LIDAR link -->
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
    <material name="silver">
      <color rgba="0.7 0.7 0.7 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.2"/>
    <inertial ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<!-- Joint connecting torso to LIDAR -->
<joint name="torso_to_lidar" type="fixed">
  <parent link="torso"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
</joint>

<!-- Gazebo LIDAR plugin -->
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Adding an IMU Sensor

Here's how to add an IMU sensor:

```xml
<!-- IMU link -->
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<!-- Joint connecting torso to IMU -->
<joint name="torso_to_imu" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>

<!-- Gazebo IMU plugin -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Processing Sensor Data in ROS 2

Here's an example of how to process sensor data in ROS 2:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create CvBridge for image processing
        self.bridge = CvBridge()

        # Subscribe to sensor topics
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Create publisher for processed data
        self.processed_pub = self.create_publisher(
            Image,
            '/processed_image',
            10
        )

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image (example: edge detection)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to ROS Image and publish
        processed_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
        self.processed_pub.publish(processed_msg)

        self.get_logger().info('Processed image from camera')

    def scan_callback(self, msg):
        # Process LIDAR data
        # Find minimum distance in the front 30 degrees
        front_scan = msg.ranges[:15] + msg.ranges[-15:]  # Approximate front 30 degrees
        min_distance = min([r for r in front_scan if r > msg.range_min and r < msg.range_max], default=float('inf'))

        if min_distance < 1.0:  # If obstacle within 1 meter
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')
        else:
            self.get_logger().info(f'Clear path: {min_distance:.2f}m ahead')

    def imu_callback(self, msg):
        # Process IMU data
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        self.get_logger().info(
            f'IMU: Roll={orientation.x:.2f}, Pitch={orientation.y:.2f}, '
            f'Yaw={orientation.z:.2f}'
        )

def main():
    rclpy.init()
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
```

## Sensor Noise Models

Realistic sensor simulation requires proper noise modeling:

- **Camera**: Gaussian noise in image data, lens distortion
- **LIDAR**: Range noise, angular resolution limitations
- **IMU**: Bias drift, Gaussian noise in measurements
- **GPS**: Position accuracy, update rate limitations

## Best Practices

- Always include realistic noise models in simulation
- Validate sensor simulation against real hardware when possible
- Use appropriate update rates for different sensor types
- Consider computational cost of high-resolution sensors
- Test perception algorithms with noisy data to ensure robustness

## Next Steps

In the next module, we'll explore NVIDIA Isaac Sim for more advanced AI-based robotics simulation and perception pipeline development.