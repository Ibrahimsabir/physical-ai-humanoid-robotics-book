import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics book
  bookSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'intro',
        {
          type: 'category',
          label: 'Module 1: Robotic Nervous System (ROS 2)',
          items: [
            'module1-ros2/ros2-fundamentals',
            'module1-ros2/robot-modeling',
            'module1-ros2/control-systems'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: Digital Twin (Gazebo & Unity)',
          items: [
            'module2-simulation/simulation-setup',
            'module2-simulation/sensor-integration',
            'module2-simulation/advanced-visualization'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
          items: [
            'module3-ai-brain/isaac-sim-environment',
            'module3-ai-brain/perception-pipelines',
            'module3-ai-brain/navigation-systems',
            'module3-ai-brain/manipulation-systems'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA)',
          items: [
            'module4-vla/natural-language-processing',
            'module4-vla/action-mapping',
            'module4-vla/context-awareness'
          ],
        },
        {
          type: 'category',
          label: 'Module 5: Capstone - Autonomous Humanoid',
          items: [
            'module5-capstone/system-integration',
            'module5-capstone/capstone-project'
          ],
        },
        {
          type: 'category',
          label: 'Integration Guides',
          items: [
            'integration/ros2-simulation-integration',
            'integration/simulation-ai-integration',
            'integration/ai-vla-integration'
          ],
        }
      ],
    },
  ],
};

export default sidebars;
