---
id: 2
title: Create Book Spec
stage: spec
date: 2025-12-17
surface: agent
model: Claude Sonnet 4.5
feature: book-spec
branch: 1-book-spec
user: Claude Code User
command: /sp.specify
labels: [spec, robotics, ai, book, curriculum]
links:
  spec: specs/1-book-spec/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-book-spec/spec.md
 - specs/1-book-spec/checklists/requirements.md
tests:
 - none
---

## Prompt

You are an expert systems architect, curriculum designer, and Spec-Kit Plus author.

Your task is to generate the **primary specification file (`book.spec.md`)** for an AI-authored technical book governed by an existing `constitution.md`.

You MUST strictly follow the rules and principles defined in `constitution.md`.
Treat it as immutable law.

────────────────────────────────────────────

## Project Name
Physical AI & Humanoid Robotics: Embodied Intelligence in the Real World

## Specification Role
This file defines:
- The complete structural specification of the book
- Module boundaries and responsibilities
- Chapter-level contracts
- Learning objectives per module
- Required artifacts per chapter
- Cross-module dependencies

This is a **design contract**, not narrative content.

Do NOT write tutorial text.
Do NOT write explanations.
Define WHAT must exist, not HOW it is implemented.

────────────────────────────────────────────

## Book-Level Specifications

### Target Audience
Advanced undergraduate / graduate-level learners with:
- Python proficiency
- Basic AI / ML background
- No prior robotics experience

### Book Outcome
By the end of the book, the reader must be able to:
- Design and simulate a humanoid robot
- Build ROS 2 control systems
- Perform perception, navigation, and manipulation
- Integrate LLM-based Vision-Language-Action pipelines
- Deploy AI workloads to Jetson edge devices
- Execute a full simulated humanoid capstone

────────────────────────────────────────────

## Module Specifications

Define the following modules EXACTLY as system layers.

For EACH module, include:
- Purpose
- Core Concepts (bullet list)
- Inputs
- Outputs
- Required Tools
- Artifacts Produced
- Dependencies
- Validation Criteria

### Module 1 — The Robotic Nervous System (ROS 2)
(ROS 2 nodes, topics, services, actions, rclpy, URDF)

### Module 2 — The Digital Twin (Gazebo & Unity)
(Physics simulation, sensors, URDF/SDF, visualization)

### Module 3 — The AI-Robot Brain (NVIDIA Isaac)
(Isaac Sim, Isaac ROS, VSLAM, Nav2, perception pipelines)

### Module 4 — Vision-Language-Action (VLA)
(Whisper, LLM planning, natural language → ROS actions)

### Capstone Module — Autonomous Humanoid
(End-to-end integration of all modules)

────────────────────────────────────────────

## Weekly Breakdown Specification

Define a **week-by-week contract** (Weeks 1–13).

For EACH week specify:
- Module alignment
- Learning objectives
- Required knowledge entering the week
- New capabilities acquired
- Deliverable artifact (code, config, simulation, or report)

No week may be empty.
Each week must produce something tangible.

────────────────────────────────────────────

## Hardware & Deployment Specifications

Formally specify:
- Workstation requirements (RTX, RAM, OS)
- Edge deployment constraints (Jetson Orin)
- Simulation vs real-world execution boundaries
- Cloud vs On-Prem tradeoffs
- Sim-to-Real model transfer rules

────────────────────────────────────────────

## Docusaurus Mapping

Define:
- Sidebar structure
- Module → folder mapping
- Chapter → markdown file rules
- Naming conventions

────────────────────────────────────────────

## Enforcement Rules

- This spec governs all future content generation
- Any content not traceable to this spec is invalid
- Claude must refuse to generate content that violates:
  - The constitution
  - This specification

────────────────────────────────────────────

## Output Rules
- Output **ONLY** the contents of `book.spec.md`
- Use clear Markdown structure
- No explanations
- No commentary
- No filler text
- No emojis

Generate the specification as a professional engineering contract.

## Response snapshot

Created comprehensive book specification for Physical AI & Humanoid Robotics curriculum with:

- 5 detailed modules (ROS 2, Simulation, AI-Brain, VLA, Capstone)
- 13-week curriculum breakdown with learning objectives
- Hardware and deployment specifications
- Docusaurus mapping structure
- Complete user scenarios and functional requirements
- Quality checklist validation

## Outcome

- ✅ Impact: Complete specification contract ready for book development
- 🧪 Tests: none
- 📁 Files: specs/1-book-spec/spec.md, specs/1-book-spec/checklists/requirements.md
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Specification aligns with constitution principles and provides clear development roadmap

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - specification complete and validated
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin module planning with /sp.plan