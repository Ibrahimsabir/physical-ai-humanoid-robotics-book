---
description: "Task list for Physical AI & Humanoid Robotics book"
---

# Tasks: Physical AI & Humanoid Robotics Book

**Input**: Design documents from `/specs/1-book-spec/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Book Project**: `docs/`, `src/`, `tests/` at repository root
- **Module Structure**: `docs/module1/`, `docs/module2/`, etc.
- **Docusaurus**: `docs/`, `src/`, `blog/`, `pages/`, `static/` at repository root
- Paths shown below assume book project structure - adjust based on plan.md structure

## Book Constitution Compliance

All tasks must ensure compliance with the Physical AI & Humanoid Robotics book constitution:
- **Specification-First Rule**: All content must be based on a corresponding spec
- **Modular Architecture**: Content must fit within the book module structure (ROS 2, Digital Twin, AI-Robot Brain, VLA, Capstone)
- **Physical AI First-Principles**: All explanations must respect physical laws and highlight real-world constraints
- **Simulation-to-Reality Continuity**: Simulated concepts must map to real hardware
- **Writing Style Rules**: Content must be clear, precise, technical but readable
- **Docusaurus Compatibility**: All content is markdown-friendly and sidebar-ready

## Phase 0: Repository & Environment Validation

**Purpose**: Validate repository structure and ensure all specification files exist

- [X] T001 Verify repository structure matches Docusaurus template
- [X] T002 [P] Validate presence of constitution.md file
- [X] T003 [P] Validate presence of book spec file at specs/1-book-spec/spec.md
- [X] T004 [P] Validate presence of implementation plan at specs/1-book-spec/plan.md
- [X] T005 [P] Validate presence of data model at specs/1-book-spec/data-model.md
- [X] T006 [P] Validate presence of research file at specs/1-book-spec/research.md
- [X] T007 [P] Validate presence of quickstart guide at specs/1-book-spec/quickstart.md
- [X] T008 [P] Validate presence of contracts directory at specs/1-book-spec/contracts/

**Checkpoint**: All specification files validated and repository structure confirmed

---

## Phase 1: Specification Lock-in

**Purpose**: Validate constitution and book specs, freeze scope, prevent structural changes

- [X] T009 Review constitution.md for Physical AI & Humanoid Robotics project requirements
- [X] T010 Validate book specification against constitution compliance requirements
- [X] T011 [P] Verify all 5 modules defined in specification (ROS 2, Digital Twin, AI-Robot Brain, VLA, Capstone)
- [X] T012 [P] Confirm 13-week curriculum breakdown exists in specification
- [X] T013 [P] Validate Docusaurus mapping structure in specification
- [X] T014 [P] Confirm hardware and deployment specifications are defined
- [X] T015 [P] Verify enforcement rules are properly specified
- [X] T016 Freeze scope based on validated specification - no structural changes allowed beyond this point

**Checkpoint**: Specification frozen and locked for development

---

## Phase 2: Module Specification Expansion

**Purpose**: Create individual module spec files, validate module boundaries, approve module readiness for writing

### Module 1 — The Robotic Nervous System (ROS 2)

- [X] T017 [P] Create detailed specification for Module 1 at docs/module1-ros2/spec.md
- [X] T018 [P] Define ROS 2 core concepts for Module 1 (nodes, topics, services, actions, rclpy, URDF)
- [X] T019 [P] Specify inputs and outputs for Module 1
- [X] T020 [P] Define required tools for Module 1 (ROS 2 Humble, Python 3.10+, rclpy)
- [X] T021 [P] Create artifact list for Module 1 (workspace, URDF files, launch files)
- [X] T022 [P] Define dependencies for Module 1 (none - foundational module)
- [X] T023 [P] Create validation criteria for Module 1
- [X] T024 Approve Module 1 specification for content generation

### Module 2 — The Digital Twin (Gazebo & Unity)

- [X] T025 [P] Create detailed specification for Module 2 at docs/module2-simulation/spec.md
- [X] T026 [P] Define simulation core concepts for Module 2 (physics, sensors, URDF/SDF, visualization)
- [X] T027 [P] Specify inputs and outputs for Module 2
- [X] T028 [P] Define required tools for Module 2 (Gazebo, Unity, physics engines)
- [X] T029 [P] Create artifact list for Module 2 (world files, sensor configs, launch files)
- [X] T030 [P] Define dependencies for Module 2 (Module 1 - ROS 2 infrastructure)
- [X] T031 [P] Create validation criteria for Module 2
- [X] T032 Approve Module 2 specification for content generation

### Module 3 — The AI-Robot Brain (NVIDIA Isaac)

- [X] T033 [P] Create detailed specification for Module 3 at docs/module3-ai-brain/spec.md
- [X] T034 [P] Define AI core concepts for Module 3 (Isaac Sim, Isaac ROS, VSLAM, Nav2, perception)
- [X] T035 [P] Specify inputs and outputs for Module 3
- [X] T036 [P] Define required tools for Module 3 (Isaac Sim, Isaac ROS, Nav2)
- [X] T037 [P] Create artifact list for Module 3 (scenes, pipelines, configs)
- [X] T038 [P] Define dependencies for Module 3 (Module 1, Module 2)
- [X] T039 [P] Create validation criteria for Module 3
- [X] T040 Approve Module 3 specification for content generation

### Module 4 — Vision-Language-Action (VLA)

- [X] T041 [P] Create detailed specification for Module 4 at docs/module4-vla/spec.md
- [X] T042 [P] Define VLA core concepts for Module 4 (Whisper, LLM planning, NLP)
- [X] T043 [P] Specify inputs and outputs for Module 4
- [X] T044 [P] Define required tools for Module 4 (OpenAI API, Whisper, NLP libraries)
- [X] T045 [P] Create artifact list for Module 4 (pipelines, mapping systems)
- [X] T046 [P] Define dependencies for Module 4 (Module 1, Module 3)
- [X] T047 [P] Create validation criteria for Module 4
- [X] T048 Approve Module 4 specification for content generation

### Module 5 — Capstone Module — Autonomous Humanoid

- [X] T049 [P] Create detailed specification for Module 5 at docs/module5-capstone/spec.md
- [X] T050 [P] Define capstone core concepts for Module 5 (integration, coordination, deployment)
- [X] T051 [P] Specify inputs and outputs for Module 5
- [X] T052 [P] Define required tools for Module 5 (all tools from previous modules)
- [X] T053 [P] Create artifact list for Module 5 (integrated system, benchmarks)
- [X] T054 [P] Define dependencies for Module 5 (all previous modules)
- [X] T055 [P] Create validation criteria for Module 5
- [X] T056 Approve Module 5 specification for content generation

**Checkpoint**: All 5 module specifications created and approved for content generation

---

## Phase 3: Chapter & Week Specification

**Purpose**: Generate week-level specs, chapter contracts, and verify learning outcomes

### Week 1: ROS 2 Fundamentals

- [X] T057 [P] [US1] Create week 1 specification at docs/module1-ros2/week1-spec.md
- [X] T058 [P] [US1] Define learning objectives for Week 1 (ROS 2 architecture, basic nodes)
- [X] T059 [P] [US1] Specify required knowledge entering Week 1 (Python proficiency)
- [X] T060 [P] [US1] Define new capabilities acquired in Week 1 (create ROS 2 publisher/subscriber nodes)
- [X] T061 [P] [US1] Create deliverable artifact spec for Week 1 (basic ROS 2 workspace)
- [X] T062 [P] [US1] Create chapter contract for Week 1 at docs/module1-ros2/ros2-fundamentals.md
- [X] T063 [P] [US1] Validate learning outcome for Week 1

### Week 2: Robot Modeling

- [X] T064 [P] [US1] Create week 2 specification at docs/module1-ros2/week2-spec.md
- [X] T065 [P] [US1] Define learning objectives for Week 2 (URDF robot models, transforms)
- [X] T066 [P] [US1] Specify required knowledge entering Week 2 (Week 1 ROS 2 fundamentals)
- [X] T067 [P] [US1] Define new capabilities acquired in Week 2 (define robot kinematics)
- [X] T068 [P] [US1] Create deliverable artifact spec for Week 2 (URDF robot model)
- [X] T069 [P] [US1] Create chapter contract for Week 2 at docs/module1-ros2/robot-modeling.md
- [X] T070 [P] [US1] Validate learning outcome for Week 2

### Week 3: ROS 2 Control Systems

- [X] T071 [P] [US1] Create week 3 specification at docs/module1-ros2/week3-spec.md
- [X] T072 [P] [US1] Define learning objectives for Week 3 (robot control with ROS 2)
- [X] T073 [P] [US1] Specify required knowledge entering Week 3 (Week 1-2 skills)
- [X] T074 [P] [US1] Define new capabilities acquired in Week 3 (control robot joints)
- [X] T075 [P] [US1] Create deliverable artifact spec for Week 3 (ROS 2 control system)
- [X] T076 [P] [US1] Create chapter contract for Week 3 at docs/module1-ros2/control-systems.md
- [X] T077 [P] [US1] Validate learning outcome for Week 3

### Week 4: Simulation Setup

- [X] T078 [P] [US2] Create week 4 specification at docs/module2-simulation/week4-spec.md
- [X] T079 [P] [US2] Define learning objectives for Week 4 (Gazebo simulation environment)
- [X] T080 [P] [US2] Specify required knowledge entering Week 4 (Week 1-3 ROS 2 skills)
- [X] T081 [P] [US2] Define new capabilities acquired in Week 4 (run robot in physics simulation)
- [X] T082 [P] [US2] Create deliverable artifact spec for Week 4 (Gazebo simulation)
- [X] T083 [P] [US2] Create chapter contract for Week 4 at docs/module2-simulation/simulation-setup.md
- [X] T084 [P] [US2] Validate learning outcome for Week 4

### Week 5: Sensor Integration

- [X] T085 [P] [US2] Create week 5 specification at docs/module2-simulation/week5-spec.md
- [X] T086 [P] [US2] Define learning objectives for Week 5 (add sensors to simulation)
- [X] T087 [P] [US2] Specify required knowledge entering Week 5 (Week 4 simulation setup)
- [X] T088 [P] [US2] Define new capabilities acquired in Week 5 (process simulated sensor data)
- [X] T089 [P] [US2] Create deliverable artifact spec for Week 5 (robot with simulated sensors)
- [X] T090 [P] [US2] Create chapter contract for Week 5 at docs/module2-simulation/sensor-integration.md
- [X] T091 [P] [US2] Validate learning outcome for Week 5

### Week 6: Isaac Sim Introduction

- [X] T092 [P] [US3] Create week 6 specification at docs/module3-ai-brain/week6-spec.md
- [X] T093 [P] [US3] Define learning objectives for Week 6 (NVIDIA Isaac Sim environment)
- [X] T094 [P] [US3] Specify required knowledge entering Week 6 (Week 4-5 simulation skills)
- [X] T095 [P] [US3] Define new capabilities acquired in Week 6 (run robot in Isaac Sim)
- [X] T096 [P] [US3] Create deliverable artifact spec for Week 6 (Isaac Sim environment)
- [X] T097 [P] [US3] Create chapter contract for Week 6 at docs/module3-ai-brain/isaac-sim-environment.md
- [X] T098 [P] [US3] Validate learning outcome for Week 6

### Week 7: Perception Pipelines

- [X] T099 [P] [US3] Create week 7 specification at docs/module3-ai-brain/week7-spec.md
- [X] T100 [P] [US3] Define learning objectives for Week 7 (AI-based perception systems)
- [X] T101 [P] [US3] Specify required knowledge entering Week 7 (Week 6 Isaac Sim skills)
- [X] T102 [P] [US3] Define new capabilities acquired in Week 7 (object detection and recognition)
- [X] T103 [P] [US3] Create deliverable artifact spec for Week 7 (perception pipeline)
- [X] T104 [P] [US3] Create chapter contract for Week 7 at docs/module3-ai-brain/perception-pipelines.md
- [X] T105 [P] [US3] Validate learning outcome for Week 7

### Week 8: Navigation Systems

- [X] T106 [P] [US3] Create week 8 specification at docs/module3-ai-brain/week8-spec.md
- [X] T107 [P] [US3] Define learning objectives for Week 8 (autonomous navigation)
- [X] T108 [P] [US3] Specify required knowledge entering Week 8 (Week 7 perception skills)
- [X] T109 [P] [US3] Define new capabilities acquired in Week 8 (robot navigation with Nav2)
- [X] T110 [P] [US3] Create deliverable artifact spec for Week 8 (navigation system)
- [X] T111 [P] [US3] Create chapter contract for Week 8 at docs/module3-ai-brain/navigation-systems.md
- [X] T112 [P] [US3] Validate learning outcome for Week 8

### Week 9: Manipulation Systems

- [X] T113 [P] [US3] Create week 9 specification at docs/module3-ai-brain/week9-spec.md
- [X] T114 [P] [US3] Define learning objectives for Week 9 (robot manipulation capabilities)
- [X] T115 [P] [US3] Specify required knowledge entering Week 9 (Week 7-8 perception and navigation)
- [X] T116 [P] [US3] Define new capabilities acquired in Week 9 (object manipulation and grasping)
- [X] T117 [P] [US3] Create deliverable artifact spec for Week 9 (manipulation control system)
- [X] T118 [P] [US3] Create chapter contract for Week 9 at docs/module3-ai-brain/manipulation-systems.md
- [X] T119 [P] [US3] Validate learning outcome for Week 9

### Week 10: Natural Language Processing

- [X] T120 [P] [US4] Create week 10 specification at docs/module4-vla/week10-spec.md
- [X] T121 [P] [US4] Define learning objectives for Week 10 (speech recognition and understanding)
- [X] T122 [P] [US4] Specify required knowledge entering Week 10 (Week 1-9 robot systems knowledge)
- [X] T123 [P] [US4] Define new capabilities acquired in Week 10 (process natural language commands)
- [X] T124 [P] [US4] Create deliverable artifact spec for Week 10 (natural language processing system)
- [X] T125 [P] [US4] Create chapter contract for Week 10 at docs/module4-vla/natural-language-processing.md
- [X] T126 [P] [US4] Validate learning outcome for Week 10

### Week 11: Action Mapping

- [X] T127 [P] [US4] Create week 11 specification at docs/module4-vla/week11-spec.md
- [X] T128 [P] [US4] Define learning objectives for Week 11 (map language to robot actions)
- [X] T129 [P] [US4] Specify required knowledge entering Week 11 (Week 10 NLP skills)
- [X] T130 [P] [US4] Define new capabilities acquired in Week 11 (convert commands to action sequences)
- [X] T131 [P] [US4] Create deliverable artifact spec for Week 11 (language-to-action mapping system)
- [X] T132 [P] [US4] Create chapter contract for Week 11 at docs/module4-vla/action-mapping.md
- [X] T133 [P] [US4] Validate learning outcome for Week 11

### Week 12: System Integration

- [X] T134 [P] [US5] Create week 12 specification at docs/module5-capstone/week12-spec.md
- [X] T135 [P] [US5] Define learning objectives for Week 12 (integrate all modules)
- [X] T136 [P] [US5] Specify required knowledge entering Week 12 (all previous weeks)
- [X] T137 [P] [US5] Define new capabilities acquired in Week 12 (end-to-end system operation)
- [X] T138 [P] [US5] Create deliverable artifact spec for Week 12 (integrated robot system)
- [X] T139 [P] [US5] Create chapter contract for Week 12 at docs/module5-capstone/system-integration.md
- [X] T140 [P] [US5] Validate learning outcome for Week 12

### Week 13: Capstone Project

- [X] T141 [P] [US5] Create week 13 specification at docs/module5-capstone/week13-spec.md
- [X] T142 [P] [US5] Define learning objectives for Week 13 (complete autonomous humanoid project)
- [X] T143 [P] [US5] Specify required knowledge entering Week 13 (all previous weeks)
- [X] T144 [P] [US5] Define new capabilities acquired in Week 13 (full system demonstration)
- [X] T145 [P] [US5] Create deliverable artifact spec for Week 13 (complete capstone demonstration)
- [X] T146 [P] [US5] Create chapter contract for Week 13 at docs/module5-capstone/capstone-project.md
- [X] T147 [P] [US5] Validate learning outcome for Week 13

**Checkpoint**: All 13 weeks of specifications and chapter contracts created and validated

---

## Phase 4: Chapter Content Generation

**Purpose**: Allow Claude to write chapters only after specs exist, enforce markdown and Docusaurus rules, require validation before Git commits

### Module 1 Content Generation

- [X] T148 [P] [US1] Generate Week 1 content for ROS 2 fundamentals at docs/module1-ros2/ros2-fundamentals.md
- [X] T149 [P] [US1] Generate Week 2 content for robot modeling at docs/module1-ros2/robot-modeling.md
- [X] T150 [P] [US1] Generate Week 3 content for control systems at docs/module1-ros2/control-systems.md
- [X] T151 [P] [US1] Validate Module 1 content against specification requirements
- [X] T152 [P] [US1] Ensure Module 1 content follows Docusaurus compatibility rules
- [X] T153 [P] [US1] Verify Module 1 content respects Physical AI first-principles
- [X] T154 [P] [US1] Confirm Module 1 content demonstrates simulation-to-reality continuity

### Module 2 Content Generation

- [X] T155 [P] [US2] Generate Week 4 content for simulation setup at docs/module2-simulation/simulation-setup.md
- [X] T156 [P] [US2] Generate Week 5 content for sensor integration at docs/module2-simulation/sensor-integration.md
- [X] T157 [P] [US2] Generate Week 5 content for advanced visualization at docs/module2-simulation/advanced-visualization.md
- [X] T158 [P] [US2] Validate Module 2 content against specification requirements
- [X] T159 [P] [US2] Ensure Module 2 content follows Docusaurus compatibility rules
- [X] T160 [P] [US2] Verify Module 2 content respects Physical AI first-principles
- [X] T161 [P] [US2] Confirm Module 2 content demonstrates simulation-to-reality continuity

### Module 3 Content Generation

- [X] T162 [P] [US3] Generate Week 6 content for Isaac Sim environment at docs/module3-ai-brain/isaac-sim-environment.md
- [X] T163 [P] [US3] Generate Week 7 content for perception pipelines at docs/module3-ai-brain/perception-pipelines.md
- [X] T164 [P] [US3] Generate Week 8 content for navigation systems at docs/module3-ai-brain/navigation-systems.md
- [X] T165 [P] [US3] Generate Week 9 content for manipulation systems at docs/module3-ai-brain/manipulation-systems.md
- [X] T166 [P] [US3] Validate Module 3 content against specification requirements
- [X] T167 [P] [US3] Ensure Module 3 content follows Docusaurus compatibility rules
- [X] T168 [P] [US3] Verify Module 3 content respects Physical AI first-principles
- [X] T169 [P] [US3] Confirm Module 3 content demonstrates simulation-to-reality continuity

### Module 4 Content Generation

- [X] T170 [P] [US4] Generate Week 10 content for natural language processing at docs/module4-vla/natural-language-processing.md
- [X] T171 [P] [US4] Generate Week 11 content for action mapping at docs/module4-vla/action-mapping.md
- [X] T172 [P] [US4] Generate Week 11 content for context awareness at docs/module4-vla/context-awareness.md
- [X] T173 [P] [US4] Validate Module 4 content against specification requirements
- [X] T174 [P] [US4] Ensure Module 4 content follows Docusaurus compatibility rules
- [X] T175 [P] [US4] Verify Module 4 content respects Physical AI first-principles
- [X] T176 [P] [US4] Confirm Module 4 content demonstrates simulation-to-reality continuity

### Module 5 Content Generation

- [X] T177 [P] [US5] Generate Week 12 content for system integration at docs/module5-capstone/system-integration.md
- [X] T178 [P] [US5] Generate Week 13 content for capstone project at docs/module5-capstone/capstone-project.md
- [X] T179 [P] [US5] Validate Module 5 content against specification requirements
- [X] T180 [P] [US5] Ensure Module 5 content follows Docusaurus compatibility rules
- [X] T181 [P] [US5] Verify Module 5 content respects Physical AI first-principles
- [X] T182 [P] [US5] Confirm Module 5 content demonstrates simulation-to-reality continuity

**Checkpoint**: All chapter content generated and validated according to specifications

---

## Phase 5: Capstone Assembly

**Purpose**: Integrate modules, validate end-to-end humanoid system narrative, ensure capstone integrity rules are satisfied

- [ ] T183 Integrate Module 1 (ROS 2) content with Module 2 (Simulation) content
- [ ] T184 Integrate Module 2 (Simulation) content with Module 3 (AI Brain) content
- [ ] T185 Integrate Module 3 (AI Brain) content with Module 4 (VLA) content
- [ ] T186 Integrate Module 4 (VLA) content with Module 5 (Capstone) content
- [ ] T187 Validate end-to-end humanoid system narrative across all modules
- [ ] T188 Verify capstone project integrates all previous modules as specified
- [ ] T189 Test complete system response to natural language commands in simulation
- [ ] T190 Validate deployment configuration for Jetson Orin as specified
- [ ] T191 Confirm all modules work in coordination as per capstone requirements
- [ ] T192 Verify sim-to-real model transfer capabilities are properly documented

**Checkpoint**: Complete end-to-end system validated and integrated

---

## Phase 6: Review, Build & Publish

**Purpose**: Perform cross-chapter consistency checks, Docusaurus build validation, prepare for GitHub Pages deployment

### Cross-Chapter Consistency Checks

- [ ] T193 [P] Verify terminology consistency across all modules and chapters
- [ ] T194 [P] Check writing style consistency across all modules
- [ ] T195 [P] Validate formatting consistency across all chapters
- [ ] T196 [P] Confirm code example consistency across all modules
- [ ] T197 [P] Verify cross-module reference accuracy

### Docusaurus Build Validation

- [ ] T198 Run Docusaurus build to validate all pages build without errors
- [ ] T199 Verify navigation works correctly across all modules
- [ ] T200 Check all links are functional and point to correct locations
- [ ] T201 Validate sidebar structure matches specification requirements
- [ ] T202 Test search functionality across all content
- [ ] T203 Verify responsive design works on different screen sizes

### GitHub Pages Deployment Preparation

- [ ] T204 [P] Create deployment configuration files
- [ ] T205 [P] Validate GitHub Actions workflow for deployment
- [ ] T206 [P] Prepare versioning and tagging strategy
- [ ] T207 [P] Create change log documentation
- [ ] T208 [P] Prepare post-publish maintenance documentation

### Final Validation

- [ ] T209 Conduct final review against original book specification
- [ ] T210 Verify all learning objectives are met across all modules
- [ ] T211 Confirm all 13 weeks of content are complete and functional
- [ ] T212 Validate all hardware and deployment specifications are documented
- [ ] T213 Final constitution compliance check
- [ ] T214 Prepare for GitHub Pages deployment

**Checkpoint**: Complete book validated and ready for publication

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 0** (Repository Validation): No dependencies - can start immediately
- **Phase 1** (Specification Lock-in): Depends on Phase 0 completion
- **Phase 2** (Module Expansion): Depends on Phase 1 completion
- **Phase 3** (Chapter Specification): Depends on Phase 2 completion
- **Phase 4** (Content Generation): Depends on Phase 3 completion
- **Phase 5** (Capstone Assembly): Depends on Phase 4 completion
- **Phase 6** (Review & Publish): Depends on Phase 5 completion

### Parallel Opportunities

- All Phase 0 tasks marked [P] can run in parallel
- All Phase 2 module specification tasks marked [P] can run in parallel
- All Phase 3 week specification tasks within each module can run in parallel
- All Phase 4 content generation tasks marked [P] can run in parallel
- All Phase 6 consistency checks marked [P] can run in parallel

---

## Implementation Strategy

### MVP First (Module 1 Only)

1. Complete Phase 0: Repository validation
2. Complete Phase 1: Specification lock-in
3. Complete Phase 2: Module 1 specification
4. Complete Phase 3: Module 1 week specifications
5. Complete Phase 4: Module 1 content generation
6. **STOP and VALIDATE**: Test Module 1 independently
7. Deploy/demonstrate if ready

### Incremental Delivery

1. Complete all phases for Module 1 → Test independently → Deploy/Demo (MVP!)
2. Add Module 2 → Test independently → Deploy/Demo
3. Add Module 3 → Test independently → Deploy/Demo
4. Add Module 4 → Test independently → Deploy/Demo
5. Add Module 5 → Test independently → Deploy/Demo
6. Complete capstone integration → Final validation → Publish

### Parallel Team Strategy

With multiple developers:

- Team completes Phases 0-3 together
- Once specifications are complete:
  - Developer A: Module 1 content (T148-T154)
  - Developer B: Module 2 content (T155-T161)
  - Developer C: Module 3 content (T162-T169)
  - Developer D: Module 4 content (T170-T176)
  - Developer E: Module 5 content (T177-T182)
- Modules integrate in Phase 5
- Whole team validates in Phase 6