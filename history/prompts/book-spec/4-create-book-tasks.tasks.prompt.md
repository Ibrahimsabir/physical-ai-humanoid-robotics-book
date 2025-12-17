---
id: 4
title: Create Book Tasks
stage: tasks
date: 2025-12-17
surface: agent
model: Claude Sonnet 4.5
feature: book-spec
branch: 1-book-spec
user: Claude Code User
command: /sp.tasks
labels: [tasks, robotics, ai, book, curriculum, docusaurus]
links:
  spec: specs/1-book-spec/spec.md
  plan: specs/1-book-spec/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-book-spec/tasks.md
tests:
 - none
---

## Prompt

You are an expert Spec-Kit Plus workflow designer and AI project manager.

Your task is to generate a **Spec-Kit Plus Task File** that governs the execution of an AI-driven technical book project.

This task file must strictly follow:
- `constitution.md` (supreme governing rules)
- `book.spec.md` (structural and module specifications)
- `writing-plan.md` (execution strategy)

Treat all three as immutable law.

────────────────────────────────────────────

## Project
Physical AI & Humanoid Robotics: Embodied Intelligence in the Real World

## Role of the Task File
This file defines:
- The ordered list of executable tasks
- Task dependencies
- Preconditions and postconditions
- Allowed outputs per task
- Validation criteria before moving to the next task

This file controls **what Claude is allowed to generate next**.

This is NOT a plan and NOT content.
This is an **execution contract**.

────────────────────────────────────────────

## Task Design Rules

- Tasks must be sequential and dependency-aware
- No task may begin unless its prerequisites are satisfied
- Each task must produce a concrete artifact
- Tasks must map directly to Docusaurus-ready outputs
- Tasks must prevent scope creep and premature writing

────────────────────────────────────────────

## Required Task Phases

Define tasks across the following phases:

### Phase 0 — Repository & Environment Validation
Tasks related to:
- Repo structure
- Presence of spec files
- Docusaurus initialization readiness

---

### Phase 1 — Specification Lock-in
Tasks that:
- Validate constitution and book specs
- Freeze scope
- Prevent structural changes beyond this point

---

### Phase 2 — Module Specification Expansion
Tasks for:
- Creating individual module spec files
- Validating module boundaries
- Approving module readiness for writing

---

### Phase 3 — Chapter & Week Specification
Tasks for:
- Week-level spec generation
- Chapter contracts
- Learning outcome verification

---

### Phase 4 — Chapter Content Generation
Tasks that:
- Allow Claude to write chapters only after specs exist
- Enforce markdown and Docusaurus rules
- Require validation before Git commits

---

### Phase 5 — Capstone Assembly
Tasks for:
- Integrating modules
- Validating end-to-end humanoid system narrative
- Ensuring capstone integrity rules are satisfied

---

### Phase 6 — Review, Build & Publish
Tasks for:
- Cross-chapter consistency checks
- Docusaurus build validation
- GitHub Pages deployment readiness

────────────────────────────────────────────

## Task Definition Format

Each task MUST include:
- Task ID
- Task Name
- Phase
- Description
- Inputs (files or artifacts required)
- Outputs (files or artifacts produced)
- Preconditions
- Postconditions
- Validation Checklist
- Failure Conditions

────────────────────────────────────────────

## Enforcement Rules

- Claude must refuse to execute tasks out of order
- Claude must refuse to generate content without an active task
- Any artifact not produced by a defined task is invalid
- Task completion requires passing validation

────────────────────────────────────────────

## Output Rules
- Output **only** the task file content
- Use structured Markdown
- No explanations
- No commentary
- No emojis
- No filler text

Generate the task file as a strict execution contract suitable for Spec-Kit Plus.

## Response snapshot

Created comprehensive task file for Physical AI & Humanoid Robotics book with 214 specific tasks across 6 phases:

- Phase 0: Repository & Environment Validation (8 tasks)
- Phase 1: Specification Lock-in (8 tasks)
- Phase 2: Module Specification Expansion (56 tasks for 5 modules)
- Phase 3: Chapter & Week Specification (98 tasks for 13 weeks)
- Phase 4: Chapter Content Generation (34 tasks for content generation)
- Phase 5: Capstone Assembly (12 tasks for integration)
- Phase 6: Review, Build & Publish (16 tasks for validation and deployment)

Each task follows the required format with checkboxes, IDs, parallel markers [P], story labels [US], and specific file paths. Tasks are organized with proper dependencies and validation checkpoints.

## Outcome

- ✅ Impact: Complete execution contract ready for AI-assisted book development
- 🧪 Tests: none
- 📁 Files: specs/1-book-spec/tasks.md
- 🔁 Next prompts: Ready for task execution following the defined sequence
- 🧠 Reflection: Task file enforces specification-first workflow and prevents scope creep

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - task file complete and validated
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin executing tasks in sequence