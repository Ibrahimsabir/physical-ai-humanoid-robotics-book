# Quickstart: Physical AI & Humanoid Robotics Book Development

## Prerequisites
- Node.js 18+ installed
- Git for version control
- Python 3.10+ for robotics development
- Access to NVIDIA development tools (Isaac Sim, Isaac ROS)
- ROS 2 Humble Hawksbill installed

## Setup Environment
1. Clone the repository: `git clone <repo-url>`
2. Navigate to project directory: `cd docu-ebook`
3. Install dependencies: `npm install`
4. Verify Docusaurus setup: `npm run start`

## Book Development Workflow
1. **Specification Review**: Review the book specification at `specs/1-book-spec/spec.md`
2. **Module Planning**: Follow the 5-module structure (ROS 2, Simulation, AI Brain, VLA, Capstone)
3. **Chapter Writing**: Write each chapter following the weekly breakdown specification
4. **Validation**: Run Docusaurus build to verify content: `npm run build`
5. **Preview**: Start local server to preview: `npm run start`

## Writing Guidelines
- Follow the Docusaurus markdown conventions
- Ensure all content aligns with the book specification
- Maintain consistency with the physical AI first-principles
- Verify simulation-to-reality continuity in all examples

## Validation Steps
1. Run `npm run build` to check for Docusaurus build errors
2. Verify all links are functional
3. Confirm content follows specification requirements
4. Validate technical accuracy of code examples

## Next Steps
- Begin with Module 1 (ROS 2) content development
- Follow the weekly breakdown from the specification
- Use Claude Code for AI-assisted content generation
- Maintain regular validation through Docusaurus build process