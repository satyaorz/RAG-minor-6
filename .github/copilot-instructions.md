
Purpose
- Provide a concise, structured set of instructions for an AI assistant (Copilot) to read `project_spec.md` and produce an actionable understanding of the project.

How to use
1. Open and read `project_spec.md` in full. Preserve headings and lists.
2. Extract and list: project name, goal(s), target users, main features, inputs/outputs, data sources, constraints, stakeholders, timeline/ deadlines, and any explicit requirements.
3. Identify missing details and produce follow-up questions.
4. Produce three outputs: a short elevator pitch (1 sentence), a one-paragraph summary (3–5 sentences), and a 5-bullet roadmap of next tasks.

Output templates
- Elevator pitch (1 sentence): "<project> — <what it does> for <users> to <benefit>."
- One-paragraph summary (3–5 sentences): describe problem, proposed solution, data/tech, and main constraints/assumptions.
- 5-bullet roadmap: prioritize next steps (e.g., clarify data, set up repo, implement core feature, evaluate, iterate). For each step include estimated effort: Tiny (1-2h), Small (1 day), Medium (3-5 days), Large (>1 week).

Checklist to produce
- Extracted facts: goals, users, features, data, constraints, success metrics.
- Questions: 5 concise follow-ups for unknowns or ambiguities.
- Risks & blockers: 3 bullets.
- Suggested first PR: one-sentence description of a minimal, testable first commit.

Behavior rules
- Prefer concise, actionable language. Use bullet lists for clarity.
- When uncertain, ask a targeted question instead of assuming.
- Do not modify `project_spec.md`; only read and summarize.

Example prompts (for interactive use)
- "Read `project_spec.md` and return: elevator pitch, paragraph summary, 5-bullet roadmap, 5 follow-up questions." 
- "From `project_spec.md`, extract the data sources, required inputs, and suggest a minimal data validation plan." 

Deliverable format (preferred)
- JSON object with keys: `elevator`, `summary`, `roadmap` (array), `questions` (array), `risks` (array), `first_pr` (string). Also provide a human-readable markdown section.

Notes for maintainers
- Update this instruction if `project_spec.md` formatting conventions change.
- Keep templates short; avoid long narrative outputs unless requested.

---
Created to help Copilot quickly understand and act on the project's specification.
