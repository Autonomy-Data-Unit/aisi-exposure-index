You are classifying seniority and management responsibility in a UK job advertisement for a labour-market research project.

Use only the advertisement title and description. Do not infer seniority from salary, occupation, or outside knowledge.

Field definitions:

`seniority_level` — choose exactly one:
- `intern_or_apprentice`: intern, apprentice, placement, trainee apprenticeship.
- `graduate_or_entry_level`: graduate scheme, entry-level, junior trainee, no-experience entry role.
- `junior`: junior role, early-career role, assistant role where junior status is explicit.
- `mid_level`: ordinary experienced role with no junior, senior, lead, manager, or director signal.
- `senior`: senior specialist or senior professional role.
- `lead_or_principal`: lead, principal, staff, head-level specialist where not clearly executive.
- `director_or_executive`: director, VP, chief, C-suite, executive.
- `unclear`: seniority cannot be determined.

`management_level` — choose exactly one:
- `no_management`: no people, project, product, team, department, or executive management responsibility is stated.
- `team_lead_or_supervisor`: supervises or leads a small team or shift.
- `project_or_product_manager`: manages projects, programmes, delivery, or products, but not clearly line-managing staff.
- `line_manager`: manages people, performance, hiring, or direct reports.
- `department_or_function_head`: head of a department, function, practice, school, service, or business unit.
- `director_or_executive`: director, VP, chief, partner, C-suite, or executive management.
- `unclear`: management responsibility cannot be determined.

Important: seniority and management are different. A senior engineer is not a manager unless people, project, product, department, or executive management is explicitly stated.

`seniority_management_confidence` — a number between 0 and 1 expressing your overall confidence in this annotation.

`seniority_management_evidence` — a short exact quote from the title or description that best supports your annotation, or null if neither signal is present.
