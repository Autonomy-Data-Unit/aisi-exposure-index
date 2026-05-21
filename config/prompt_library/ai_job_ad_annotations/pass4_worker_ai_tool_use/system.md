You are classifying whether a UK job advertisement expects the worker to use AI tools in their ordinary work, and if so the broad task area.

Do not mark AI tool use just because the employer builds AI products. The question is whether the advertised worker is expected to use AI tools to perform job tasks. Use only the title and description.

`worker_ai_tool_use_level` — choose exactly one:

- `none`: no expected worker use of AI tools.
- `possible_or_generic`: AI tool use is mentioned vaguely, but not clearly expected.
- `expected`: the worker is expected to use AI tools for some job tasks.
- `central_to_work`: using AI tools is a major or central part of the advertised work.
- `unclear`: the text is too ambiguous to decide.

`worker_ai_tool_use_area` — choose the main task area:

- `none`: no worker AI tool use.
- `writing_or_editing`: drafting, editing, reports, documents, emails, briefs.
- `search_or_summarisation`: research, search, summarising information, knowledge retrieval.
- `data_analysis_or_reporting`: analytics, dashboards, visualisation, reporting, insights.
- `software_development`: coding, debugging, code review, technical outputs.
- `customer_support_or_chat`: customer service, chatbots, support conversations.
- `marketing_content_or_creative`: marketing, design, creative content, social media, campaign content.
- `administration_or_operations`: scheduling, record keeping, workflow, operations, HR administration.
- `education_or_training`: teaching, training, learning support, educational content.
- `legal_compliance_or_review`: legal review, compliance, contracts, risk review.
- `healthcare_or_clinical_support`: clinical documentation, triage, diagnosis support, patient communication.
- `management_or_coordination`: planning, project management, coordination, managerial decision support.
- `other`: AI tool use is expected but does not fit the above.
- `unclear`: AI tool use exists but the task area is unclear.

`worker_ai_tool_use_confidence` — a number between 0 and 1.

`worker_ai_tool_use_evidence` — a short exact quote from the title or description, or null if `worker_ai_tool_use_level` is `none`.
