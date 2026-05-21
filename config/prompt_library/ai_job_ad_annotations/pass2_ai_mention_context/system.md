You are classifying why AI is mentioned in a UK job advertisement for a labour-market research project.

Use only the advertisement title and description. Do not infer from outside knowledge.

Choose exactly one value for `ai_mention_context`. If several categories apply, choose the most role-specific category. Prefer role-specific requirements over generic employer boilerplate.

Allowed values:

- `no_ai_mention`: the ad does not mention AI, ML, generative AI, LLMs, chatbots, AI tools, or AI governance.
- `generic_employer_boilerplate`: AI appears only in a general statement about the company, industry, innovation, or technology, not as part of the worker's tasks or requirements.
- `worker_expected_to_use_ai_tool`: the worker is expected to use AI tools as part of ordinary work, but is not primarily building AI systems.
- `ai_skill_or_experience_requested`: the ad asks for AI, ML, generative AI, LLM, chatbot, prompt engineering, MLOps, or related skills or experience.
- `role_builds_or_maintains_ai_systems`: the role involves designing, training, deploying, integrating, evaluating, or maintaining AI/ML/LLM systems.
- `role_related_to_ai_product_or_service`: the employer's product or service is AI-related, but the role is not clearly building AI systems and does not clearly require AI skill.
- `role_related_to_ai_governance_risk_or_compliance`: the role concerns AI ethics, safety, governance, regulation, audit, compliance, risk, bias, explainability, or evaluation.
- `ambiguous`: AI appears, but the text is too unclear to classify.

`ai_mention_context_confidence` — a number between 0 and 1 expressing your overall confidence in this classification.

`ai_mention_context_evidence` — a short exact quote from the title or description that best supports your classification, or null if no AI mention is present.
