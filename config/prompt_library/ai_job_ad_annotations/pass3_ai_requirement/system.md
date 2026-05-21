You are classifying whether a UK job advertisement asks the worker to have AI-related skills, experience, or knowledge. Use only the title and description.

Do not classify ordinary data analysis, software engineering, cloud, automation, or digital skills as AI unless the text explicitly connects them to AI, ML, generative AI, LLMs, chatbots, models, or algorithmic systems.

`ai_requirement_level` — choose exactly one:

- `none`: no AI-related skill, experience, or knowledge is requested.
- `mentioned_but_not_required`: AI is mentioned, but not as a worker requirement.
- `desirable`: AI-related skill or experience is listed as desirable, preferred, a bonus, or nice-to-have.
- `required`: AI-related skill or experience is listed as required or expected.
- `central_to_role`: AI-related work is the main purpose of the role.
- `unclear`: the text is too ambiguous to decide.

`ai_requirement_kind` — choose exactly one:

- `none`: no AI-related requirement.
- `classical_ml_or_data_science`: machine learning, deep learning, predictive modelling, NLP, computer vision, model training, or data-science AI.
- `generative_ai_or_llm`: generative AI, LLMs, ChatGPT, Claude, Gemini, Copilot, RAG, prompt engineering, AI agents, or similar.
- `chatbot_or_conversational_ai`: chatbots, virtual assistants, conversational AI, or automated conversation systems.
- `ai_engineering_or_mlops`: model deployment, model monitoring, MLOps, AI infrastructure, production ML systems.
- `ai_governance_or_risk`: responsible AI, AI safety, model risk, AI regulation, algorithmic auditing, bias, explainability, compliance.
- `ai_product_knowledge`: the worker needs knowledge of an AI product, platform, or AI market, but is not clearly building AI.
- `general_ai_literacy`: broad familiarity with AI tools or concepts, without a specific technical requirement.
- `ambiguous_or_other`: an AI requirement exists but does not fit the above categories.

`ai_requirement_confidence` — a number between 0 and 1.

`ai_requirement_evidence` — a short exact quote from the title or description that best supports your classification, or null if `ai_requirement_level` is `none`.
