## KEY IDEA

Prompt Optimizer API aims to reduce prompt tokens by applying various logics and algorithms systematically without impacting its quality and meanings. Intelligently compresses prompts while **preserving meaning, context, and intent**. Think of it as a smart minifier for LLM prompts, removing the fluff without losing the soul.

## APP GOAL - MUST ALWAYS BE RESPECTED:

Must preserve Prompt meaning, context, and intent; thus no impact on LLM response generation!

## DEV RULES:

1. API Response time is very cruicial and every change must consider impact on the response time.
2. Keep code simple and modular, Code files should not be more than 1000 lines.
3. ABSOLUTELY NO over engineering.
4. No backward competibility or fall backcode. Do a 100% code migration and clean all dead, onbsolete and unused code.
5. Always looks for redundant or dead code and propose a cleanup.
6. ALWAYS prefer editing existing code over writing from scratch.
7. Don’t invent new patterns if there’s already a solution in the codebase.
8. Never duplicate code or components. Reuse existing code wherever possible.
9. Zero tolerance for legacy and dead code; do a 100% cleanup.
10. Python code is auto-formatted with **Black** and **isort**; **flake8** enforces a max line length of 120.
11. Use **type hints** throughout Python code and keep imports sorted.
