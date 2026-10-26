# Comparison with the supplied MACOS baseline

Baseline: the user's `SD multiagent os recipe gen` directory, checked 26 October
2026. No `.env`, private corpus, logs, source PDFs or generated indexes were copied.

Before this cloud change, the public release's two runtime prompt files were
byte-for-byte equivalent after UTF-8 decoding to the baseline. All named function
bodies in `app_multiagent.py` and `multiagent_refinement.py` were AST-equivalent;
`pdf2recipe.py` and `experiment/openai_utils.py` were identical. Predictor changes
were shared prompt path defaults and the batch prompt path resolver.

## Preserved research workflow

- Target material/properties and constraints drive initial recipe generation.
- The two runtime initial-generation prompt texts are unchanged.
- Material reviewer precedes process reviewer; process review receives material
  feedback. Supervisor integrates both reviews into the recipe.
- Researcher feedback can revise the contribution and regenerate/refine a recipe.
- Structured JSON tables, rationales, reference IDs, history and exports remain.
- Reviewer and Supervisor prompt text is unchanged; API calls now receive the
  current user's key explicitly through error-sanitizing wrappers.

## Intentional deployment changes

- Explicit per-request credentials replace the app's global credential settings;
  operators can configure `st.secrets`, or visitors can provide their own keys.
- LLM-only and uploaded-PDF modes require no retrieval corpus. PDF context uses
  the same RAG-format prompt without initializing a retrieval dataset.
- RAG uses the selected authorized HF dataset instead of a mandatory local file.
  Access is checked by the caller's explicit token on first load in that session.
- Dataset records, clients and FAISS indexes are isolated in session memory;
  process-wide HF login and persisted FAISS files are removed from this path.
- Specialist KB loading is explicit and session-local; missing KBs are disclosed.
- The original prediction iterator's early `break` dropped later batches; removed.
- PDF plus retrieved references now get distinct sequential IDs; an old recipe
  is revision context, not a newly fabricated literature reference.
- Recommend starts a fresh generation, rather than leaving the prior conversation
  as the active result. Credential changes and disconnect clear previous state.

This is deployment compatibility and structural comparison, not numerical
reproduction of the original study. Tests use synthetic records and mocked API
responses. Original-data retrieval rankings, real provider access and physical
experiment outcomes have not been validated by those tests.
