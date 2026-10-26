# Local retrieval assets

The repository includes only the two runtime prompt templates. Do not commit
literature-derived Parquet data, FAISS indexes, or source PDFs.

The cloud app starts without this corpus in LLM-only/PDF modes. For RAG, provide
a dataset on Hugging Face that your account can access, using the schema below.
The library also supports a local Parquet path for authorized local use.
The loader in `experiment/predict.py` expects one row per retrievable recipe:

| Column | Format | Requirement |
| --- | --- | --- |
| `contributions_embedding` | List of finite numbers | Required; use the same embedding model and vector dimension as app queries (`text-embedding-3-large` with its default dimensions). All rows must have the same dimension. |
| `contribution_clean` or `contribution` | Non-null text | At least one required; use complete text values. |
| `recipe` | Text | Required; recipe associated with that contribution. |
| `id` | Unique text | Recommended; otherwise derived from optional metadata. |
| `paper_id`, `sample_label`, `pdf_file` | Text | Optional provenance fields; use only metadata you may share. |
| `classification_result` | Text | Optional metadata. |

Rows with missing embeddings are dropped; remaining vectors must be finite and
have one consistent dimension. The cloud RAG index is built in session memory and
is not saved to a shared file. The cloud loader does not automatically read a
Parquet file merely because one exists beside these prompt templates.

A wholly fictional row might pair the contribution text "Synthetic test: oxide
channel deposited by ALD" with the recipe text "Fictional example; not an
experimental recommendation". Compute a real embedding of your own synthetic text
if using such a row for a technical smoke test; do not use dummy vectors to assess
retrieval quality. No source-paper text or real literature records are provided here.

Specialist PDFs may be placed in `kb_pdfs/` at the repository root when you have
the necessary rights. They are excluded from the source release and must be
obtained separately. A replacement corpus cannot substantiate reproduction of
the original paper's reported numerical results.
