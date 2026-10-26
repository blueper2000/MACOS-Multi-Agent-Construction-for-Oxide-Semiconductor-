# Local retrieval assets

The repository includes only the two runtime prompt templates. Do not commit
literature-derived Parquet data, FAISS indexes, or source PDFs.

To run the app, prepare your own lawfully obtained data and save it as
`s_d_plus_extracted_markdown_with_contribution.parquet` in this directory.
The loader in `experiment/predict.py` expects one row per retrievable recipe:

| Column | Format | Requirement |
| --- | --- | --- |
| `contributions_embedding` | List of finite numbers | Required; use the same embedding model and vector dimension as app queries (`text-embedding-3-large` with its default dimensions). All rows must have the same dimension. |
| `contribution_clean` or `contribution` | Non-null text | At least one required; use complete text values. |
| `recipe` | Text | Required; recipe associated with that contribution. |
| `id` | Unique text | Recommended; otherwise derived from optional metadata. |
| `paper_id`, `sample_label`, `pdf_file` | Text | Optional provenance fields; use only metadata you may share. |
| `classification_result` | Text | Optional metadata. |

Rows with missing embeddings are dropped by the loader. The app builds a local
FAISS index beside the Parquet file. Remove the generated index after replacing
the dataset or changing its embedding model or vector dimension, so it is rebuilt.

A wholly fictional row might pair the contribution text "Synthetic test: oxide
channel deposited by ALD" with the recipe text "Fictional example; not an
experimental recommendation". Compute a real embedding of your own synthetic text
if using such a row for a technical smoke test; do not use dummy vectors to assess
retrieval quality. No source-paper text or real literature records are provided here.

Specialist PDFs may be placed in `kb_pdfs/` at the repository root when you have
the necessary rights. They are excluded from the source release and must be
obtained separately. A replacement corpus cannot substantiate reproduction of
the original paper's reported numerical results.
