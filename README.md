# MACOS: Multi-Agent Construction of Oxide-Semiconductor Recipes

Streamlit application for generating and refining oxide-semiconductor synthesis recipes with language models and retrieval.

## Run locally

Use Python 3.11. Install dependencies and run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_multiagent.py
```

Enter your own OpenAI key in the app sidebar and select **Apply credentials**.
Select **LLM-only**, **PDF**, or **RAG**. LLM-only and PDF require no retrieval
corpus or Hugging Face token. RAG additionally requires your read-only HF token,
an authorized dataset ID and a compatible dataset schema. Keys belong to your
session; changing credentials clears previous results and loaded data.

For deployment, privacy details, mode differences and memory limits see
[DEPLOYMENT.md](DEPLOYMENT.md). Select `app_multiagent.py` on branch `main` in
Streamlit Community Cloud and Python 3.11. No shared API secrets are required.

## Retrieval assets

The initial-generation runtime prompt templates are included in `data_collection/`.
The original literature-derived Parquet corpus, FAISS indexes and specialist
PDFs are excluded. See [data schema](data_collection/README.md) and
[baseline comparison](BASELINE_COMPARISON.md). No dataset is downloaded for
LLM-only or PDF mode. The cloud RAG path builds an index in the current session.

## Notes

API usage may incur charges. Model availability and pricing depend on your provider account.

## License and permitted use

Source is available for inspection under the custom [MACOS Research and
Verification License](LICENSE). Individuals may download, run, and locally
modify the code and prompt templates for their own non-commercial research,
paper reproduction, and verification. Redistribution of the original or modified
software, including patches and packaged copies, requires prior written
permission, subject to applicable GitHub platform rights and statutory exceptions.
Commercial use and hosted services are not permitted. This is not an open-source
license. Third-party materials retain their own licenses.

Research findings, metrics, and generated recipe outputs may be published subject
to the license and third-party rights. Please cite the associated paper and
repository, and report the version used and any local changes.
For other permissions, contact blueper2000@skku.edu.

## Data availability and reproduction limits

The literature-derived retrieval Parquet dataset, its FAISS index, and specialist
PDF knowledge bases are not distributed. Redistribution rights for the underlying
source materials have not been established. Public access to a paper does not by
itself establish permission to redistribute it or a dataset derived from it.
No access to the original dataset is promised by this repository.

LLM-only and PDF modes work without the original retrieval dataset. RAG requires
a separately authorized HF corpus matching the documented schema. A replacement
corpus can demonstrate the workflow but cannot reproduce the paper's original
benchmark. Specialist knowledge bases are not bundled; reviews without them use
model knowledge and available recipe references.

See [data_collection/README.md](data_collection/README.md) for the input schema.
The runtime prompts are shared by the app and predictor classes. The legacy batch
entry point also expects a separate evaluation dataset at `data/omg`; it is not
the cloud app entry point or a validated standalone reproduction pipeline.
