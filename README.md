# MACOS: Multi-Agent Construction of Oxide-Semiconductor Recipes

Streamlit application for generating and refining oxide-semiconductor synthesis recipes with language models and retrieval.

## Setup

Use Python 3.10 or newer. Install dependencies and create a local environment file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your own `OPENAI_API_KEY` to `.env`, then run:

```bash
streamlit run app_multiagent.py
```

The key is read from the environment and is not included in this repository. Do not commit `.env` or paste API keys into source files. `HF_TOKEN` is optional and is only needed when loading a gated Hugging Face dataset.

## Retrieval assets

The app expects these locally supplied files under `data_collection/`:

- `s_d_plus_extracted_markdown_with_contribution.parquet`
- `_runtime_sd_plus_rag_prompt.txt`
- `_runtime_sd_plus_prediction_prompt.txt`

The runtime prompt files are included. The retrieval Parquet dataset is not included; provide it separately before launching the app. The app creates a FAISS index from that dataset when needed. Specialist PDF knowledge bases can be added under `kb_pdfs/`. Those research papers are also excluded from this source release.

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

The current app requires a local retrieval dataset even when choosing a
non-retrieval generation setting. Without it, startup stops with a missing-file
message. Source inspection is possible without the dataset; executing the full
workflow requires separately and lawfully prepared assets. A replacement dataset
can demonstrate the workflow but does not reproduce the paper's original benchmark.

See [data_collection/README.md](data_collection/README.md) for the input schema.
The two included prompt templates are the shared prompts for the app and predictor
classes. The legacy batch entry point in `experiment/predict.py` also expects a
separate local evaluation dataset at `data/omg`; that dataset is not included, and
the batch entry point has not been validated as a standalone reproduction pipeline.
