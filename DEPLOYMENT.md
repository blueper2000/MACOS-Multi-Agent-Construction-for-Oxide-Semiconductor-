# Streamlit Community Cloud deployment

1. Sign in to Streamlit Community Cloud with the GitHub account that can access
   this repository. Create an app from the existing GitHub repository.
2. Select branch `main`, entry point `app_multiagent.py`, and Python **3.11**.
3. Request an available subdomain such as `macos-recipes`. The resulting address
   is `https://macos-recipes.streamlit.app` if that name is available. This does
   not reserve that name or configure a separately purchased domain.
4. For operator-funded access, open **Advanced settings > Secrets** and enter the
   TOML described below. Dependencies are installed from `requirements.txt`.
   Original datasets, source PDFs and actual secrets must stay out of GitHub.
5. Deploy. With an operator key configured, the app uses it by default. Select a
   mode, fill in targets, then select **Recommend**. Without an operator key,
   users enter their own credentials and select **Apply credentials**.

## Operator-managed Secrets

Paste this template into the private Streamlit Secrets editor and fill the two
empty credential strings there. Do not put real values in `config.toml`, commit
them to GitHub, or post them in chat.

```toml
OPENAI_API_KEY = ""
HF_TOKEN = ""
HF_DATASET_ID = "blueper/oxidesc-recipe-ver2"
HF_DATASET_SPLIT = "all"
REVIEWER_KB_REPO = ""
```

The key and token are read server-side using `st.secrets`. When `OPENAI_API_KEY`
is set, **Use app-managed credentials** is selected by default. Visitors do not
need to enter keys. HF credentials are needed only for RAG. The operator chooses
the RAG dataset/split in Secrets; visitors cannot select arbitrary repositories
using the operator's token. Leave the HF token empty if only LLM-only/PDF is needed.
For local development, the same values may be stored in the Git-ignored file
`.streamlit/secrets.toml`; `.streamlit/secrets.toml.example` is an empty template.

The operator pays for all API requests using this connection. Anyone who can use
the deployed app can request outputs based on its configured HF corpus, including
retrieved recipe text exposed by the existing references and export features.
A private HF repository hides the raw repository from unauthenticated HF users;
it does not by itself make this app's outputs private. Configure app access in
Streamlit for the intended audience and use a token restricted to this dataset.
This app does not implement its own visitor login or per-user spending quotas.

Visitors can turn off the app-managed connection and supply their own credentials.
The app never pre-fills input fields with operator secrets or combines a personal
OpenAI key with the operator's HF token. Switching connections clears old results
and private records from that session. Disconnect pauses the managed connection
until the visitor explicitly reconnects; it does not remove platform Secrets.

## Modes

| Mode | Required assets | Behavior |
| --- | --- | --- |
| LLM-only | Your OpenAI key | Generates a recipe without a literature dataset or retrieval embeddings. |
| PDF | Your OpenAI key and uploaded readable PDF(s) | Extracts recipe context from the PDFs, then generates with that context; no HF data required. |
| RAG | Your OpenAI key, read-only HF token, authorized dataset ID and split | Streams records from HF, builds a session-local in-memory FAISS index, and retrieves recipes. PDFs can additionally be supplied. |

Sequential material/process review and Supervisor integration remain active in
all modes. Without specialist PDF knowledge bases, reviewers use model knowledge
and the available recipe references; this is not the full literature-supported
configuration of the paper. The omitted specialist PDFs are not fetched implicitly.
The **Reviewer literature** sidebar has exactly two collections:
**Material & Precursor Reviewer** and **Process Conditions Reviewer**.
Upload readable PDFs to the corresponding collection and select **Load reviewer
papers**. Text extraction and embedding use the connected API account; indexes
and uploaded papers belong only to the current session. For local deployments,
authorized PDFs may alternatively be placed in `kb_pdfs/material_properties/` and
`kb_pdfs/process_conditions/` and loaded with the same button. The old
`gate_dielectric` and `sd_relation` folders are not separate reviewers and are
not loaded by this workflow. Specialist PDFs are not downloaded from the recipe
Parquet dataset or silently published in GitHub.

Loading papers affects subsequent reviews only. Generate a new recipe or request
a revision after loading; an already completed result is not automatically rerun.
If no papers are loaded, the sidebar states this explicitly. **Clear reviewer
papers** removes the session's uploaded reviewer selections and indexes.

To provide default papers to every visitor without publishing PDFs in GitHub,
store authorized PDFs in a private Hugging Face dataset with folders
`material_properties/` and `process_conditions/`. Give the connected HF token
read access and set `REVIEWER_KB_REPO` in Streamlit Secrets to that dataset ID.
The app then downloads and indexes the two collections automatically per session;
visitors do not need to upload the standard papers. Downloads use a temporary
directory that is removed after indexing. A loaded KB remains in the session's
memory and may supply excerpts to reviews and Q&A. Private storage does not
replace permissions needed to process or expose source text to the app audience.
The recipe retrieval dataset configured by `HF_DATASET_ID` is separate.
The same two folders may be packaged in a single `reviewer_library.zip` at the
dataset root. The app downloads that archive once, reads PDF entries directly,
and never extracts ZIP paths into the server filesystem.

The HF dataset must match `data_collection/README.md`. Tokens do not create access
rights: grant access on HF first. A dataset in a different schema or embedding
dimension is not a substitute for the original retrieval corpus. Streaming avoids
persisting dataset records through the app, but the selected corpus and FAISS index
still consume server memory per active user. Large datasets/concurrent sessions
may exceed Community Cloud resources; use smaller authorized corpora or a suitably
provisioned deployment. HF/library metadata caching is separate from corpus storage.

## Credentials and privacy

For personal credentials, password fields mask keys visually. Keys are transmitted over the deployed HTTPS
connection to the app server to perform requests and remain in session memory;
they are not end-to-end hidden from the server operator. This app does not write
them to files, Git, logs, global credential variables, or shared Streamlit caches.
Do not enable HTTP/provider debug logging in production. Infrastructure operators
and external API providers have their own policies.

Apply credentials again after changing a key or token: previous results, clients,
and loaded private records are cleared from that session. **Disconnect and clear
session** removes the app's references to credentials and conversation state;
it is not a guarantee of immediate physical memory erasure. API calls already in
flight can finish. Do not use a shared browser session for different users.

PDF text, targets, retrieved snippets and feedback needed for generation are sent
to the selected API provider. Only upload material you are authorized to process.
Model calls, PDF extraction and sequential reviews incur charges to the connected
OpenAI account (the operator's account for app-managed access). Use account/project
limits appropriate to your intended usage.
The initial model can be selected in the app; PDF extraction and sequential
specialist/Supervisor calls currently use `gpt-4o`, so the account needs access
to that model as well. Scanned/image-only PDFs require OCR beforehand.

## Local checks

Install `requirements.txt`, then run `python -m unittest discover -s tests -v`.
Checks use synthetic data and mocked API responses: no private dataset or real
API credentials are required. Launch locally with `streamlit run app_multiagent.py`.

Custom source-available licensing is described in `LICENSE`; this hosted deployment
is prepared for the copyright holder. The license does not grant third parties
permission to run a public hosted service.
