# Streamlit Community Cloud deployment

1. Sign in to Streamlit Community Cloud with the GitHub account that can access
   this repository. Create an app from the existing GitHub repository.
2. Select branch `main`, entry point `app_multiagent.py`, and Python **3.11**.
3. Request an available subdomain such as `macos-recipes`. The resulting address
   is `https://macos-recipes.streamlit.app` if that name is available. This does
   not reserve that name or configure a separately purchased domain.
4. Deploy. Dependencies are installed from `requirements.txt`. No original
   dataset, source PDFs, API key, or HF token belongs in the repository or the
   app's shared Secrets settings for this user-supplied-key deployment.
5. Open the app. Enter your own key and select **Apply credentials**. Select a
   mode, fill in targets, then select **Recommend**.

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
Locally supplied authorized specialist PDFs in `kb_pdfs/` can be loaded explicitly
using the review KB button; their indexes belong only to that user's session.

The HF dataset must match `data_collection/README.md`. Tokens do not create access
rights: grant access on HF first. A dataset in a different schema or embedding
dimension is not a substitute for the original retrieval corpus. Streaming avoids
persisting dataset records through the app, but the selected corpus and FAISS index
still consume server memory per active user. Large datasets/concurrent sessions
may exceed Community Cloud resources; use smaller authorized corpora or a suitably
provisioned deployment. HF/library metadata caching is separate from corpus storage.

## Credentials and privacy

Password fields mask keys visually. Keys are transmitted over the deployed HTTPS
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
Model calls, PDF extraction and sequential reviews incur charges to the user's
OpenAI account. Use account/project limits appropriate to your intended usage.
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
