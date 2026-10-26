"""Cloud helpers without global credentials, user caches, or persistent indexes."""
import hashlib
import hmac
import secrets

_FINGERPRINT_SALT = secrets.token_bytes(32)


def credential_fingerprint(*values):
    """An internal cache identity; never expose credentials or hashes in the UI."""
    return hmac.new(_FINGERPRINT_SALT, "\0".join(values).encode(), hashlib.sha256).hexdigest()


def safe_error(error):
    """Do not display provider exception messages that can contain request secrets."""
    name = type(error).__name__
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return "Authentication or access denied. Check your credentials and permissions."
    if name == "RateLimitError":
        return "API quota or rate limit reached. Check your account and try again later."
    return "The request failed. Check credentials, model access, input format, and connectivity."


def _provider_call(method, **kwargs):
    import litellm
    try:
        return getattr(litellm, method)(**kwargs)
    except Exception as error:
        raise RuntimeError(safe_error(error)) from None


def safe_completion(**kwargs):
    return _provider_call("completion", **kwargs)


def safe_embedding(**kwargs):
    return _provider_call("embedding", **kwargs)


def safe_batch_completion(**kwargs):
    return _provider_call("batch_completion", **kwargs)


def load_hf_retrieval(dataset_id, token, split="all"):
    """Stream authorized records into this user's memory; do not persist the corpus."""
    if not token:
        raise ValueError("An explicit Hugging Face token is required for hosted RAG.")
    from datasets import Dataset, load_dataset
    # No huggingface_hub.login(): it would change process-wide/persistent auth.
    source = load_dataset(dataset_id, token=token, streaming=True)
    splits = list(source) if split == "all" else [split]
    rows = []
    for split_name in splits:
        if split_name not in source:
            raise ValueError("Requested split is not available.")
        for row in source[split_name]:
            if row.get("contributions_embedding") is None:
                continue
            contribution = row.get("contribution_clean") or row.get("contribution")
            if not isinstance(contribution, str) or not isinstance(row.get("recipe"), str):
                raise ValueError("Dataset requires text contribution and recipe columns.")
            rows.append({
                "id": str(row.get("id") or f"{split_name}:{len(rows)}"),
                "contribution": contribution,
                "recipe": row["recipe"],
                "contributions_embedding": row["contributions_embedding"],
                "pdf_file": str(row.get("pdf_file") or ""),
            })
    if not rows:
        raise ValueError("Dataset has no usable records.")
    return Dataset.from_list(rows)


def validate_retrieval_dataset(dataset):
    import numpy as np
    if not len(dataset):
        raise ValueError("Retrieval dataset is empty.")
    dimension = None
    for row in dataset:
        vector = np.asarray(row["contributions_embedding"], dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("All embeddings must be finite, nonempty vectors.")
        dimension = dimension or vector.size
        if vector.size != dimension:
            raise ValueError("All embeddings must have the same dimension.")
