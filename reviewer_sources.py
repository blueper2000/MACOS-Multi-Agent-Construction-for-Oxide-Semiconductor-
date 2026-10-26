"""Literature sources for the paper's two sequential reviewers."""
from pathlib import Path

REVIEWERS = {
    "material_properties": "Material & Precursor Reviewer",
    "process_conditions": "Process Conditions Reviewer",
}


def load_reviewer_sources(uploaded, api_key, local_root=None):
    """Build isolated KBs; never invent extra agents or reassign legacy folders."""
    from multiagent_refinement import build_kb_from_pdfs
    knowledge_bases, source_names, failures = {}, {}, []
    for role in REVIEWERS:
        files = uploaded.get(role) or []
        names = [file.name for file in files]
        contents = [file.getvalue() for file in files]
        if not files and local_root:
            paths = sorted((Path(local_root) / role).rglob("*.pdf"))
            names = [path.name for path in paths]
            contents = [path.read_bytes() for path in paths]
        if not contents:
            continue
        kb = build_kb_from_pdfs(contents, api_key=api_key, source_names=names)
        if kb is None or not len(kb):
            failures.append(role)
            continue
        knowledge_bases[role] = kb
        source_names[role] = names
    return knowledge_bases, source_names, failures


def load_hosted_reviewer_sources(repo_id, hf_token, api_key):
    """Read the operator's private PDF corpus with explicit per-call auth.

    The hosting repository is configured server-side. PDF downloads use a
    temporary directory, not a public repository or a shared Hugging Face cache.
    """
    if not hf_token:
        raise ValueError("A Hugging Face token is required for the reviewer library.")
    from huggingface_hub import HfApi, hf_hub_download
    from multiagent_refinement import build_kb_from_pdfs
    import tempfile
    from contextlib import ExitStack
    import zipfile

    files = HfApi(token=hf_token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    knowledge_bases, source_names, failures = {}, {}, []
    with tempfile.TemporaryDirectory(prefix="macos-reviewers-") as cache, ExitStack() as stack:
        archive = None
        if "reviewer_library.zip" in files:
            package = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                filename="reviewer_library.zip", token=hf_token, cache_dir=cache)
            archive = stack.enter_context(zipfile.ZipFile(package))
            files = archive.namelist()
        for role in REVIEWERS:
            selected = sorted(name for name in files
                              if name.startswith(role + "/") and name.lower().endswith(".pdf"))
            if not selected:
                failures.append(role)
                continue
            contents, names = [], []
            for name in selected:
                if archive is not None:
                    # Read entries directly: never extract user-controlled ZIP paths.
                    contents.append(archive.read(name))
                else:
                    downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                        filename=name, token=hf_token, cache_dir=cache)
                    contents.append(Path(downloaded).read_bytes())
                names.append(Path(name).name)
            kb = build_kb_from_pdfs(contents, api_key=api_key, source_names=names)
            if kb is None or not len(kb):
                failures.append(role)
                continue
            knowledge_bases[role] = kb
            source_names[role] = names
    return knowledge_bases, source_names, failures
