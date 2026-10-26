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
