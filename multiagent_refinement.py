import numpy as np
import litellm


# ─────────────────────────────────────────────────────────────────────────────
# Specialist Knowledge Base (PDF → chunk → embed → cosine retrieval)
# ─────────────────────────────────────────────────────────────────────────────

class SpecialistKnowledgeBase:
    """
    PDF 문서들을 청크로 분할·임베딩하여 코사인 유사도 검색을 제공하는 경량 벡터 스토어.

    build_kb_from_pdfs() 또는 build_kb_from_directory() 로 생성합니다.
    """

    def __init__(self, chunks: list[str], embeddings: np.ndarray, source_labels: list[str] | None = None):
        self.chunks = chunks
        self.embeddings = embeddings                  # shape (N, D), L2-normalised
        self.source_labels = source_labels or [""] * len(chunks)

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        """query_embedding 과 가장 유사한 top_k 청크를 반환합니다."""
        if len(self.chunks) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return self.chunks[:top_k]
        q = q / q_norm
        sims = self.embeddings @ q          # (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]
        out = []
        for i in top_idx:
            label = self.source_labels[i] if i < len(self.source_labels) else ""
            prefix = f"[Source: {label}]\n" if label else ""
            out.append(prefix + self.chunks[i])
        return out

    def __len__(self):
        return len(self.chunks)


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PyMuPDF4LLM 으로 PDF 바이트 → Markdown 텍스트 추출."""
    import pymupdf4llm
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)
    return md_text


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """텍스트를 겹침이 있는 고정 크기 문자 청크로 분할합니다."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:          # 너무 짧은 청크 제거
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_kb_from_pdfs(
    pdf_bytes_list: list[bytes],
    api_key: str,
    source_names: list[str] | None = None,
    chunk_size: int = 1200,
    overlap: int = 200,
    embedding_model: str = "text-embedding-3-large",
    batch_size: int = 20,
) -> "SpecialistKnowledgeBase | None":
    """
    PDF 바이트 목록에서 SpecialistKnowledgeBase 를 생성합니다.

    Parameters
    ----------
    pdf_bytes_list : list of bytes
        각 PDF 파일의 바이트 내용
    api_key : str
        OpenAI API 키
    source_names : list[str] | None
        각 PDF 파일명 (로그/레이블용)
    chunk_size : int
        청크 문자 수 (기본 1200)
    overlap : int
        인접 청크 겹침 문자 수 (기본 200)
    embedding_model : str
        임베딩 모델 (기본 text-embedding-3-large)
    batch_size : int
        임베딩 API 배치 크기 (기본 20)

    Returns
    -------
    SpecialistKnowledgeBase | None
        PDF 가 없거나 텍스트 추출 실패 시 None
    """
    all_chunks: list[str] = []
    all_labels: list[str] = []

    for i, pdf_bytes in enumerate(pdf_bytes_list):
        label = (source_names[i] if source_names and i < len(source_names) else f"doc_{i}")
        try:
            text = _extract_text_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            print(f"[KB] PDF 텍스트 추출 실패 ({label}): {e}")
            continue
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        all_labels.extend([label] * len(chunks))

    if not all_chunks:
        return None

    # 임베딩 배치 호출
    all_embeddings: list[list[float]] = []
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start: batch_start + batch_size]
        resp = litellm.embedding(model=embedding_model, input=batch, api_key=api_key)
        all_embeddings.extend([item["embedding"] for item in resp["data"]])

    emb_array = np.array(all_embeddings, dtype=np.float32)
    # L2 정규화
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array = emb_array / np.maximum(norms, 1e-9)

    return SpecialistKnowledgeBase(all_chunks, emb_array, all_labels)


def build_kb_from_directory(
    directory: str,
    api_key: str,
    **kwargs,
) -> "SpecialistKnowledgeBase | None":
    """
    디렉토리 내 모든 PDF 파일에서 SpecialistKnowledgeBase 를 생성합니다.

    Parameters
    ----------
    directory : str
        PDF 파일들이 위치한 디렉토리 경로
    api_key : str
        OpenAI API 키
    **kwargs
        build_kb_from_pdfs 에 전달할 추가 옵션
    """
    import os, glob
    pdf_paths = sorted(glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        return None

    pdf_bytes_list = []
    source_names = []
    for path in pdf_paths:
        try:
            with open(path, "rb") as f:
                pdf_bytes_list.append(f.read())
            source_names.append(os.path.basename(path))
        except Exception as e:
            print(f"[KB] 파일 읽기 실패 ({path}): {e}")

    return build_kb_from_pdfs(pdf_bytes_list, api_key, source_names=source_names, **kwargs)


# (build_refinement_workflow, _run_pipeline_async, run_refinement_pipeline_sync 제거됨)
# 현재 앱은 run_refinement_sequential 만 사용합니다.

# ─────────────────────────────────────────────────────────────────────────────
# 순차 실행 버전 (Streamlit 실시간 UI 콜백 지원) — 메인 진입점
# ─────────────────────────────────────────────────────────────────────────────

def run_refinement_sequential(
    recipe_json: str,
    api_key: str,
    specialist_model: str = "gpt-4o",
    step_callback=None,
    knowledge_bases: dict | None = None,
    kb_top_k: int = 3,
    reference_context: str = "",
) -> dict[str, str]:
    """
    3개의 Specialist Agent 를 순차 실행합니다.
    Streamlit 실시간 UI 업데이트(st.status 콜백)에 최적화되어 있습니다.

    Parameters
    ----------
    recipe_json : str
        개량할 초기 레시피 JSON 문자열
    api_key : str
        OpenAI API 키
    specialist_model : str
        LLM 모델명 (기본 "gpt-4o")
    step_callback : callable | None
        step_callback(step_num: int, label: str, feedback: str | None)
        feedback=None → 스텝 시작 시 호출
        feedback=str  → 스텝 완료 시 호출
    knowledge_bases : dict | None
        각 specialist 전용 SpecialistKnowledgeBase 딕셔너리.
        키: "material_properties", "gate_dielectric", "process_conditions"
        값: SpecialistKnowledgeBase 인스턴스 (없는 키는 KB 없이 실행)
    kb_top_k : int
        KB 에서 검색할 청크 수 (기본 3)
    reference_context : str
        Ref1, Ref2... 라벨과 실제 reference 근거를 연결한 짧은 텍스트

    Returns
    -------
    dict with keys:
        material_properties : 조성비·결정성·프리커서 통합 검토 피드백
        gate_dielectric      : 게이트 절연막 계면 검토 피드백
        process_conditions   : 채널 증착 온도/시간 + PDA 검토 피드백
        refined_recipe       : 개량된 레시피 JSON 문자열
    """

    knowledge_bases = knowledge_bases or {}

    def _cb(step_num: int, label: str, feedback=None):
        if step_callback:
            step_callback(step_num, label, feedback)

    def _llm(prompt: str, max_tokens: int = 1500) -> str:
        resp = litellm.completion(
            model=specialist_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            api_key=api_key,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def _embed(text: str) -> list[float]:
        resp = litellm.embedding(model="text-embedding-3-large", input=[text], api_key=api_key)
        return resp["data"][0]["embedding"]

    def _kb_context(kb_key: str, query: str) -> str:
        """KB 가 있으면 관련 청크 top_k 를 반환, 없으면 빈 문자열."""
        kb: SpecialistKnowledgeBase | None = knowledge_bases.get(kb_key)
        if kb is None or len(kb) == 0:
            return ""
        q_emb = _embed(query)
        chunks = kb.query(q_emb, top_k=kb_top_k)
        if not chunks:
            return ""
        joined = "\n\n---\n\n".join(chunks)
        return (
            "\n\n════════ RELEVANT LITERATURE (from uploaded PDFs) ════════\n"
            f"{joined}\n"
            "══════════════════════════════════════════════════════════\n"
        )

    result: dict[str, str] = {
        "material_properties": "",
        "process_conditions": "",
        "refined_recipe": "",
    }

    # ── Step 1: 재료 특성 통합 검토 (조성비 + 결정성 + 프리커서) ─────────────
    _cb(1, "재료 특성 (조성비·결정성·프리커서) 검토 중…")
    ctx1 = _kb_context(
        "material_properties",
        "oxide semiconductor composition cation ratio crystallinity amorphous precursor ALD "
        "CVD sputtering dopant carrier density grain boundary phase",
    )
    mat = _llm(f"""You are a specialist in oxide semiconductor material engineering,
covering composition, crystallinity, and precursor chemistry simultaneously.

Review the following oxide semiconductor deposition recipe with a holistic perspective
on how composition, crystallinity, and precursor choices interact:

COMPOSITION:
1. Cation ratios (e.g., In:Ga:Zn for IGZO, In:Zn for IZO) — optimal range for
   the stated mobility and stability targets
2. Dopant type and concentration — effect on carrier density and Fermi level
3. Risk of phase separation or excess carrier density

CRYSTALLINITY:
4. Desired phase: amorphous vs. polycrystalline — consistency with stability goals
5. How deposition conditions (substrate temperature, O2/Ar ratio, RF power, pressure)
   influence phase formation
6. Grain boundary trap density and its impact on bias stress stability

PRECURSOR:
7. Metal precursor selection (TMIn, DEZ, TMA, InCp, etc.) — match to target composition
8. Oxidant selection (H2O, O3, O2 plasma) — oxidation efficiency and carbon removal
9. ALD window consistency with desired composition and crystallinity
10. Cross-check: are precursor choices aligned with the composition ratios and phase goals?
{ctx1}
Recipe JSON:
{recipe_json}

Return a unified, actionable feedback covering all three aspects and their interactions.
If literature excerpts are provided above, cite specific findings where relevant.""")
    result["material_properties"] = mat
    _cb(1, "재료 특성 검토 완료", mat)

    # ── Step 2: 채널 증착 조건 + PDA 검토 ────────────────────────────────────
    _cb(2, "채널 증착 조건 & Post-Deposition Annealing 검토 중…")
    ctx2 = _kb_context(
        "process_conditions",
        "channel deposition temperature time substrate heating rate ALD cycle "
        "post deposition annealing PDA temperature time atmosphere oxygen nitrogen air forming gas",
    )
    proc = _llm(f"""You are a specialist in oxide semiconductor thin film process engineering,
focused on deposition conditions and post-deposition annealing (PDA).

Review the following oxide semiconductor deposition recipe, focusing on:

CHANNEL DEPOSITION CONDITIONS:
1. Substrate temperature during deposition — optimal range for target phase and mobility
2. Deposition time / number of ALD cycles / deposition rate — effect on thickness uniformity
3. Ramp-up and stabilization time before deposition — thermal budget concerns
4. O2 partial pressure or O2/Ar ratio during sputtering / gas flow during ALD
5. Consistency between deposition temperature and desired crystallinity / phase

POST-DEPOSITION ANNEALING (PDA):
6. Annealing temperature — sufficient for defect passivation without inducing crystallization
7. Annealing time — trade-off between trap passivation and thermal budget
8. Annealing atmosphere (O2, N2, air, forming gas H2/N2) — effect on oxygen vacancy
   density, carrier concentration, and interface quality
9. Ramp rate (°C/min) — thermal shock risk to thin film and substrate
10. Whether PDA conditions are consistent with the material thermal budget

Material properties context: {mat}
{ctx2}
Recipe JSON:
{recipe_json}

Return concise, actionable feedback on deposition conditions and PDA.
If literature excerpts are provided above, cite specific findings where relevant.""")
    result["process_conditions"] = proc
    _cb(2, "채널 증착 조건 & PDA 검토 완료", proc)

    # ── Step 3: 피드백 통합 → 개량 레시피 JSON ──────────────────────────────
    _cb(3, "피드백 통합 중… (개량 레시피 생성)")
    refined = _llm(f"""You are a senior oxide semiconductor process engineer.
Produce an IMPROVED version of the recipe JSON that incorporates all specialist feedback.

CRITICAL RULES:
- Preserve the EXACT same JSON schema, keys, and structure as the input
- Only modify parameter values where the feedback suggests improvement
- For each modified parameter, update the "Rationale" field to explain the change
- Preserve all existing "Rationale" and "Ref IDs" fields.
- Every row that has a "Rationale" field must keep a brief factual rationale.
- Every row that has a "Ref IDs" field must contain one or more valid reference labels from the available references when evidence supports the row; otherwise use "N/A".
- Do not invent reference labels that are not listed in the available references.
- Keep all sections (tables, scorecard, meta) intact
- The recipe uses long table rows. Every row in every table must have exactly:
  "Variable", "Predicted Value", "Rationale", "Ref IDs".
- Do not convert long table rows into wide tables.
- The JSON must be complete and parseable. Do not stop early.
- Return ONLY a valid JSON object — no markdown fences, no preamble, no trailing text

Original Recipe JSON:
{recipe_json}

Available references for "Ref IDs":
{reference_context or "N/A"}

[1. MATERIAL PROPERTIES REVIEW — Composition · Crystallinity · Precursor]
{mat}

[2. CHANNEL DEPOSITION CONDITIONS & POST-DEPOSITION ANNEALING REVIEW]
{proc}

Return the complete refined recipe as a single JSON object.""", max_tokens=12000)
    result["refined_recipe"] = refined
    _cb(3, "개량 완료 ✅", refined)

    return result
