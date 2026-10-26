from pathlib import Path


import json
import pandas as pd
from io import BytesIO
import streamlit as st
import os
from experiment.predict import RAGRecipePredictor, RecipePredictor, ReferenceRecipePredictor
from cloud_runtime import safe_error
from cloud_runtime import safe_embedding as embedding
import litellm
from pdf2recipe import pdf_bytelist_to_recipes, pdf_bytes_to_markdown
from cloud_runtime import safe_completion as completion
from datetime import datetime

# ── Multi-Agent Refinement 모듈 ──
from multiagent_refinement import (
    run_refinement_sequential,
    build_kb_from_directory,
)

APP_DIR = Path(__file__).resolve().parent
DATA_COLLECTION_DIR = APP_DIR / "data_collection"
LOCAL_RETRIEVAL_PARQUET = DATA_COLLECTION_DIR / "s_d_plus_extracted_markdown_with_contribution.parquet"
RUNTIME_RAG_PROMPT_PATH = DATA_COLLECTION_DIR / "_runtime_sd_plus_rag_prompt.txt"
RUNTIME_PREDICTION_PROMPT_PATH = DATA_COLLECTION_DIR / "_runtime_sd_plus_prediction_prompt.txt"

# ---------------- Streamlit setup ----------------
st.set_page_config(
    page_title="MACOS (Multi-Agent Construction of Oxide-Semiconductor Recipes)",
    page_icon=":microscope:",
    layout="wide",
)
st.title("MACOS (Multi-Agent Construction of Oxide-Semiconductor Recipes)")

st.markdown(
    """<p style='font-size:14px'>개발자: 홍지훈, 소속: 성균관대 조성범 교수 연구실, e-mail: blueper2000@skku.edu</p>""", unsafe_allow_html=True
)

st.markdown(
    """<p style='font-size:14px'>문의 사항이나 협업 같은 사항이 있다면 연락해주세요.</p>""", unsafe_allow_html=True
)

st.sidebar.title("Input Parameters")

def _clear_session():
    st.session_state.clear()


def _apply_credentials():
    api_key = st.session_state.get("openai_key_input", "").strip()
    token = st.session_state.get("hf_token_input", "").strip()
    # Changing credentials invalidates all previous outputs and loaded private data.
    st.session_state.clear()
    st.session_state.openai_key_input = api_key
    st.session_state.hf_token_input = token
    st.session_state.active_openai_key = api_key
    st.session_state.active_hf_token = token


def _start_generation():
    for key in ("messages", "references", "latest_recipe_json", "fixed_examples",
                "retrieval_examples", "original_contribution", "current_contribution",
                "current_recipe", "latest_meta", "refined_recipe", "ma_result",
                "initial_pdf_context", "initial_pdf_names", "initial_pdf_warnings"):
        st.session_state.pop(key, None)


with st.sidebar:
    with st.form("credentials"):
        st.subheader("API connection")
        st.text_input("OpenAI API key", type="password", key="openai_key_input")
        st.text_input("Hugging Face token (optional)", type="password", key="hf_token_input")
        st.caption("Keys are sent to this app server for API requests and held in your session. "
                   "They are not written to files or shared caches. API charges apply to your account. "
                   "Use a read-only HF token limited to the required dataset.")
        st.form_submit_button("Apply credentials", on_click=_apply_credentials)
    st.button("Disconnect and clear session", on_click=_clear_session)
    generation_mode = st.selectbox("Generation mode", ["LLM-only", "PDF", "RAG"], key="generation_mode")
    dataset_id = ""
    dataset_split = "all"
    if generation_mode == "RAG":
        dataset_id = st.text_input("Hugging Face dataset", "blueper/oxidesc-recipe-ver2")
        dataset_split = st.text_input("Dataset split (or all)", "all")
        st.caption("Your HF account must already have access. A token does not grant new permissions.")
    st.caption("Sequential reviews use model knowledge and supplied references. "
               "The paper's specialist PDF knowledge bases are not bundled.")

openai_api_key = st.session_state.get("active_openai_key", "")
hf_token = st.session_state.get("active_hf_token", "")

# --- Inputs (left sidebar form) ---
with st.sidebar, st.form("recipe_form"):
    material_name = st.text_input("Deposited materials", "In2O3")
    synthesis_technique = st.text_input("Key Deposition/Process Method", "ALD")

    # 분리: 이동도 / 안정성
    mobility_target  = st.text_input("Target Mobility",  "> 20 cm²/V·s")
    stability_target = st.text_input("Target Stability", "< 200 mV")

    other_contstraints = st.text_area("Other Constraints", "")

    model = st.selectbox(
        "Model",
        [
            "gpt-5-mini",
            "gpt-5-high",
            "gpt-5",
            "gpt-5-low",
            "o4-mini-high",
            "o4-mini",
            "o4-mini-low",
            "gpt-4o"
        ],
    )

    requested_top_k = st.slider("Number of Retrievals (RAG only)", 1, 10, 5, disabled=generation_mode != "RAG")
    top_k = requested_top_k if generation_mode == "RAG" else 0

    files = st.file_uploader("Upload PDF Files (PDF/RAG modes)", type=["pdf"], accept_multiple_files=True, disabled=generation_mode == "LLM-only")
    if generation_mode == "LLM-only":
        files = []
    if files:
        for file in files:
            st.write(file.name)

    generate_btn = st.form_submit_button("Recommend", on_click=_start_generation)

# No API requests or dataset access before the user supplies a key and submits.
if not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar and select Apply credentials to start.")
    st.stop()
if generate_btn and generation_mode == "PDF" and not files:
    st.warning("Upload at least one PDF for PDF mode, or choose LLM-only.")
    st.stop()

# === META helpers ===
def _build_key_contributions(material_name, deposition_method, mobility_target, stability_target, other_constraints):
    return {
        "Deposited material": material_name,
        "Key Deposition/Process Method": deposition_method,
        "Target Mobility": mobility_target,
        "Target Stability": stability_target,
        "Other Constraints": other_constraints or "",
    }

def _build_meta(material_name, deposition_method, mobility_target, stability_target, other_constraints, model, top_k, references=None):
    return {
        "key_contributions": _build_key_contributions(material_name, deposition_method, mobility_target, stability_target, other_constraints),
        "model": model,
        "rag_topk": top_k,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "references": references or [],  # dict 리스트(권장) 또는 문자열 리스트
    }

def _dfs_to_json_text(dfs: dict, score: dict | None = None, meta: dict | None = None) -> str:
    """{section: DataFrame} dict → LLM JSON 스키마 문자열 + meta 포함"""
    data = {"meta": meta or {}, "tables": {}, "scorecard": score or {}}
    for section, df in dfs.items():
        if df is None or df.empty:
            continue
        data["tables"][section] = df.to_dict(orient="records")
    return json.dumps(data, ensure_ascii=False, indent=2)

# ====== NEW: References helpers ======
def _normalize_references(refs):
    """PDF 추출 등 문자열 리스트를 구조화된 dict 리스트로 변환"""
    out = []
    for r in (refs or []):
        if isinstance(r, dict) and {"s2_url", "contribution", "recipe"} <= set(r.keys()):
            out.append({
                "s2_url": r.get("s2_url"),
                "contribution": r.get("contribution", ""),
                "recipe": r.get("recipe", ""),
                "rid": r.get("rid", None),
                "pdf_file": r.get("pdf_file", ""),
                "ref_id": r.get("ref_id", None) or r.get("Ref ID", None),
            })
        elif isinstance(r, str):
            out.append({"s2_url": None, "contribution": r, "recipe": "", "rid": None, "ref_id": None})
    return out

def _assign_ref_ids(refs):
    """Reference 리스트에 Ref1, Ref2... 라벨을 부여해 JSON의 Ref IDs와 매핑합니다."""
    out = []
    for i, r in enumerate(refs or [], 1):
        if isinstance(r, dict):
            item = dict(r)
            item["ref_id"] = f"Ref{i}"
            out.append(item)
        else:
            out.append({"s2_url": None, "contribution": str(r), "recipe": "", "rid": None, "ref_id": f"Ref{i}"})
    return out

def _build_reference_context(refs, max_chars: int = 1200) -> str:
    """Ref IDs를 최종 refinement prompt에 넣기 위한 짧은 매핑 문자열."""
    lines = []
    for i, r in enumerate(refs or [], 1):
        if not isinstance(r, dict):
            continue
        ref_id = r.get("ref_id") or f"Ref{i}"
        rid = r.get("rid") or ""
        pdf_file = r.get("pdf_file") or ""
        url = r.get("s2_url") or ""
        contribution = (r.get("contribution") or "").replace("\n", " ").strip()
        recipe = (r.get("recipe") or "").replace("\n", " ").strip()
        evidence = contribution or recipe
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars].rstrip() + "..."
        lines.append(
            f"- {ref_id}: source_id={rid or 'N/A'}; pdf_file={pdf_file or 'N/A'}; "
            f"url={url or 'N/A'}; evidence={evidence or 'N/A'}"
        )
    return "\n".join(lines)

def _refs_to_dataframe(refs):
    """구조화된 ref 리스트 → DataFrame"""
    rows = []
    for i, r in enumerate(refs or [], 1):
        rows.append({
            "Ref ID": r.get("ref_id") or f"Ref{i}",
            "Semantic Scholar URL": r.get("s2_url") or "",
            "Source ID": r.get("rid") or "",
            "PDF File": r.get("pdf_file") or "",
            "Contribution": r.get("contribution") or "",
            "Recipe": r.get("recipe") or "",
        })
    return pd.DataFrame(rows)

with st.sidebar.expander("Load previous prediction tables"):
    prev_file = st.file_uploader("Upload CSV or Excel (previous prediction)", type=["csv", "xlsx"])
    if prev_file is not None:
        dfs = {}
        # ✅ ① 파일 타입 분기 전에 먼저 초기화
        loaded_meta = None
        loaded_refs = []
        try:
            name_low = prev_file.name.lower()
            if name_low.endswith(".csv"):
                df = pd.read_csv(prev_file)
                if "Section" not in df.columns:
                    st.error("CSV에 'Section' 컬럼이 필요합니다.")
                else:
                    for sec, sub in df.groupby("Section"):
                        sub = sub.drop(columns=["Section"], errors="ignore")
                        dfs[str(sec)] = sub.reset_index(drop=True)
            else:
                # Excel은 ExcelFile 핸들을 끝까지 재사용
                xls = pd.ExcelFile(prev_file)
                sheet = "tables_full" if "tables_full" in xls.sheet_names else xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
                if "Section" not in df.columns:
                    st.error(f"엑셀 시트 '{sheet}'에 'Section' 컬럼이 필요합니다.")
                else:
                    for sec, sub in df.groupby("Section"):
                        sub = sub.drop(columns=["Section"], errors="ignore")
                        dfs[str(sec)] = sub.reset_index(drop=True)

                # === NEW: meta & references 시트가 있으면 함께 복원 ===
                if "meta" in xls.sheet_names:
                    try:
                        meta_df = pd.read_excel(xls, sheet_name="meta")
                        if "meta_json" in meta_df.columns and len(meta_df) > 0:
                            mj = meta_df.loc[0, "meta_json"]
                            # ✅ ② meta_json이 이미 dict일 수도 있음
                            loaded_meta = mj if isinstance(mj, dict) else json.loads(mj)
                    except Exception:
                        loaded_meta = None

                if "references" in xls.sheet_names:
                    try:
                        refs_df = pd.read_excel(xls, sheet_name="references").fillna("")
                        for _, row in refs_df.iterrows():
                            loaded_refs.append({
                                "ref_id": row.get("Ref ID", ""),
                                "s2_url": row.get("Semantic Scholar URL", ""),
                                "contribution": row.get("Contribution", ""),
                                "recipe": row.get("Recipe", ""),
                            })
                    except Exception:
                        loaded_refs = []

            if dfs:
                # CSV/엑셀 내 Ref IDs 컬럼 수집
                ref_ids = []
                try:
                    for _, _df in dfs.items():
                        if "Ref IDs" in _df.columns:
                            ref_ids.extend(_df["Ref IDs"].dropna().astype(str).tolist())
                    ref_ids = sorted(set(ref_ids))
                except Exception:
                    ref_ids = []

                # meta/refs가 번들에서 왔으면 우선 사용
                if loaded_meta:
                    if loaded_refs:
                        loaded_meta["references"] = loaded_refs
                    loaded_meta["references"] = _assign_ref_ids(loaded_meta.get("references", []))
                    meta = loaded_meta
                else:
                    meta = _build_meta(
                        material_name, synthesis_technique,
                        mobility_target, stability_target, other_contstraints,
                        model, top_k, references=_assign_ref_ids(ref_ids)
                    )

                json_text = _dfs_to_json_text(dfs, score=None, meta=meta)
                st.session_state.latest_recipe_json = json_text
                st.session_state.loaded_prev_tables = json_text
                st.session_state.latest_meta = meta
                st.session_state.references = meta.get("references", [])
                kc = meta.get("key_contributions", {})
                restored_contribution = PREDICTION_PROMPT.format(
                    material_name=kc.get("Deposited material", ""),
                    deposition_method=kc.get("Key Deposition/Process Method", ""),
                    mobility_target=kc.get("Target Mobility", ""),
                    stability_target=kc.get("Target Stability", ""),
                )
                other_c = kc.get("Other Constraints", "")
                if other_c:
                    restored_contribution += f"\n\n## Other Constraints\n{other_c}"

                st.session_state.original_contribution = restored_contribution
                st.session_state.current_contribution = restored_contribution
                st.session_state.current_recipe = json_text
                st.session_state.fixed_examples = []
                st.session_state.retrieval_examples = meta.get("references", [])
                st.toast("번들 파일을 불러왔습니다. 다음 질문부터 이 컨텍스트를 사용합니다.")
        except Exception as e:
            st.error(f"파일을 해석하는 중 오류가 발생했습니다: {safe_error(e)}")


clear_btn = st.sidebar.button("Clear Conversation")
if clear_btn:
    st.session_state.pop("messages", None)
    st.session_state.pop("references", None)
    st.session_state.pop("latest_recipe_json", None)
    st.session_state.pop("fixed_examples", None)
    st.session_state.pop("retrieval_examples", None)
    st.session_state.pop("original_contribution", None)
    st.session_state.pop("current_contribution", None)
    st.session_state.pop("current_recipe", None)
    st.session_state.pop("latest_meta", None)
    st.session_state.pop("refined_recipe", None)
    st.session_state.pop("initial_pdf_context", None)
    st.session_state.pop("initial_pdf_names", None)
    st.session_state.pop("initial_pdf_warnings", None)
    st.rerun()

# ── Specialist Knowledge Base (폴더 자동 로드) ───────────────────────────
KB_PDF_ROOT = os.path.join(os.path.dirname(__file__), "kb_pdfs")

SPECIALIST_KB_META = {
    "material_properties": ("1️⃣ 재료 특성 (조성비·결정성·프리커서)", os.path.join(KB_PDF_ROOT, "material_properties")),
    "gate_dielectric":     ("2️⃣ 게이트 절연막 계면",                  os.path.join(KB_PDF_ROOT, "gate_dielectric")),
    "process_conditions":  ("3️⃣ 채널 증착 조건 & PDA",               os.path.join(KB_PDF_ROOT, "process_conditions")),
    "sd_relation":         ("4️⃣ 리뷰 논문",                          os.path.join(KB_PDF_ROOT, "sd_relation")),
}


def _load_all_kbs(api_key: str) -> dict:
    """
    kb_pdfs/ 하위 폴더에서 각 specialist KB 를 빌드합니다.
    명시적으로 빌드하며 현재 사용자 세션에만 보관합니다.
    PDF 파일이 없는 폴더는 건너뜁니다.
    """
    import glob
    kbs = {}
    for key, (label, folder) in SPECIALIST_KB_META.items():
        pdfs = glob.glob(os.path.join(folder, "*.pdf"))
        if not pdfs:
            continue
        try:
            kb = build_kb_from_directory(folder, api_key=api_key)
            if kb and len(kb) > 0:
                kbs[key] = kb
        except Exception as e:
            # KB 빌드 실패는 앱 실행을 막지 않음
            st.warning(f"{label}: specialist PDF loading failed.")
    return kbs


# KB 로드 (API 키가 있을 때만)
_kb_cache: dict = st.session_state.get("specialist_kbs", {})

# 사이드바: KB 상태 표시 + 수동 갱신 버튼
with st.sidebar.expander("📚 리뷰 논문", expanded=False):
    st.caption(
        f"**kb_pdfs/** 하위 폴더에 PDF 를 넣으면 답변과 레시피 검토에 자동으로 참조합니다.\n\n"
        "Specialist sources are optional and are not included in this public release."
    )

    import glob as _glob
    any_pdf = False
    for key, (label, folder) in SPECIALIST_KB_META.items():
        pdfs = _glob.glob(os.path.join(folder, "*.pdf"))
        kb = _kb_cache.get(key)
        if pdfs:
            any_pdf = True
            status = f"✅ {len(kb)} chunks ({len(pdfs)} files)" if kb else "⚠️ 빌드 실패"
        else:
            status = "— (PDF 없음)"
        st.write(f"**{label}**: {status}")
        if pdfs:
            with st.expander(f"  파일 목록", expanded=False):
                for p in pdfs:
                    st.write(f"  • {os.path.basename(p)}")

    st.markdown("---")
    if st.button("🔄 KB 재빌드 (파일 변경 시)", use_container_width=True):
        st.session_state.specialist_kbs = _load_all_kbs(openai_api_key)
        st.rerun()

    kb_top_k = st.slider("KB 검색 청크 수 (top-k)", 1, 6, 3, key="kb_top_k")

    if not any_pdf:
        st.info(
            "아직 PDF 가 없습니다. 위 경로의 폴더에 논문 PDF 를 넣은 뒤 "
            "앱을 재시작하거나 '재빌드' 버튼을 누르세요."
        )


def _get_knowledge_bases() -> dict:
    """캐싱된 Specialist KB 딕셔너리를 반환합니다.
    키: material_properties, gate_dielectric, process_conditions
    """
    return _kb_cache

if not generate_btn and not hasattr(st.session_state, "messages") and not st.session_state.get("latest_recipe_json"):
    st.write(
        "This is a demo of the Materials Synthesis Recipe Generator. "
        "Please enter the desired material properties and click on the 'Recommend' button "
        "to get a list of materials synthesis recipes that can be used to synthesize materials with the desired properties."
    )
    st.stop()

use_rag = top_k >= 1

# ---------------- Model helpers ----------------
def get_model_config(model_name):
    """모델별 설정을 반환하는 통합 함수"""
    if model_name.startswith("gpt-5"):
        base_model = "gpt-5" if model_name != "gpt-5-mini" else "gpt-5-mini"
        reasoning_effort = None
        if "high" in model_name:
            reasoning_effort = "high"
        elif "low" in model_name:
            reasoning_effort = "low"
        return base_model, reasoning_effort, True  # max_completion_tokens 사용

    elif model_name.startswith("o4"):
        base_model = "o4-mini"
        reasoning_effort = None
        if "high" in model_name:
            reasoning_effort = "high"
        elif "low" in model_name:
            reasoning_effort = "low"
        return base_model, reasoning_effort, True

    elif model_name.startswith("o1") or model_name.startswith("o3"):
        base_model = model_name.replace("-high", "").replace("-low", "")
        reasoning_effort = "high" if "high" in model_name else "low" if "low" in model_name else None
        return base_model, reasoning_effort, True

    else:
        # 기존 GPT 모델들
        return model_name, None, False

def call_completion_model(messages, model_name):
    """통합된 모델 호출 함수"""
    base_model, reasoning_effort, use_completion_tokens = get_model_config(model_name)

    if use_completion_tokens:
        kwargs = {
            "model": base_model,
            "messages": messages,
            "max_completion_tokens": 16384
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs = {
            "model": base_model,
            "messages": messages,
            "max_tokens": 4096
        }

    return completion(api_key=openai_api_key, **kwargs)

# ---------------- JSON parsing & rendering utils ----------------
def _pull_json_like(text: str) -> str | None:
    """텍스트에서 가장 믿을만한 JSON 블록 하나만 추출 (마지막 등장 우선)."""
    if not isinstance(text, str) or "{" not in text or "}" not in text:
        return None
    blocks = []

    raw = text
    # 1) 코드펜스 내부 우선 수집
    if "```" in raw:
        parts = raw.split("```")
        for i in range(1, len(parts), 2):
            inner = parts[i]  # 코드펜스 내부 블록
            if "{" in inner and "}" in inner:
                s, e = inner.find("{"), inner.rfind("}")
                if s >= 0 and e > s:
                    blocks.append(inner[s : e + 1].strip())

    # 2) 코드펜스 밖 후보
    try:
        s, e = raw.index("{"), raw.rindex("}")
        blocks.append(raw[s : e + 1].strip())
    except Exception:
        pass

    # 3) 가장 마지막 유효 JSON 반환
    for cand in reversed(blocks):
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return None

def _split_prose_and_json(text: str):
    """답변에서 줄글과 JSON을 분리. JSON 없으면 (원문, None)."""
    if not isinstance(text, str):
        return "", None
    json_block = _pull_json_like(text)
    if not json_block:
        return text.strip(), None

    prose = text
    fenced = f"```json\n{json_block}\n```"
    if fenced in prose:
        prose = prose.replace(fenced, "").strip()
    else:
        prose = prose.replace(json_block, "").strip()
    return prose, json_block

def _json_to_tables(json_text: str):
    """
    LLM이 반환한 JSON(또는 JSON이 포함된 텍스트)을 받아
    scorecard(dict)와 {섹션명: DataFrame} 딕셔너리를 리턴.
    유효하지 않으면 (None, {}) 반환.
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return None, {}

    raw = json_text.strip()

    # 1) 코드펜스 제거
    if "```" in raw:
        try:
            fence_blocks = raw.split("```")
            candidate = None
            for i in range(1, len(fence_blocks), 2):
                block = fence_blocks[i]
                if "{" in block and "}" in block:
                    candidate = block
                    break
            if candidate:
                raw = candidate.strip()
        except Exception:
            pass

    # 2) 첫 '{' ~ 마지막 '}' 사이
    if "{" in raw and "}" in raw:
        try:
            start = raw.index("{")
            end = raw.rindex("}")
            raw = start and raw[start : end + 1] or raw
        except Exception:
            pass

    # 3) JSON 로드
    try:
        data = json.loads(raw)
    except Exception:
        return None, {}

    if not isinstance(data, dict):
        return None, {}

    score = data.get("scorecard", None)
    tables = data.get("tables", {})

    dfs = {}
    if isinstance(tables, dict):
        for section, rows in tables.items():
            if isinstance(rows, list):
                try:
                    dfs[section] = pd.DataFrame(rows)
                except Exception:
                    pass

    return score, _ensure_rationale_refs(dfs)

def _ensure_rationale_refs(dfs: dict) -> dict:
    """모델이 누락해도 표/다운로드에 Rationale과 Ref IDs 컬럼이 항상 보이도록 보강."""
    fixed = {}
    for section, df in (dfs or {}).items():
        if df is None or df.empty:
            fixed[section] = df
            continue

        out = df.copy()
        if "Rationale" not in out.columns:
            out["Rationale"] = ""
        if "Ref IDs" not in out.columns:
            out["Ref IDs"] = ""

        out["Rationale"] = (
            out["Rationale"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Generated from target constraints, specialist review, and retrieved examples.")
        )
        out["Ref IDs"] = (
            out["Ref IDs"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "N/A")
        )
        fixed[section] = out
    return fixed

def _concat_tables(dfs: dict) -> pd.DataFrame:
    """섹션별 DataFrame(dict) → 단일 DataFrame(Section 컬럼 포함)로 병합"""
    rows = []
    for section, df in dfs.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "Section", section)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _render_scorecard(score: dict):
    """Streamlit에 scorecard를 렌더링."""
    if not isinstance(score, dict):
        return
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence (0–5)", f"{score.get('confidence', 'N/A')}")
    with c2:
        st.metric("Similar refs", f"{score.get('similar_refs', 'N/A')}")
    st.write("**Why**:", score.get("why", ""))
    risks = score.get("risks", [])
    if isinstance(risks, list) and len(risks) == 2:
        st.write(f"**Risks:** 1) {risks[0]}  2) {risks[1]}")

def _offer_downloads_unified(all_df_simple: pd.DataFrame,
                             all_df_full: pd.DataFrame,
                             base_name: str = "prediction_tables",
                             key_prefix: str = "dl"):
    """단일 CSV/Excel 다운로드 제공."""
    if all_df_full is None or all_df_full.empty:
        return

    st.subheader("Download")

    # ---- CSV (simple)
    csv_simple = all_df_simple.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV — All tables (simple)",
        data=csv_simple,
        file_name=f"{base_name}__all_simple.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}__csv_simple"
    )

    # ---- CSV (full)
    csv_full = all_df_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV — All tables (with rationale & refs)",
        data=csv_full,
        file_name=f"{base_name}__all_full.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}__csv_full"
    )

    # ---- Excel (두 시트 제공: simple / full)
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        writer.book.use_zip64()
        all_df_simple.to_excel(writer, index=False, sheet_name="tables_simple")
        all_df_full.to_excel(writer, index=False, sheet_name="tables_full")
    st.download_button(
        label="⬇️ Download Excel — All tables",
        data=xbuf.getvalue(),
        file_name=f"{base_name}__all.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"{key_prefix}__excel"
    )

def _render_json_tables_block(text: str,
                              base_name: str = "prediction_tables",
                              expanded: bool = True,
                              title: str = "📦 Parse JSON to tables"):
    """어떤 텍스트든 JSON 테이블을 찾아 렌더링 + 단일 CSV/Excel 다운로드 제공"""
    with st.expander(title, expanded=expanded):
        score, dfs = _json_to_tables(text)
        if not dfs:
            st.info("응답에서 유효한 JSON 테이블 구조를 찾지 못했습니다. 템플릿 준수 여부를 확인하세요.")
            return

        # 상단 요약 카드
        if score:
            _render_scorecard(score)
            st.markdown("---")

        # 🔑 모든 위젯에 고유 key 부여
        key_prefix = f"{base_name}__{abs(hash(text))}"

        show_details = st.checkbox(
            "🔎 Show rationale & Ref IDs",
            value=True,
            key=f"{key_prefix}__show_details"
        )

        st.subheader("Tables")
        for section, df in dfs.items():
            st.markdown(f"**{section}**")
            df_show = df if show_details else df.drop(columns=["Rationale", "Ref IDs"], errors="ignore")
            st.dataframe(df_show, use_container_width=True, key=f"{key_prefix}__df__{section}")

        # 단일 다운로드용 전체 테이블 생성
        all_df_full = _concat_tables(dfs)
        all_df_simple = all_df_full.drop(columns=["Rationale", "Ref IDs"], errors="ignore")

        _offer_downloads_unified(
            all_df_simple, all_df_full,
            base_name=base_name,
            key_prefix=key_prefix,
        )

def classify_feedback_intent(model_name: str, current_recipe: str, user_feedback: str) -> str:
    prompt = f"""
You are classifying the user's follow-up intent.

Current recipe:
{current_recipe}

User feedback:
{user_feedback}

Decide one of the following only:
- qa : the user is asking a simple question about the current recipe, references, or rationale
- revise : the user wants the recipe to be regenerated, updated, improved, or modified based on feedback

Return only one word:
qa
or
revise
""".strip()

    resp = call_completion_model(
        [{"role": "user", "content": prompt}],
        model_name
    )
    text = resp["choices"][0]["message"]["content"].strip().lower()

    if "revise" in text:
        return "revise"
    return "qa"

def revise_contribution_with_feedback(
    model_name: str,
    original_contribution: str,
    current_contribution: str,
    current_recipe: str,
    user_feedback: str,
) -> str:
    prompt = f"""
You improve the contribution/specification for a new retrieval query.

Original contribution:
{original_contribution}

Current contribution:
{current_contribution}

Current recipe:
{current_recipe}

Human feedback:
{user_feedback}

Rewrite the contribution so that:
1. the user's original goal is preserved
2. the human feedback is explicitly reflected
3. it is better suited for retrieving useful examples
4. it remains concise and structured

Return only the revised contribution text.
""".strip()

    resp = call_completion_model(
        [{"role": "user", "content": prompt}],
        model_name
    )
    return resp["choices"][0]["message"]["content"].strip()

def regenerate_recipe_with_feedback(
    model_name: str,
    revised_contribution: str,
    current_recipe: str,
    fixed_examples: list | None,
    top_k: int,
):
    emb = get_embedding(revised_contribution) if top_k > 0 else None

    predictor = rag_predictor
    predictor.model = model_name

    fixed_examples = fixed_examples or []
    retrieval_examples = []

    if top_k > 0:
        ref_df = predictor.search(emb, k=top_k, return_rows=True)
        if ref_df is not None and len(ref_df) > 0:
            limit = min(top_k, len(ref_df["recipe"]))
            for i in range(limit):
                rid = ref_df["id"][i] if "id" in ref_df else ""
                contribution = ref_df["contribution"][i] if "contribution" in ref_df else ""
                recipe_txt = ref_df["recipe"][i] if "recipe" in ref_df else ""
                pdf_file = ref_df["pdf_file"][i] if "pdf_file" in ref_df else ""
                retrieval_examples.append({
                    "s2_url": rid if isinstance(rid, str) and rid.startswith("http") else None,
                    "contribution": contribution,
                    "recipe": recipe_txt,
                    "rid": rid,
                    "pdf_file": pdf_file,
                    "ref_id": f"Ref{len(fixed_examples) + i + 1}",
                })

    base_examples = fixed_examples + retrieval_examples
    predictor.base_references = base_examples

    prev_rag_topk = getattr(predictor, "rag_topk", 0)
    predictor.rag_topk = 0

    batch = [
        {
            "contribution": revised_contribution + "\n\nPrevious recipe to revise (not literature evidence):\n" + current_recipe,
            "recipe": "",
            "contributions_embedding": emb,
        }
    ]

    output = ""
    try:
        for _, output in predictor.predict(batch):
            pass
    finally:
        predictor.rag_topk = prev_rag_topk

    used_references = _assign_ref_ids(list(fixed_examples) + list(retrieval_examples))
    return output, retrieval_examples, used_references, revised_contribution

# ---------------- Prompt ----------------
PREDICTION_PROMPT = """## Key Contributions
- **Deposited material**: {material_name}
- **Key Deposition/Process Method**: {deposition_method}
- **Target Mobility**: {mobility_target}
- **Target Stability**: {stability_target}
""".strip()

# ---------------- Embedding & Predictors ----------------
def get_embedding(contributions):
    response = embedding(model='text-embedding-3-large', input=[contributions], api_key=openai_api_key)
    emb = response['data'][0]['embedding']
    return emb

def get_predictors(model_name, api_key, rag_topk):
    from cloud_runtime import credential_fingerprint
    signature = (model_name, rag_topk, generation_mode, dataset_id, dataset_split,
                 credential_fingerprint(api_key, hf_token))
    if st.session_state.get("predictor_signature") != signature:
        reference_predictor = ReferenceRecipePredictor(model=model_name, api_key=api_key)
        if generation_mode == "RAG":
            if not dataset_id.strip() or not hf_token:
                raise ValueError("Provide a dataset ID and HF token for RAG mode.")
            rag = RAGRecipePredictor(
                model=model_name, api_key=api_key, rag_topk=rag_topk,
                retrieval_split=dataset_split.strip() or "all",
                hf_token=hf_token, dataset_id=dataset_id.strip())
        else:
            rag = reference_predictor
        base = RecipePredictor(model=model_name, api_key=api_key)
        st.session_state.predictors = (rag, base)
        st.session_state.predictor_signature = signature
    return st.session_state.predictors


with st.status("Preparing generation mode…", expanded=False) as _init_status:
    try:
        rag_predictor, base_predictor = get_predictors(model, openai_api_key, top_k)
        _init_status.update(label="Ready", state="complete")
    except Exception:
        st.session_state.pop("predictors", None)
        st.session_state.pop("predictor_signature", None)
        _init_status.update(label="Connection or dataset unavailable", state="error")
        st.error("Could not prepare this mode. For RAG, check the read token, dataset access, "
                 "split and documented schema. You can select LLM-only or PDF without a dataset.")
        st.stop()

# ---------------- Core predict & ask flows ----------------
def predict_recipe(
    material_name,
    synthesis_technique,
    mobility_target,
    stability_target,
    other_contstraints,
    top_k,
    model,
    use_rag,
    files=None,
    step_callback=None,
):
    def _step(msg):
        if step_callback:
            step_callback(msg)

    contributions = PREDICTION_PROMPT.format(
        material_name=material_name,
        deposition_method=synthesis_technique,
        mobility_target=mobility_target,
        stability_target=stability_target,
    )

    if use_rag or files:
        predictor = rag_predictor
        emb = get_embedding(contributions) if use_rag else None
    else:
        predictor = base_predictor
        emb = None

    pdf_bytes_list = []
    if files:
        _step(f"📄 PDF {len(files)}개에서 레시피 추출 중…")
        pdf_bytes_list = [file.getvalue() for file in files]
        initial_pdf_context, initial_pdf_names, initial_pdf_warnings = _build_initial_pdf_context(files)
        st.session_state.initial_pdf_context = initial_pdf_context
        st.session_state.initial_pdf_names = initial_pdf_names
        st.session_state.initial_pdf_warnings = initial_pdf_warnings
        for warning in initial_pdf_warnings:
            _step(f"⚠️ {warning}")
        references = pdf_bytelist_to_recipes(pdf_bytes_list, api_key=openai_api_key)
        if not references:
            raise ValueError("No readable recipe content was extracted from the uploaded PDFs.")
        _step(f"✅ PDF 추출 완료 ({len(references)}개 참조)")
    else:
        st.session_state.initial_pdf_context = ""
        st.session_state.initial_pdf_names = []
        st.session_state.initial_pdf_warnings = []
        references = None

    references = _normalize_references(references)
    fixed_examples = list(references) if references else []
    st.session_state.fixed_examples = fixed_examples
    predictor.base_references = fixed_examples
    predictor.model = model

    if other_contstraints:
        contributions += f"\n\n## Other Constraints\n{other_contstraints}"

    if use_rag and top_k > 0:
        _step(f"🔍 FAISS 유사 논문 검색 중 (top-{top_k})…")

    batch = [
        {
            "contribution": contributions,
            "recipe": "",
            "contributions_embedding": emb,
        }
    ]

    _step(f"🤖 {model} 으로 레시피 초안 생성 중…")
    output = ""
    for _, output in predictor.predict(batch):
        pass
    _step("✅ 레시피 초안 생성 완료")

    retrieval_examples = []
    if use_rag:
        ref_df = predictor.search(emb, k=top_k, return_rows=True)
        if ref_df is not None and len(ref_df) > 0:
            limit = min(top_k, len(ref_df["recipe"]))
            for i in range(limit):
                rid = ref_df["id"][i] if "id" in ref_df else ""
                contribution = ref_df["contribution"][i] if "contribution" in ref_df else ""
                recipe_txt = ref_df["recipe"][i] if "recipe" in ref_df else ""
                pdf_file = ref_df["pdf_file"][i] if "pdf_file" in ref_df else ""
                retrieval_examples.append({
                    "s2_url": rid if isinstance(rid, str) and rid.startswith("http") else None,
                    "contribution": contribution,
                    "recipe": recipe_txt,
                    "rid": rid,
                    "pdf_file": pdf_file,
                    "ref_id": f"Ref{len(fixed_examples) + i + 1}",
                })
        _step(f"✅ 유사 논문 {len(retrieval_examples)}개 검색 완료")

    st.session_state.retrieval_examples = retrieval_examples
    references = _assign_ref_ids(fixed_examples + retrieval_examples) if (fixed_examples or retrieval_examples) else None

    prompt = predictor.build_prompt(batch[0])[0]["content"]

    meta = _build_meta(
        material_name, synthesis_technique,
        mobility_target, stability_target, other_contstraints,
        model, top_k, references=references or []
    )
    st.session_state.latest_meta = meta
    st.session_state.original_contribution = contributions
    st.session_state.current_contribution = contributions
    st.session_state.current_recipe = output
    return output, references, prompt


def ask_with_previous_recipe(messages_display, model_name, recipe_text: str, user_question: str) -> str:
    """화면 히스토리는 건드리지 않고, 이전 레시피 JSON을 문맥으로 모델에 묻는다."""
    prev_json = st.session_state.get("latest_recipe_json") or _pull_json_like(recipe_text)

    if prev_json:
        composed = (
            "You are given the previously predicted recipe in JSON.\n"
            "Use it as context to answer the user's question.\n"
            "If you include any JSON in your reply, put it in ONE ```json code fence.\n\n"
            "<RECIPE_JSON>\n"
            f"{prev_json}\n"
            "</RECIPE_JSON>\n\n"
            "<USER_QUESTION>\n"
            f"{user_question}\n"
            "</USER_QUESTION>"
        )
    else:
        composed = (
            "No structured JSON was found for the previous prediction; the raw text follows.\n"
            "If you include any JSON in your reply, put it in ONE ```json code fence.\n\n"
            "<RECIPE_RAW>\n"
            f"{recipe_text}\n"
            "</RECIPE_RAW>\n\n"
            "<USER_QUESTION>\n"
            f"{user_question}\n"
            "</USER_QUESTION>"
        )

    # 모델 호출용 임시 히스토리만 생성: 사용자 원문 대신 composed를 넣는다.
    messages_for_model = messages_display[:-1] + [{"role": "user", "content": composed}]
    resp = call_completion_model(messages_for_model, model_name)
    asst_text = resp["choices"][0]["message"]["content"]

    # 화면 히스토리에는 assistant 답변만 추가
    messages_display.append({"role": "assistant", "content": asst_text})
    return asst_text


def _dataframe_preview_text(df: pd.DataFrame, max_rows: int = 80, max_cols: int = 20) -> str:
    """첨부 표 데이터를 LLM 컨텍스트에 넣기 좋은 작은 텍스트로 변환."""
    if df is None or df.empty:
        return "(empty table)"
    preview = df.iloc[:max_rows, :max_cols].copy()
    return preview.to_csv(index=False)


def _clip_or_focus_text(text: str, max_chars: int = 50000) -> str:
    """긴 PDF 텍스트는 앞부분 + 공정/전극/성능 관련 핵심 발췌로 줄인다."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    keywords = [
        "source/drain", "source-drain", "source and drain", "s/d",
        "source", "drain", "contact", "electrode",
        "deposition", "ald", "sputter", "anneal", "pda",
        "mobility", "stability", "threshold", "bias stress",
        "precursor", "temperature", "oxygen", "vacancy",
    ]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    focused = []
    focused_chars = 0
    focus_budget = max_chars // 2
    for paragraph in paragraphs:
        low = paragraph.lower()
        if any(keyword in low for keyword in keywords):
            focused.append(paragraph)
            focused_chars += len(paragraph)
            if focused_chars >= focus_budget:
                break

    head_budget = max_chars - min(focused_chars, focus_budget) - 500
    head = text[:max(8000, head_budget)].rstrip()
    focused_text = "\n\n".join(focused).strip()
    if focused_text:
        return (
            f"{head}\n\n"
            "[FOCUSED_EXCERPTS_FROM_LONG_DOCUMENT]\n"
            f"{focused_text}\n"
            "[/FOCUSED_EXCERPTS_FROM_LONG_DOCUMENT]\n\n"
            "[TRUNCATED_AND_FOCUSED]"
        )
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def _read_uploaded_file_for_context(uploaded_file, max_chars: int = 50000) -> tuple[str, str | None]:
    """PDF/Excel/CSV 첨부파일을 질문 컨텍스트용 텍스트로 변환."""
    name = uploaded_file.name
    ext = os.path.splitext(name.lower())[1]
    raw = uploaded_file.getvalue()

    try:
        if ext == ".pdf":
            text = pdf_bytes_to_markdown(raw)
        elif ext in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(BytesIO(raw))
            parts = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                parts.append(f"## Sheet: {sheet}\n{_dataframe_preview_text(df)}")
            text = "\n\n".join(parts)
        elif ext == ".csv":
            df = pd.read_csv(BytesIO(raw))
            text = _dataframe_preview_text(df)
        else:
            return "", f"{name}: 지원하지 않는 파일 형식입니다."
    except Exception as e:
        return "", f"{name}: 파일을 읽는 중 오류가 발생했습니다: {safe_error(e)}"

    text = (text or "").strip()
    if not text:
        return "", f"{name}: 추출된 텍스트가 없습니다."
    text = _clip_or_focus_text(text, max_chars=max_chars)
    return f"<ATTACHED_FILE name=\"{name}\">\n{text}\n</ATTACHED_FILE>", None


def _pdf_bytes_to_context(name: str, raw: bytes, max_chars: int = 50000) -> tuple[str, str | None]:
    """초기 업로드 PDF 원문을 이후 Q&A에서 재사용할 컨텍스트로 변환."""
    try:
        text = pdf_bytes_to_markdown(raw)
    except Exception as e:
        return "", f"{name}: PDF 원문을 읽는 중 오류가 발생했습니다: {safe_error(e)}"

    text = (text or "").strip()
    if not text:
        return "", f"{name}: 추출된 PDF 원문 텍스트가 없습니다."
    text = _clip_or_focus_text(text, max_chars=max_chars)
    return f"<INITIAL_UPLOADED_PDF name=\"{name}\">\n{text}\n</INITIAL_UPLOADED_PDF>", None


def _build_attachment_context(uploaded_files) -> tuple[str, list[str], list[str]]:
    """여러 첨부파일을 하나의 모델 컨텍스트로 묶는다."""
    contexts = []
    names = []
    warnings = []
    for uploaded_file in uploaded_files or []:
        names.append(uploaded_file.name)
        context, warning = _read_uploaded_file_for_context(uploaded_file)
        if context:
            contexts.append(context)
        if warning:
            warnings.append(warning)

    if not contexts:
        return "", names, warnings

    attachment_context = (
        "The user attached the following files. Use them together with the recipe context "
        "when answering or revising.\n\n"
        + "\n\n".join(contexts)
    )
    return attachment_context, names, warnings


def _build_initial_pdf_context(uploaded_files) -> tuple[str, list[str], list[str]]:
    """Recommend 단계에서 올린 PDF 원문을 Q&A 재사용 컨텍스트로 저장."""
    contexts = []
    names = []
    warnings = []
    for uploaded_file in uploaded_files or []:
        names.append(uploaded_file.name)
        context, warning = _pdf_bytes_to_context(uploaded_file.name, uploaded_file.getvalue())
        if context:
            contexts.append(context)
        if warning:
            warnings.append(warning)

    if not contexts:
        return "", names, warnings

    initial_context = (
        "The following PDFs were uploaded when the recipe was initially generated. "
        "Use them as primary context for follow-up Q&A when relevant.\n\n"
        + "\n\n".join(contexts)
    )
    return initial_context, names, warnings


def _build_qa_kb_context(user_text: str, top_k: int | None = None) -> str:
    """Q&A에서 모든 specialist KB를 검색해 질문 컨텍스트로 추가."""
    kbs = _get_knowledge_bases()
    if not kbs:
        return ""

    try:
        recipe_context = st.session_state.get("latest_recipe_json", "")
        query = (
            "oxide semiconductor thin-film transistor recipe review context. "
            f"User question: {user_text}\n\n"
            f"Current recipe context:\n{recipe_context[:6000]}"
        )
        q_emb = get_embedding(query)
    except Exception as e:
        return f"Specialist KB search failed: {safe_error(e)}"

    per_kb_top_k = top_k or st.session_state.get("kb_top_k", 3)
    sections = []
    for key, kb in kbs.items():
        if kb is None or len(kb) == 0:
            continue
        label = SPECIALIST_KB_META.get(key, (key, ""))[0]
        try:
            chunks = kb.query(q_emb, top_k=per_kb_top_k)
        except Exception as e:
            sections.append(f"## {label}\nSearch failed: {safe_error(e)}")
            continue
        if chunks:
            sections.append(
                f"## {label}\n"
                + "\n\n---\n\n".join(chunks)
            )

    if not sections:
        return ""

    return (
        "Relevant specialist literature excerpts from kb_pdfs. "
        "Use these as supporting context when answering or revising, but prioritize the current recipe and user question.\n\n"
        + "\n\n==========\n\n".join(sections)
    )


def _compose_user_prompt_with_attachments(user_text: str, attachment_context: str, qa_kb_context: str = "") -> str:
    """사용자 질문과 첨부파일 텍스트를 모델 입력용으로 결합."""
    user_text = (user_text or "").strip()
    initial_pdf_context = st.session_state.get("initial_pdf_context", "")
    context_parts = []
    if initial_pdf_context:
        context_parts.append(
            "<INITIAL_RECIPE_UPLOADS_CONTEXT>\n"
            f"{initial_pdf_context}\n"
            "</INITIAL_RECIPE_UPLOADS_CONTEXT>"
        )
    if attachment_context:
        context_parts.append(
            "<USER_ATTACHED_FILES_CONTEXT>\n"
            f"{attachment_context}\n"
            "</USER_ATTACHED_FILES_CONTEXT>"
        )
    if qa_kb_context:
        context_parts.append(
            "<QA_SPECIALIST_KB_CONTEXT>\n"
            f"{qa_kb_context}\n"
            "</QA_SPECIALIST_KB_CONTEXT>"
        )

    if not context_parts:
        return user_text
    return (
        f"{user_text}\n\n"
        + "\n\n".join(context_parts)
    ).strip()

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent UI helper
# ─────────────────────────────────────────────────────────────────────────────

STEP_LABELS = {
    1: ("🔬 Step 1/3", "재료 특성 (조성비·결정성·프리커서) 검토"),
    2: ("🌡️ Step 2/3", "채널 증착 조건 & Post-Deposition Annealing 검토"),
    3: ("🔗 Step 3/3", "피드백 통합 → 개량 레시피 생성"),
}

def _run_refinement_with_status(raw_recipe_json: str, label_prefix: str = "") -> dict:
    """
    st.status 위젯으로 각 스텝을 실시간 표시하며 refinement 파이프라인 실행.
    반환값: run_refinement_sequential 결과 dict
    """
    result_holder: dict = {}

    step_containers: dict = {}

    with st.status(f"{label_prefix}Multi-Agent 검토 진행 중…", expanded=True) as status_widget:

        def _on_step(step_num: int, label: str, feedback=None):
            icon, title = STEP_LABELS.get(step_num, ("🔹", label))
            if feedback is None:
                # 스텝 시작
                st.write(f"{icon} **{title}** — 시작")
                step_containers[step_num] = st.empty()
            else:
                # 스텝 완료 → expander로 피드백 표시
                if step_num < 4:
                    with st.expander(f"{icon} {title} — 완료", expanded=False):
                        st.write(feedback)
                else:
                    st.write(f"{icon} **{title}** — 완료 ✅")

        try:
            result = run_refinement_sequential(
                recipe_json=raw_recipe_json,
                api_key=openai_api_key,
                specialist_model="gpt-4o",
                step_callback=_on_step,
                knowledge_bases=_get_knowledge_bases(),
                kb_top_k=st.session_state.get("kb_top_k", 3),
                reference_context=_build_reference_context(
                    st.session_state.get("references")
                    or (st.session_state.get("latest_meta") or {}).get("references", [])
                ),
            )
            refined = result.get("refined_recipe", "")
            if refined:
                _, refined_dfs = _json_to_tables(refined)
                if not refined_dfs:
                    json_only = _pull_json_like(refined)
                    _, json_only_dfs = _json_to_tables(json_only or "")
                    if json_only_dfs:
                        result["refined_recipe"] = json_only
                    else:
                        _, raw_dfs = _json_to_tables(raw_recipe_json)
                        if raw_dfs:
                            result["refined_recipe"] = raw_recipe_json
                            result["refinement_warning"] = (
                                "Multi-agent refinement returned non-parseable JSON, "
                                "so the parseable draft recipe is shown instead."
                            )
            result_holder.update(result)
            status_widget.update(label=f"{label_prefix}Multi-Agent 검토 완료 ✅", state="complete", expanded=False)
        except Exception as e:
            status_widget.update(label=f"{label_prefix}오류 발생", state="error", expanded=True)
            st.error(safe_error(e))

    return result_holder


def _render_refinement_result(result: dict, base_key: str = "initial"):
    """개량 결과 dict를 테이블 + 다운로드 버튼으로 렌더링."""
    refined_json_str = result.get("refined_recipe", "")

    # Specialist 피드백 (접힌 expander)
    if result.get("refinement_warning"):
        st.warning(result.get("refinement_warning"))

    with st.expander("📋 Specialist 검토 요약 보기", expanded=False):
        with st.expander("1️⃣ 재료 특성 (조성비·결정성·프리커서) 검토", expanded=False):
            st.write(result.get("material_properties") or "_결과 없음_")
        with st.expander("2️⃣ 채널 증착 조건 & PDA 검토", expanded=False):
            st.write(result.get("process_conditions") or "_결과 없음_")

    # 개량 레시피 테이블
    if refined_json_str:
        _render_json_tables_block(
            refined_json_str,
            base_name=f"refined_recipe__{base_key}",
            expanded=True,
            title="📦 Multi-Agent 개량 레시피",
        )
        # 다운로드
        meta_show = st.session_state.get("latest_meta")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ 개량 레시피 (JSON)",
                data=refined_json_str.encode("utf-8"),
                file_name=f"refined_recipe_{base_key}.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_refined_json__{base_key}",
            )
        with dl_col2:
            if meta_show:
                _, _dfs = _json_to_tables(refined_json_str)
                _all_df = _concat_tables(_dfs) if _dfs else pd.DataFrame()
                _refs_df = _refs_to_dataframe(meta_show.get("references", []))
                _xbuf = BytesIO()
                with pd.ExcelWriter(_xbuf, engine="xlsxwriter") as _writer:
                    _writer.book.use_zip64()
                    if not _all_df.empty:
                        _all_df.to_excel(_writer, index=False, sheet_name="tables_full")
                        _all_df.drop(columns=["Rationale", "Ref IDs"], errors="ignore").to_excel(
                            _writer, index=False, sheet_name="tables_simple"
                        )
                    pd.DataFrame([{"meta_json": json.dumps(meta_show, ensure_ascii=False)}]).to_excel(
                        _writer, index=False, sheet_name="meta"
                    )
                    if not _refs_df.empty:
                        _refs_df.to_excel(_writer, index=False, sheet_name="references")
                st.download_button(
                    label="⬇️ All-in-One Excel",
                    data=_xbuf.getvalue(),
                    file_name=f"refined_bundle_{base_key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"dl_refined_excel__{base_key}",
                )
    else:
        st.warning("개량 레시피 JSON 을 파싱할 수 없습니다.")
        st.text_area("Raw output", refined_json_str, height=200, key=f"raw__{base_key}")


# ─────────────────────────────────────────────────────────────────────────────
# Session bootstrap  ── 초기 레시피 생성 & 자동 multi-agent 실행
# ─────────────────────────────────────────────────────────────────────────────
if not hasattr(st.session_state, "messages"):
    st.session_state.messages = []

    if st.session_state.get("latest_recipe_json"):
        # 이전 파일 로드 경로: 이미 레시피가 있으면 그 레시피를 refinement 입력으로 사용
        recipe = st.session_state.latest_recipe_json
        references = st.session_state.get("references") or (st.session_state.get("latest_meta") or {}).get("references", [])
        references = _assign_ref_ids(references)
        st.session_state.references = references
        st.session_state.current_recipe = recipe
        st.session_state.messages.append({"role": "user", "content": "Loaded previous prediction tables as context."})
        st.session_state.messages.append({"role": "assistant", "content": recipe})
    else:
        # 신규 생성 경로
        with st.status("📋 레시피 생성 진행 중…", expanded=True) as _predict_status:
            try:
                recipe, references, user_prompt = predict_recipe(
                    material_name, synthesis_technique,
                    mobility_target, stability_target, other_contstraints,
                    top_k, model, use_rag, files=files,
                    step_callback=st.write,
                )
            except Exception as error:
                st.session_state.pop("messages", None)
                _predict_status.update(label="Generation failed", state="error")
                st.error(safe_error(error))
                st.stop()
            _predict_status.update(label="✅ 레시피 초안 완성", state="complete", expanded=False)
            st.session_state.references = references
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.session_state.messages.append({"role": "assistant", "content": recipe})
            st.session_state.latest_recipe_json = recipe
            st.session_state.current_recipe = recipe

    # 자동 multi-agent refinement 실행
    st.subheader("🔬 Multi-Agent 검토 & 개량")
    ma_result = _run_refinement_with_status(recipe, label_prefix="초기 레시피 ")
    if ma_result:
        refined_str = ma_result.get("refined_recipe", "")
        st.session_state.refined_result_initial = ma_result
        if refined_str:
            st.session_state.latest_recipe_json = refined_str
            st.session_state.current_recipe = refined_str

else:
    recipe = st.session_state.get("current_recipe") or (
        st.session_state.messages[1]["content"] if len(st.session_state.messages) > 1 else ""
    )
    references = st.session_state.get("references")

# ─────────────────────────────────────────────────────────────────────────────
# UI: 초기 개량 결과 표시
# ─────────────────────────────────────────────────────────────────────────────
with st.chat_message("assistant"):
    st.header("Multi-Agent 개량 레시피")

    initial_result = st.session_state.get("refined_result_initial")
    if initial_result:
        _render_refinement_result(initial_result, base_key="initial")
    else:
        # 세션 재진입 시 latest_recipe_json 으로 표시
        display_recipe = st.session_state.get("latest_recipe_json", recipe)
        score0, dfs0 = _json_to_tables(display_recipe)
        if dfs0:
            _render_json_tables_block(display_recipe, base_name="prediction_tables__initial",
                                      expanded=True, title="📦 레시피 테이블")
        else:
            st.markdown(display_recipe)

    # Meta / References
    meta_show = st.session_state.get("latest_meta")
    if meta_show:
        with st.expander("🧭 Meta — Key Contributions & References", expanded=False):
            kc = meta_show.get("key_contributions", {})
            st.markdown("**Key Contributions**")
            st.write(
                f"- Deposited material: {kc.get('Deposited material','')}\n"
                f"- Key Method: {kc.get('Key Deposition/Process Method','')}\n"
                f"- Target Mobility: {kc.get('Target Mobility','')}\n"
                f"- Target Stability: {kc.get('Target Stability','')}\n"
                f"- Other Constraints: {kc.get('Other Constraints','')}"
            )
            st.markdown("---")
            st.write(f"**Model**: {meta_show.get('model','')}  |  **RAG top-k**: {meta_show.get('rag_topk','')}  |  **Time**: {meta_show.get('timestamp','')}")
            refs_list = meta_show.get("references", [])
            if refs_list:
                st.markdown("**References**")
                for i, r in enumerate(refs_list, 1):
                    if isinstance(r, dict):
                        link = r.get("s2_url")
                        contrib = r.get("contribution", "")
                        recipe_txt = r.get("recipe", "")
                        with st.expander(f"Reference {i}", expanded=False):
                            if link:
                                st.markdown(f"[Semantic Scholar]({link})")
                            else:
                                st.markdown("_Semantic Scholar link: N/A_")
                            st.markdown("**Contribution**")
                            st.write(contrib if contrib else "")
                            st.markdown("**Recipe**")
                            st.write(recipe_txt if recipe_txt else "")
                    else:
                        with st.expander(f"Reference {i}", expanded=False):
                            st.markdown(str(r))

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 채팅 인터페이스  ── QA 또는 Revise (revise는 항상 multi-agent 통과)
# ─────────────────────────────────────────────────────────────────────────────
hist_container = st.container()

st.subheader("Q&A / Feedback")
with st.form("qa_feedback_form", clear_on_submit=True):
    prompt = st.text_area(
        "질문 또는 레시피 개량 요청",
        placeholder="예: 이 레시피에서 PDA 온도를 250°C 이하로 제한하면 어떤 조건이 바뀌어야 하나요?",
        height=110,
    )
    uploaded_chat_files = st.file_uploader(
        "참고 파일 첨부 (PDF, Excel, CSV)",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="첨부파일 내용은 답변 또는 레시피 개량 컨텍스트로 함께 사용됩니다.",
    )
    submit_prompt = st.form_submit_button("Send", use_container_width=True)

recipe = st.session_state.get("latest_recipe_json", recipe)
just_submitted = False

if submit_prompt and (prompt.strip() or uploaded_chat_files):
    just_submitted = True
    attachment_context, attachment_names, attachment_warnings = _build_attachment_context(uploaded_chat_files)
    display_prompt = prompt.strip() or "첨부파일에 대해 답변해주세요."
    qa_kb_context = _build_qa_kb_context(display_prompt)
    prompt_for_model = _compose_user_prompt_with_attachments(prompt, attachment_context, qa_kb_context)
    if attachment_names:
        display_prompt += "\n\n첨부파일: " + ", ".join(attachment_names)

    with st.chat_message("user"):
        st.markdown(display_prompt)
        for warning in attachment_warnings:
            st.warning(warning)
        if qa_kb_context:
            st.caption("Specialist KB 검색 결과를 답변 컨텍스트에 포함했습니다.")
        if not prompt_for_model:
            st.error("질문을 입력하거나 읽을 수 있는 파일을 첨부해주세요.")
            st.stop()
    st.session_state.messages.append({"role": "user", "content": display_prompt})

    intent = classify_feedback_intent(model, recipe, prompt_for_model)

    if intent == "qa":
        # ── QA 경로: 빠른 답변 ───────────────────────────────────────────────
        with st.spinner("답변 생성 중…"):
            asst_text = ask_with_previous_recipe(
                st.session_state.messages, model, recipe, prompt_for_model,
            )
        prose, json_part = _split_prose_and_json(asst_text)
        with st.chat_message("assistant"):
            if prose:
                st.markdown(prose)
            if json_part:
                st.session_state.latest_recipe_json = json_part
                _render_json_tables_block(
                    json_part,
                    base_name=f"chat_qa__{len(st.session_state.messages)}",
                    expanded=True, title="📦 답변 테이블",
                )

    else:
        # ── Revise 경로: RAG 재생성 → multi-agent 자동 통과 ─────────────────
        with st.chat_message("assistant"):
            st.markdown("**레시피 개량 시작:** RAG 재검색 후 Multi-Agent 검토를 진행합니다.")

            original_contribution = st.session_state.get("original_contribution", "")
            current_contribution = st.session_state.get("current_contribution", original_contribution)
            fixed_examples = st.session_state.get("fixed_examples", [])

            with st.status("🔄 레시피 재생성 중…", expanded=True) as _revise_status:
                st.write("🤖 피드백 반영해 기여문 수정 중…")
                revised_contribution = revise_contribution_with_feedback(
                    model_name=model,
                    original_contribution=original_contribution,
                    current_contribution=current_contribution,
                    current_recipe=recipe,
                    user_feedback=prompt_for_model,
                )
                st.write("✅ 기여문 수정 완료")
                st.write(f"🔢 임베딩 재계산 & FAISS 재검색 중…")
                new_raw_recipe, new_retrieval_examples, new_refs, revised_contribution = regenerate_recipe_with_feedback(
                    model_name=model,
                    revised_contribution=revised_contribution,
                    current_recipe=recipe,
                    fixed_examples=fixed_examples,
                    top_k=top_k,
                )
                st.write("✅ 레시피 초안 재생성 완료")
                _revise_status.update(label="✅ 레시피 재생성 완료", state="complete", expanded=False)

            st.session_state.current_contribution = revised_contribution
            st.session_state.retrieval_examples = new_retrieval_examples
            st.session_state.references = new_refs
            st.session_state.latest_meta = _build_meta(
                material_name, synthesis_technique,
                mobility_target, stability_target, other_contstraints,
                model, top_k, references=new_refs or [],
            )

            # Multi-agent 실행 (실시간 st.status)
            turn_key = f"chat_{len(st.session_state.messages)}"
            ma_result = _run_refinement_with_status(new_raw_recipe, label_prefix="개량 레시피 ")

            refined_str = ma_result.get("refined_recipe", "") if ma_result else ""
            final_recipe = refined_str if refined_str else new_raw_recipe

            st.session_state.current_recipe = final_recipe
            st.session_state.latest_recipe_json = final_recipe
            recipe = final_recipe

            if ma_result:
                _render_refinement_result(ma_result, base_key=turn_key)
            else:
                score_c, dfs_c = _json_to_tables(final_recipe)
                if dfs_c:
                    _render_json_tables_block(final_recipe,
                                              base_name=f"chat_revise__{turn_key}",
                                              expanded=True, title="📦 개량 레시피")
                else:
                    st.markdown(final_recipe)

            asst_text = final_recipe
            st.session_state.messages.append({"role": "assistant", "content": asst_text})

# ─── 히스토리 출력 ────────────────────────────────────────────────────────────
with hist_container:
    if just_submitted:
        hist = st.session_state.messages[2:-2] if len(st.session_state.messages) > 4 else []
    else:
        hist = st.session_state.messages[2:] if len(st.session_state.messages) > 2 else []

    for idx, message in enumerate(hist, start=2):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                prose, json_part = _split_prose_and_json(message["content"])
                if prose:
                    st.markdown(prose)
                if json_part:
                    _render_json_tables_block(
                        json_part, expanded=False,
                        title="📦 이전 개량 레시피",
                    )
            else:
                st.markdown(message["content"])
