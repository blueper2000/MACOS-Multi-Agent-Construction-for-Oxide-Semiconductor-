import os
from pathlib import Path
from cloud_runtime import safe_batch_completion as batch_completion, safe_completion as completion
import openai
import fire
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from pprint import pprint
import jsonlines
from tqdm import tqdm
from . import openai_utils
import numpy as np
import pandas as pd


def _normalize_embedding(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _load_local_retrieval_dataset(parquet_path: str) -> Dataset:
    """Load a local retrieval parquet as a HF Dataset for FAISS search."""
    df = pd.read_parquet(parquet_path).copy()
    if "contributions_embedding" not in df.columns:
        raise ValueError(f"'contributions_embedding' column not found in {parquet_path}")

    if "contribution_clean" in df.columns:
        df["contribution"] = df["contribution_clean"].fillna(df.get("contribution", ""))
    elif "contribution" not in df.columns:
        raise ValueError(f"'contribution' column not found in {parquet_path}")

    if "recipe" not in df.columns:
        raise ValueError(f"'recipe' column not found in {parquet_path}")

    if "id" not in df.columns:
        df["id"] = df.apply(
            lambda row: "::".join(
                str(part)
                for part in (
                    row.get("paper_id", ""),
                    row.get("sample_label", ""),
                    row.get("pdf_file", ""),
                )
                if str(part)
            ),
            axis=1,
        )

    if "classification_result" not in df.columns:
        df["classification_result"] = df.get("paper_id", "local-parquet")

    if "pdf_file" not in df.columns:
        df["pdf_file"] = ""

    df = df[df["contributions_embedding"].notna()].copy()
    df["contributions_embedding"] = df["contributions_embedding"].apply(_normalize_embedding)

    retrieval_df = df[
        [
            "id",
            "classification_result",
            "recipe",
            "contribution",
            "contributions_embedding",
            "pdf_file",
        ]
    ].reset_index(drop=True)
    return Dataset.from_pandas(retrieval_df, preserve_index=False)


# Shared app and batch prompt paths, independent of the working directory.
_REPOSITORY_DIR = Path(__file__).resolve().parents[1]
_PROMPT_DIR = _REPOSITORY_DIR / "data_collection"
_DEFAULT_PREDICTION_PROMPT = str(_PROMPT_DIR / "_runtime_sd_plus_prediction_prompt.txt")
_DEFAULT_RAG_PROMPT = str(_PROMPT_DIR / "_runtime_sd_plus_rag_prompt.txt")


def _resolve_prompt_path(prompt_name):
    if prompt_name == "prediction":
        return _DEFAULT_PREDICTION_PROMPT
    if prompt_name == "rag":
        return _DEFAULT_RAG_PROMPT
    path = Path(prompt_name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Custom prompt file not found: {path}")
    return str(path)


class RecipePredictor:
        

    def __init__(self, 
                 model="gpt-4o-mini",
                 batch_size=1,
                 max_tokens=4096,
                 max_completion_tokens=16384,
                 temperature=0.0,
                 api_key=None,
                 prompt_filename: str=_DEFAULT_PREDICTION_PROMPT,
                 ):
        self.model = model
        self.prompt_filename = prompt_filename
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.prediction_prompt = Path(prompt_filename).read_text(encoding="utf-8")
        self.job_description = f"material prediction job w/ {model}"
        self.max_completion_tokens = max_completion_tokens  

    def _get_model_config(self, model_name):
        """모델별 설정을 반환하는 통합 함수"""
        if model_name.startswith("gpt-5") or model_name.startswith("o4"):
            # GPT-5 모델들
            if model_name.startswith("gpt-5"):
                base_model = "gpt-5" if model_name != "gpt-5-mini" else "gpt-5-mini"
            else:  # o4 모델들
                base_model = "o4-mini"
            
            reasoning_effort = None
            if "high" in model_name:
                reasoning_effort = "high"
            elif "low" in model_name:
                reasoning_effort = "low"
            
            return base_model, reasoning_effort, True  # max_completion_tokens 사용
            
        elif model_name.startswith("o1") or model_name.startswith("o3"):
            base_model = model_name.replace("-high", "").replace("-low", "")
            reasoning_effort = "high" if "high" in model_name else "low" if "low" in model_name else None
            return base_model, reasoning_effort, True
            
        else:
            # 기존 GPT 모델들
            return model_name, None, False

    def build_prompt(self, item):
        prompt = self.prediction_prompt.format(
            contributions=item["contribution"],
            recipe=item.get("recipe", ""),
            references="",
            examples="",
        )
        return [
            {
                "content": prompt,
                "role": "user"
            }
        ]

    def predict_batch(self, prompts):
        completions = []
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        # 배치 처리 설정
        if use_completion_tokens:
            batch_completion_kwargs = {"max_completion_tokens": self.max_completion_tokens}
            if reasoning_effort:
                batch_completion_kwargs["reasoning_effort"] = reasoning_effort
        else:
            batch_completion_kwargs = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        
        messages = batch_completion(
            model=base_model,
            messages=prompts,
            api_key=self.api_key,
            **batch_completion_kwargs
        )
        contents = [message['choices'][0]['message']['content'] for message in messages]
        completions.extend(contents)
        return completions


    def predict_single(self, prompts):
        completions = []
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        for prompt in prompts:
            if use_completion_tokens:
                kwargs = {
                    "model": base_model,
                    "messages": prompt,
                    "max_completion_tokens": self.max_completion_tokens
                }
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs = {
                    "model": base_model,
                    "messages": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            
            message = completion(api_key=self.api_key, **kwargs)
            completions.append(message['choices'][0]['message']['content'])
        
        return completions
    
    def predict_batch_openai(self, prompts):
        completions = [None] * len(prompts)
        id_list = [f"request-{i}" for i in range(len(prompts))]
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        # OpenAI 배치 처리 설정
        if use_completion_tokens:
            body_kwargs = {"max_completion_tokens": self.max_completion_tokens}
            if reasoning_effort:
                body_kwargs["reasoning_effort"] = reasoning_effort
        else:
            body_kwargs = {"max_tokens": self.max_tokens, "temperature": self.temperature}

        response_dict = openai_utils.process_batch(
            self.client, 
            base_model, 
            id_list, 
            prompts, 
            job_description=self.job_description, 
            body_kwargs=body_kwargs
        )
        
        for id, response in response_dict.items():
            completions[int(id.split("-")[1])] = response
        
        return completions
    
    def predict(self, dataset, batch_size=1, use_openai_batch=False, completed_batch_id=None):
        if completed_batch_id:
            response_dict = openai_utils.retrieve_batch_responses(self.client, completed_batch_id)
            dataset_dict = {item["id"]: item for item in dataset}
            for id, response in response_dict.items():
                index = int(id.split("-")[1])
                item = dataset_dict[index]
                yield item, response

            return
        
        if use_openai_batch:
            assert batch_size > 1, "Batch size must be greater than 1 for OpenAI batch completion"
            predict_func = self.predict_batch_openai
        else:
            predict_func = self.predict_batch if batch_size > 1 else self.predict_single
        
        # batch iteration
        batch = []
        for i, item in enumerate(dataset):
            prompt = self.build_prompt(item)
            if prompt is None:
                continue
            
            batch.append(prompt)
            if len(batch) == batch_size:
                predictions = predict_func(batch)
                for j, prediction in enumerate(predictions):
                    yield dataset[i - batch_size + j + 1], prediction
                batch = []


        if batch:
            predictions = predict_func(batch)
            for j, prediction in enumerate(predictions):
                yield dataset[i - len(batch) + j + 1], prediction
            batch = []


class RAGRecipePredictor(RecipePredictor):


    def __init__(self, model="gpt-4o-mini", batch_size=1, max_tokens=4096, max_completion_tokens=16384, temperature=0, api_key=None, prompt_filename = _DEFAULT_RAG_PROMPT,
                 rag_topk: int = 5, retrieval_split: str = "all", retrieval_parquet_path: str | None = None,
                 hf_token: str | None = None, dataset_id: str = "blueper/oxidesc-recipe-ver2"):
        super().__init__(model, batch_size, max_tokens, max_completion_tokens, temperature, api_key, prompt_filename)
        self.job_description = f"RAG material prediction job w/ {model}"

        from cloud_runtime import load_hf_retrieval, validate_retrieval_dataset

        if retrieval_parquet_path:
            self.retrieval_set = _load_local_retrieval_dataset(retrieval_parquet_path)
        else:
            self.retrieval_set = load_hf_retrieval(dataset_id, hf_token, retrieval_split)
        validate_retrieval_dataset(self.retrieval_set)
        self.rag_topk = rag_topk
        # Per-predictor in-memory index: no shared FAISS or private dataset cache.
        self.retrieval_set.add_faiss_index("contributions_embedding")

        self.base_references = None

    def search(self, contribution, k=5, return_rows=False):
        if k <= 0:
            return None if return_rows else ""
        query_embedding = np.asarray(contribution, dtype=np.float32)
        if query_embedding.ndim != 1 or not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding must be a finite vector.")
        if len(query_embedding) != len(self.retrieval_set[0]["contributions_embedding"]):
            raise ValueError("Dataset and query embedding dimensions differ; use text-embedding-3-large with default dimensions.")
        scores, results = self.retrieval_set.get_nearest_examples(
            "contributions_embedding", query_embedding, min(k, len(self.retrieval_set)))
        if return_rows:
            return results
        else:
            retrieval_prompts = []
            offset = len(self.base_references or [])
            for i, (contribution, recipe) in enumerate(zip(results["contribution"], results['recipe'])):
                retrieval_prompts.append(f"# Ref{offset + i + 1}:\n{contribution}\n\n{recipe}")
            retrieval_prompts = "\n\n".join(retrieval_prompts)
            return retrieval_prompts

    def build_prompt(self, item):
        contributions, recipe, embeddings = item["contribution"], item["recipe"], item["contributions_embedding"]
        if self.rag_topk > 0:
            retrieval_prompts = self.search(embeddings, k=self.rag_topk)
        else:
            retrieval_prompts = ""

        if self.base_references:
            references = []
            for i, ref in enumerate(self.base_references):
                if isinstance(ref, dict):
                    ref_body = ref.get("recipe") or ref.get("contribution") or ""
                else:
                    ref_body = str(ref)
                references.append(f"# Ref{i + 1}:\n{ref_body}")
            retrieval_prompts = "\n\n".join(references) + "\n\n" + retrieval_prompts

        prompt = self.prediction_prompt.format(
            contributions=contributions,
            references=retrieval_prompts,
            examples=retrieval_prompts,
        )
        return [
            {
                "content": prompt,
                "role": "user"
            }
        ]

class ReferenceRecipePredictor(RecipePredictor):
    """PDF/current-recipe context with no retrieval dataset or embedding calls."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("prompt_filename", _DEFAULT_RAG_PROMPT)
        super().__init__(*args, **kwargs)
        self.rag_topk = 0
        self.base_references = []

    build_prompt = RAGRecipePredictor.build_prompt



def main(
        model: str = "gpt-4o-mini",
        prompt_name: str = "prediction",
        batch_size: int = 1,
        use_openai_batch: bool = False,
        use_rag: bool = False,
        top_k: int = 5,
        split: str = "test_high_impact",
):
    ds = load_from_disk("data/omg", split=split)

    model_name = model.split("/", 1)[-1]

    if use_rag:
        if prompt_name == "prediction":
            prompt_name = "rag"
            print("Using RAG prompt instead")
        
        prompt_filename = _resolve_prompt_path(prompt_name)
        predictor = RAGRecipePredictor(model=model, prompt_filename=prompt_filename, rag_topk=top_k)
        output_filename = f"data/{split}/{model_name}/{prompt_name}__k{top_k}.jsonl"

    else:
        prompt_filename = _resolve_prompt_path(prompt_name)
        predictor = RecipePredictor(model=model, prompt_filename=prompt_filename)
        output_filename = f"data/{split}/{model_name}/{prompt_name}.jsonl"


    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        ds = ds.select(range(skip, len(ds)))
        print(f"Skipping {skip} items")

    with jsonlines.open(output_filename, "a") as fout:
        for item, prediction in tqdm(predictor.predict(ds, batch_size=batch_size, use_openai_batch=use_openai_batch), total=len(ds)):
            item["prediction"] = prediction
            if 'contributions_embedding' in item:
                del item['contributions_embedding']
            fout.write(item)

    return output_filename

if __name__ == "__main__":
    fire.Fire(main)
