"""Offline regression checks. All records, credentials and responses are synthetic."""
import ast
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from cloud_runtime import load_hf_retrieval, safe_completion, validate_retrieval_dataset
from datasets import Dataset
from experiment.predict import RecipePredictor, ReferenceRecipePredictor, RAGRecipePredictor

RECIPE = json.dumps({
    "meta": {},
    "tables": {"1. Materials": [{"Variable": "Channel material name",
        "Predicted Value": "IZO", "Rationale": "Synthetic test", "Ref IDs": "User Input"}]},
    "scorecard": {"confidence": 1, "similar_refs": 0, "why": "Synthetic", "risks": ["Test", "Test"]},
})


def response(**kwargs):
    from litellm import ModelResponse
    return ModelResponse(choices=[{"message": {"role": "assistant", "content": RECIPE}}])


def records():
    return [{"id": "synthetic-1", "contribution": "Synthetic ALD IZO",
             "recipe": "Fictional recipe", "contributions_embedding": [1.0, 0.0]}]


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class PredictorTests(unittest.TestCase):
    def test_no_dataset_for_reference_predictor(self):
        with patch("datasets.load_dataset", side_effect=AssertionError("Unexpected retrieval")):
            predictor = ReferenceRecipePredictor(api_key="synthetic-api-a")
            predictor.base_references = [{"recipe": "Uploaded synthetic paper"}]
            prompt = predictor.build_prompt({"contribution": "Target", "recipe": "", "contributions_embedding": None})
        self.assertIn("Uploaded synthetic paper", prompt[0]["content"])
        self.assertIn("Ref1", prompt[0]["content"])

    def test_distinct_predictor_keys_and_all_batches(self):
        first = RecipePredictor(model="gpt-4o", api_key="synthetic-api-a")
        second = RecipePredictor(model="gpt-4o", api_key="synthetic-api-b")
        items = [{"contribution": f"Target {i}"} for i in range(3)]
        with patch("litellm.completion", side_effect=response) as api:
            self.assertEqual(len(list(first.predict(items))), 3)
            second.predict_single([[{"role": "user", "content": "Second user"}]])
        self.assertEqual([c.kwargs["api_key"] for c in api.call_args_list],
                         ["synthetic-api-a"] * 3 + ["synthetic-api-b"])

    def test_batched_requests_use_explicit_key(self):
        predictor = RecipePredictor(model="gpt-4o", api_key="synthetic-api-a")
        with patch("litellm.batch_completion", return_value=[response(), response()]) as api:
            output = predictor.predict_batch([[{"role": "user", "content": "A"}], [{"role": "user", "content": "B"}]])
        self.assertEqual(len(output), 2)
        self.assertEqual(api.call_args.kwargs["api_key"], "synthetic-api-a")

    def test_hf_token_explicit_and_no_records_cache(self):
        with patch("datasets.load_dataset", return_value={"train": records()}) as loader:
            dataset = load_hf_retrieval("synthetic/repo", "synthetic-hf-a", "train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(loader.call_args.kwargs, {"token": "synthetic-hf-a", "streaming": True})
        with patch("datasets.load_dataset") as loader:
            with self.assertRaises(ValueError):
                load_hf_retrieval("synthetic/repo", "", "train")
            loader.assert_not_called()

    def test_other_users_cannot_reuse_authorized_records(self):
        def authorized_only(*args, **kwargs):
            if kwargs["token"] != "synthetic-authorized-user":
                raise PermissionError("Synthetic access denied")
            return {"train": records()}
        with patch("datasets.load_dataset", side_effect=authorized_only) as loader:
            self.assertEqual(len(load_hf_retrieval("synthetic/repo", "synthetic-authorized-user")), 1)
            with self.assertRaises(PermissionError):
                load_hf_retrieval("synthetic/repo", "synthetic-unauthorized-user")
        self.assertEqual(loader.call_count, 2)

    def test_rag_index_search_reference_ids_and_dimensions(self):
        with patch("cloud_runtime.load_hf_retrieval", return_value=Dataset.from_list(records())):
            predictor = RAGRecipePredictor(api_key="synthetic-api-a", hf_token="synthetic-hf-a")
        predictor.base_references = [{"recipe": "Synthetic uploaded PDF"}]
        prompt = predictor.build_prompt({"contribution": "Target", "recipe": "", "contributions_embedding": [1.0, 0.0]})[0]["content"]
        self.assertIn("# Ref1:", prompt)
        self.assertIn("# Ref2:", prompt)
        self.assertEqual(predictor.search([1.0, 0.0], k=10, return_rows=True)["id"], ["synthetic-1"])
        with self.assertRaises(ValueError):
            predictor.search([1.0, 0.0, 0.0])

    def test_embedding_validation(self):
        with self.assertRaises(ValueError):
            validate_retrieval_dataset([{"contributions_embedding": [float("nan")]}])
        with self.assertRaises(ValueError):
            validate_retrieval_dataset([{"contributions_embedding": [1.0]}, {"contributions_embedding": [1.0, 2.0]}])

    def test_provider_error_never_echoes_secret(self):
        with patch("litellm.completion", side_effect=ValueError("synthetic-secret-do-not-show")):
            with self.assertRaises(RuntimeError) as context:
                safe_completion(api_key="synthetic-secret-do-not-show")
        self.assertNotIn("synthetic-secret", str(context.exception))

    def test_pdf_extraction_forwards_key(self):
        import fitz
        import pdf2recipe
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Synthetic oxide semiconductor recipe test. " * 4)
        data = doc.tobytes()
        doc.close()
        with patch("litellm.batch_completion", return_value=[response()]) as api:
            result = pdf2recipe.pdf_bytelist_to_recipes([data], api_key="synthetic-api-a")
        self.assertEqual(len(result), 1)
        self.assertEqual(api.call_args.kwargs["api_key"], "synthetic-api-a")

    def test_pdf_generation_and_feedback_without_embedding_or_dataset(self):
        # Execute the real core flow without a browser/file-uploader fixture.
        tree = ast.parse((ROOT / "app_multiagent.py").read_text(encoding="utf-8"))
        names = {"predict_recipe", "regenerate_recipe_with_feedback"}
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        state = State()
        reference = ReferenceRecipePredictor(model="gpt-4o", api_key="synthetic-api-a")
        ns = {"st": SimpleNamespace(session_state=state), "rag_predictor": reference,
              "base_predictor": RecipePredictor(api_key="synthetic-api-a"),
              "get_embedding": Mock(side_effect=AssertionError("No PDF embedding required")),
              "PREDICTION_PROMPT": "Target {material_name} {deposition_method} {mobility_target} {stability_target}",
              "_build_initial_pdf_context": lambda files: ("Synthetic text", ["test.pdf"], []),
              "pdf_bytelist_to_recipes": Mock(return_value=["Synthetic PDF recipe"]),
              "_normalize_references": lambda refs: [{"recipe": ref} for ref in refs or []],
              "_assign_ref_ids": lambda refs: refs, "_build_meta": lambda *args, **kwargs: {},
              "openai_api_key": "synthetic-api-a"}
        exec(compile(functions, "app_core", "exec"), ns)
        upload = SimpleNamespace(getvalue=lambda: b"synthetic-pdf")
        with patch("litellm.completion", side_effect=response):
            result = ns["predict_recipe"]("IZO", "ALD", "Target", "Target", "", 0, "gpt-4o", False, [upload])
            revised = ns["regenerate_recipe_with_feedback"]("gpt-4o", "Revised target", result[0], state.fixed_examples, 0)
        self.assertEqual(result[0], RECIPE)
        self.assertEqual(revised[0], RECIPE)
        ns["get_embedding"].assert_not_called()
        self.assertEqual(ns["pdf_bytelist_to_recipes"].call_args.kwargs["api_key"], "synthetic-api-a")


class StreamlitTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "app_multiagent.py"), default_timeout=60)

    @staticmethod
    def button(app, label):
        return next(button for button in app.button if button.label == label)

    def connect(self, app, key="synthetic-api-a", token=""):
        app.text_input(key="openai_key_input").set_value(key)
        app.text_input(key="hf_token_input").set_value(token)
        self.button(app, "Apply credentials").click().run()
        self.assertFalse(app.exception)

    def test_start_without_dataset_or_credentials(self):
        with patch("datasets.load_dataset", side_effect=AssertionError("No startup retrieval")), patch("litellm.completion") as api:
            app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("OpenAI API key" in item.value for item in app.info))
        api.assert_not_called()

    def test_llm_generation_reviews_new_request_and_disconnect(self):
        with patch("datasets.load_dataset", side_effect=AssertionError("No LLM retrieval")), patch("litellm.embedding", side_effect=AssertionError("No LLM embedding")), patch("litellm.completion", side_effect=response) as api:
            app = self.app().run()
            self.connect(app)
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertIn("IZO", app.session_state["latest_recipe_json"])
            self.assertGreaterEqual(api.call_count, 4)  # initial + two reviewers + supervisor
            count = api.call_count
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertGreaterEqual(api.call_count - count, 4)
            self.assertTrue(all(call.kwargs["api_key"] == "synthetic-api-a" for call in api.call_args_list))
            self.button(app, "Disconnect and clear session").click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("active_openai_key", app.session_state)
            self.assertNotIn("predictors", app.session_state)

    def test_missing_pdf_and_rag_access_do_not_fall_back(self):
        with patch("litellm.completion") as api:
            app = self.app().run()
            self.connect(app)
            app.selectbox(key="generation_mode").select("PDF").run()
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Upload at least one PDF" in item.value for item in app.warning))
            app.selectbox(key="generation_mode").select("RAG").run()
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            api.assert_not_called()

    def test_separate_users_and_key_change_reset(self):
        with patch("litellm.completion", side_effect=response):
            first, second = self.app().run(), self.app().run()
            self.connect(first, "synthetic-api-a", "synthetic-hf-a")
            self.connect(second, "synthetic-api-b", "synthetic-hf-b")
            self.button(first, "Recommend").click().run()
            self.button(second, "Recommend").click().run()
            self.assertIsNot(first.session_state["predictors"][1], second.session_state["predictors"][1])
            self.assertEqual(first.session_state["predictors"][1].api_key, "synthetic-api-a")
            self.assertEqual(second.session_state["predictors"][1].api_key, "synthetic-api-b")
            self.connect(first, "synthetic-api-c")
            self.assertNotIn("predictors", first.session_state)
            self.assertNotIn("latest_recipe_json", first.session_state)
            self.assertEqual(second.session_state["active_openai_key"], "synthetic-api-b")

    def test_authorized_rag_end_to_end(self):
        with patch("cloud_runtime.load_hf_retrieval", side_effect=lambda *args: Dataset.from_list(records())) as loader, patch("litellm.embedding", return_value={"data": [{"embedding": [1.0, 0.0]}]}) as embed, patch("litellm.completion", side_effect=response) as api:
            app = self.app().run()
            self.connect(app, "synthetic-api-a", "synthetic-hf-a")
            app.selectbox(key="generation_mode").select("RAG").run()
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["retrieval_examples"]), 1)
            self.assertEqual(loader.call_args.args[1], "synthetic-hf-a")
            self.assertTrue(all(c.kwargs["api_key"] == "synthetic-api-a" for c in embed.call_args_list))
            self.assertGreaterEqual(api.call_count, 4)

    def test_generation_failure_is_sanitized_and_retryable(self):
        with patch("litellm.completion", side_effect=ValueError("synthetic-secret-do-not-show")):
            app = self.app().run()
            self.connect(app)
            self.button(app, "Recommend").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn("synthetic-secret", app.error[0].value)
        self.assertNotIn("messages", app.session_state)
        with patch("litellm.completion", side_effect=response):
            self.button(app, "Recommend").click().run()
        self.assertFalse(app.exception)
        self.assertIn("latest_recipe_json", app.session_state)

    def managed_app(self):
        app = self.app()
        app.secrets["OPENAI_API_KEY"] = "synthetic-operator-api"
        app.secrets["HF_TOKEN"] = "synthetic-operator-hf"
        app.secrets["HF_DATASET_ID"] = "synthetic/operator-corpus"
        app.secrets["HF_DATASET_SPLIT"] = "train"
        return app.run()

    def test_operator_secrets_default_without_exposing_keys(self):
        with patch("litellm.completion", side_effect=response) as api:
            app = self.managed_app()
            self.assertFalse(app.exception)
            self.assertTrue(app.checkbox(key="use_managed_credentials").value)
            self.assertFalse(any(item.key == "openai_key_input" for item in app.text_input))
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertGreaterEqual(api.call_count, 4)
            self.assertTrue(all(c.kwargs["api_key"] == "synthetic-operator-api" for c in api.call_args_list))
            self.assertNotIn("synthetic-operator-api", str(app.markdown))
            self.assertNotIn("synthetic-operator-hf", str(app.info))

    def test_operator_rag_source_is_fixed_and_personal_override_is_separate(self):
        with patch("cloud_runtime.load_hf_retrieval", side_effect=lambda *args: Dataset.from_list(records())) as loader, patch("litellm.embedding", return_value={"data": [{"embedding": [1.0, 0.0]}]}), patch("litellm.completion", side_effect=response) as api:
            app = self.managed_app()
            app.selectbox(key="generation_mode").select("RAG").run()
            self.assertFalse(any(item.label == "Hugging Face dataset" for item in app.text_input))
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(loader.call_args.args, ("synthetic/operator-corpus", "synthetic-operator-hf", "train"))
            app.checkbox(key="use_managed_credentials").uncheck().run()
            self.assertNotIn("predictors", app.session_state)
            self.assertNotIn("latest_recipe_json", app.session_state)
            self.connect(app, "synthetic-personal-api", "")
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(api.call_args.kwargs["api_key"], "synthetic-personal-api")
            # No fallback to the owner's HF token in personal mode.
            app.selectbox(key="generation_mode").select("RAG").run()
            self.assertTrue(app.error)
            self.assertEqual(loader.call_count, 1)

    def test_operator_disconnect_and_reconnect(self):
        with patch("litellm.completion", side_effect=response):
            app = self.managed_app()
            self.button(app, "Recommend").click().run()
            self.button(app, "Disconnect and clear session").click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("predictors", app.session_state)
            self.assertTrue(app.session_state["managed_disconnected"])
            self.button(app, "Recommend").click().run()
            self.assertNotIn("predictors", app.session_state)
            self.button(app, "Reconnect app-managed connection").click().run()
            self.button(app, "Recommend").click().run()
            self.assertFalse(app.exception)
            self.assertIn("latest_recipe_json", app.session_state)


if __name__ == "__main__":
    unittest.main()
