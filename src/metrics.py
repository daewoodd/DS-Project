import json

with open("models/eval_results.json", "r") as f:
    EVAL_RESULTS = json.load(f)

def get_metrics(model_name: str):
    if model_name not in EVAL_RESULTS:
        raise ValueError("Model not found")

    return EVAL_RESULTS[model_name]
