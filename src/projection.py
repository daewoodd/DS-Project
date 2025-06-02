import json

def get_projection(method: str = "pca"):
    if method not in {"pca", "tsne"}:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    with open("models/projections.json", "r") as f:
        projections = json.load(f)

    return projections[method]
