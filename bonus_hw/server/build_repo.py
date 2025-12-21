import os
import numpy as np
import faiss
import shutil
from transformers import AutoTokenizer, AutoModel

REPO_PATH = "/models"
MODEL_NAME = "faiss_search"
VERSION = "1"
DIM = 384
NUM_OBJECTS = 100_000


def create_structure():
    path = os.path.join(REPO_PATH, MODEL_NAME, VERSION)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def create_config(base_path):
    # once again it just doesn't work for me without KIND_GPU
    config = f"""name: "{MODEL_NAME}"
backend: "python"
max_batch_size: 32
input [
  {{ name: "QUERY_VECTOR", data_type: TYPE_FP32, dims: [ {DIM} ] }},
  {{ name: "TOP_K", data_type: TYPE_INT32, dims: [ 1 ], optional: true }}
]
output [
  {{ name: "NEIGHBOR_IDS", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "DISTANCES", data_type: TYPE_FP32, dims: [ -1 ] }}
]
instance_group [ {{ kind: KIND_CPU }} ]
"""
    with open(os.path.join(REPO_PATH, MODEL_NAME, "config.pbtxt"), "w") as f:
        f.write(config)


def generate_data(version_path):
    print("Caching model")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    AutoTokenizer.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)

    print(f"Generating {NUM_OBJECTS} embeddings")
    embeddings = np.random.rand(NUM_OBJECTS, DIM).astype("float32")
    faiss.normalize_L2(embeddings)

    print("Building Index")
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    index_path = os.path.join(version_path, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    v_path = create_structure()
    create_config(v_path)
    generate_data(v_path)

    shutil.copy("model.py", os.path.join(v_path, "model.py"))
