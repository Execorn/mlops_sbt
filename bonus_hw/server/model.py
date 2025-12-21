import triton_python_backend_utils as pb_utils
import numpy as np
import faiss
import os
import json


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(model_dir, "faiss.index")

        if not os.path.exists(index_path):
            raise pb_utils.TritonModelException(
                f"Index not found at {index_path}")

        print(f"[Server] Loading FAISS index from {index_path}", flush=True)
        self.index = faiss.read_index(index_path)
        print(
            f"[Server] Index loaded. Total vectors: {self.index.ntotal}", flush=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(
                    request, "QUERY_VECTOR")
                k_tensor = pb_utils.get_input_tensor_by_name(request, "TOP_K")

                query_vec = in_tensor.as_numpy()

                k = 5
                if k_tensor is not None:
                    k = int(k_tensor.as_numpy().flatten()[0])
                distances, ids = self.index.search(query_vec, k)

                out_ids = pb_utils.Tensor("NEIGHBOR_IDS", ids.astype(np.int64))
                out_dists = pb_utils.Tensor(
                    "DISTANCES", distances.astype(np.float32))

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[out_ids, out_dists]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Error: {e}")))
        return responses
