import os
import shutil
import argparse
import numpy as np
import sys
import time

REPO_DIR = "model_repository"

def create_config(path, content):
    with open(path, "w") as f:
        f.write(content)

def step_3_setup():
    print("\n[Step 3] triton repo + GPU support")

    dirs = [
        f"{REPO_DIR}/resnet_onnx/1",
        f"{REPO_DIR}/preprocess/1",
        f"{REPO_DIR}/ensemble/1",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    onnx_src = "onnx_output/model.onnx"
    onnx_dst = f"{REPO_DIR}/resnet_onnx/1/model.onnx"
    if os.path.exists(onnx_src):
        shutil.copy(onnx_src, onnx_dst)
        print(f"moved model.onnx to {onnx_dst}")
    else:
        print(f"{onnx_src} not found, run step 1")
        sys.exit(1)

    # didn't work on my hybrid amd+nvidia laptop without 'kind: KIND_GPU'
    create_config(
        f"{REPO_DIR}/resnet_onnx/config.pbtxt",
        """name: "resnet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [ { name: "INPUT_IMAGE" data_type: TYPE_FP32 dims: [-1, 3, 224, 224] } ]
output [ { name: "OUTPUT_PROJECTION" data_type: TYPE_FP32 dims: [-1, 32] } ]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
""",
    )

    create_config(
        f"{REPO_DIR}/preprocess/config.pbtxt",
        """name: "preprocess"
backend: "python"
max_batch_size: 0
input [ { name: "RAW_IMAGE" data_type: TYPE_FP32 dims: [-1, 224, 224, 3] } ]
output [ { name: "PREPROCESSED_IMAGE" data_type: TYPE_FP32 dims: [-1, 3, 224, 224] } ]
""",
    )

    create_config(
        f"{REPO_DIR}/preprocess/1/model.py",
        """import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            img = in_tensor.as_numpy()
            
            
            norm_img = (img - mean) / std
            out_img = norm_img.transpose(0, 3, 1, 2)
            
            out_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", out_img.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
""",
    )

    create_config(
        f"{REPO_DIR}/ensemble/config.pbtxt",
        """name: "ensemble"
platform: "ensemble"
max_batch_size: 0
input [ { name: "ENSEMBLE_INPUT" data_type: TYPE_FP32 dims: [-1, 224, 224, 3] } ]
output [ { name: "ENSEMBLE_OUTPUT" data_type: TYPE_FP32 dims: [-1, 32] } ]
ensemble_scheduling {
  step [
    {
      model_name: "preprocess"
      model_version: -1
      input_map { key: "RAW_IMAGE" value: "ENSEMBLE_INPUT" }
      output_map { key: "PREPROCESSED_IMAGE" value: "preprocessed_data" }
    },
    {
      model_name: "resnet_onnx"
      model_version: -1
      input_map { key: "INPUT_IMAGE" value: "preprocessed_data" }
      output_map { key: "OUTPUT_PROJECTION" value: "ENSEMBLE_OUTPUT" }
    }
  ]
}
""",
    )


def step_4_check():
    print("\n[Step 4] sanity check")
    import tritonclient.http as httpclient

    url = "triton_server:8000"

    try:
        client = httpclient.InferenceServerClient(url=url)
        for _ in range(15):
            if client.is_server_live():
                break
            time.sleep(1)
            print(".", end="", flush=True)

        if not client.is_server_live():
            print("\nserver is NOT live")
            return
    except Exception as e:
        print(f"\nno connection: {e}")
        return

    input_data = np.random.rand(2, 224, 224, 3).astype(np.float32)

    inputs = httpclient.InferInput("ENSEMBLE_INPUT", input_data.shape, "FP32")
    inputs.set_data_from_numpy(input_data)
    outputs = httpclient.InferRequestedOutput("ENSEMBLE_OUTPUT")

    res = client.infer(model_name="ensemble", inputs=[
                       inputs], outputs=[outputs])
    result = res.as_numpy("ENSEMBLE_OUTPUT")

    print(f"final shape: {result.shape} (what we want: [2, 32])")
    print("sanity check passed" if result.shape ==
          (2, 32) else "sanity check FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["setup", "check"], required=True)
    args = parser.parse_args()

    if args.mode == "setup":
        step_3_setup()
    elif args.mode == "check":
        step_4_check()
