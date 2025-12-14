import argparse
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForImageClassification
from thop import profile
import os

# you need to clone the resnet repo before using this (if you run it outside of docker)
# because on rolling release distros hf_transfer lib gets a segfault during load with python 3.13+
class ResNetWrapper(nn.Module):
    def __init__(self, model_name='microsoft/resnet-18'):
        super().__init__()
        
        # change to this line if you use python 3.13+
        # self.base_model = AutoModelForImageClassification.from_pretrained("./resnet-18")
        self.base_model = AutoModelForImageClassification.from_pretrained(
            model_name) 
        self.in_features = self.base_model.config.num_labels
        self.projection = nn.Linear(self.in_features, 32)

    def forward(self, input_image):
        outputs = self.base_model(pixel_values=input_image)
        logits = outputs.logits
        projected = self.projection(logits)
        return projected


def step_1_export():
    print("\n[Step 1] model setup")
    model = ResNetWrapper()
    model.eval()

    save_dir = "onnx_output"
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, "model.onnx")
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['INPUT_IMAGE'],
        output_names=['OUTPUT_PROJECTION'],
        dynamic_axes={
            'INPUT_IMAGE': {0: 'batch_size'},
            'OUTPUT_PROJECTION': {0: 'batch_size'}
        }
    )
    
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {'INPUT_IMAGE': dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    diff = np.abs(ort_out - torch_out).max()
    print(f"max diff: {diff}")

    if diff < 1e-4:
        print("diff good")
    else:
        print("diff bad")


def step_2_analysis():
    print("\n[Step 2] analysis")
    model = ResNetWrapper()
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops = 2 * macs
    print(f"total GFLOPs (batch=1): {flops / 1e9:.4f}")

    batches = [4, 32, 128, 512]

    Cin, Cout, K, H, W = 64, 64, 3, 56, 56
    ridge_point = 52.0

    print(f"\nanalysis for ResNet block (Conv 3x3, 64->64, 56x56)")
    print(f"{'Batch':<8} | {'GFLOs':<10} | {'Mem(MB)':<10} | {'AI':<10} | {'Limiter'}")
    print("-" * 60)

    for B in batches:
        layer_flops = 2 * B * Cout * Cin * (K**2) * H * W
        mem_bytes = 4 * ((B * Cin * H * W) +
                         (Cout * Cin * K**2) + (B * Cout * H * W))
        ai = layer_flops / mem_bytes
        limiter = "Compute" if ai > ridge_point else "Memory"
        print(
            f"{B:<8} | {layer_flops/1e9:<10.4f} | {mem_bytes/1e6:<10.4f} | {ai:<10.2f} | {limiter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()

    if args.step == 1:
        step_1_export()
    elif args.step == 2:
        step_2_analysis()
