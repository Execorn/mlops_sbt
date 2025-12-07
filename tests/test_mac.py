import numpy as np
import tensor_ops
import time
import sys
def test_mac():
    depth, height, width = 64, 64, 64
    print(f"Shape: ({depth}, {height}, {width})")
    
    np.random.seed(42)
    A = np.random.rand(depth, height, width)
    B = np.random.rand(depth, height, width)
    C = np.random.rand(depth, height, width)
    
    start = time.time()
    res_cpp = tensor_ops.tensor_mac(A, B, C)
    cpp_dur = time.time() - start
    
    start = time.time()
    res_ref = A * B + C
    ref_dur = time.time() - start
    
    if np.allclose(res_cpp, res_ref):
        print("[OK] Results match NumPy reference.")
    else:
        print("[FAIL] Results DO NOT match!")
        sys.exit(1)
    print(f"Time C++: {cpp_dur:.6f} s")
    print(f"Time NP:  {ref_dur:.6f} s")
    
    print("\nBenchmark (256x256x256)")
    big_shape = (256, 256, 256)
    A = np.random.rand(*big_shape)
    B = np.random.rand(*big_shape)
    C = np.random.rand(*big_shape)

    tensor_ops.tensor_mac(A[:10], B[:10], C[:10])
    
    start = time.time()
    tensor_ops.tensor_mac(A, B, C)
    end = time.time()
    print(f"C++ Optimized (OpenMP+AVX): {end - start:.5f} sec")
if __name__ == "__main__":
    test_mac()