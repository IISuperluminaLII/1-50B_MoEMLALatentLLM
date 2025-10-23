import os, sys, ctypes

# Add DLL directories BEFORE importing faiss
dll_paths = [
    r'C:\Users\Shashank Murthy\.conda\envs\150BLLM\Lib\site-packages\faiss',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin',
    r'C:\Users\Shashank Murthy\.conda\envs\150BLLM\Library\bin',
]

for p in dll_paths:
    p = os.path.abspath(p)
    if os.path.exists(p):
        try:
            os.add_dll_directory(p)
            print(f"Added DLL directory: {p}")
        except:
            pass

# NOW import faiss
import faiss
import numpy as np

print(f"FAISS version: {faiss.__version__}")
print(f"GPU support available: {hasattr(faiss, 'StandardGpuResources')}")

# GPU self-check
ngpu = faiss.get_num_gpus()
print(f"[INFO] Num GPUs detected by FAISS: {ngpu}")

if ngpu > 0:
    print("SUCCESS! FAISS-GPU is working!")

    # Quick test
    res = faiss.StandardGpuResources()
    d = 64
    nb = 1000
    xb = np.random.random((nb, d)).astype('float32')

    index_flat = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(xb)
    print(f"GPU index size: {gpu_index.ntotal}")
    print("GPU test successful!")
else:
    print("WARNING: No GPUs detected by FAISS")
