import struct
import numpy as np

def test_rmsnorm():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    eps = 1e-5
    
    # RMSNorm formula
    mean_sq = np.mean(x ** 2)
    scale = 1.0 / np.sqrt(mean_sq + eps)
    out = x * scale * weight
    print(f"[Python Ref] RMSNorm out: {out}")

def inspect_model_header(path):
    with open(path, "rb") as f:
        # Config header: 7 int32
        data = f.read(28)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", data)
        print("========================================")
        print("  Model Checkpoint Header Verification")
        print("========================================")
        print(f"  dim:         {dim}")
        print(f"  hidden_dim:  {hidden_dim}")
        print(f"  n_layers:    {n_layers}")
        print(f"  n_heads:     {n_heads}")
        print(f"  n_kv_heads:  {n_kv_heads}")
        print(f"  vocab_size:  {vocab_size}")
        print(f"  seq_len:     {seq_len}")
        print("========================================")

if __name__ == "__main__":
    test_rmsnorm()
    inspect_model_header("models/stories15M.bin")
