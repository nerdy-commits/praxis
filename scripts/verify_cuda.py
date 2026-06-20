import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    # Quick tensor test on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    print(f"GPU tensor test: OK (shape={y.shape})")
else:
    print("GPU: not available")
