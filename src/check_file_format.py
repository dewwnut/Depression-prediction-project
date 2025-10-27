import os

path = r"C:/Users/Owner/depression-prediction/data/cache/embeddings_minilm_small.pkl"

print(f"Checking file: {path}")
size = os.path.getsize(path)
print("File size (bytes):", size)


with open(path, "rb") as f:
    head = f.read(50)

print("First 50 bytes:", head[:50])


if head.startswith(b'\x80\x04'):
    print("→ Looks like a pickle (protocol 4).")
elif head[:6] == b'\x93NUMPY':
    print("→ This is a .npy NumPy file (rename to .npy).")
elif head[:8] == b'PK\x03\x04':
    print("→ This is a zipped file (possibly .npz or HF dataset cache).")
elif head[:2] == b'\x1f\x8b':
    print("→ This is a gzipped file (maybe a compressed dataset).")
elif head[:7] == b'joblib\x01':
    print("→ Looks like a joblib file.")
else:
    print("→ Unknown signature. Might be torch, HF dataset, or another binary type.")
