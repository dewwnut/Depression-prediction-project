import joblib
import numpy as np
from pathlib import Path
from pprint import pprint

EMB_PATH = Path(r"C:/Users/Owner/depression-prediction/data/cache/embeddings_minilm_small.pkl")

def to_numpy(obj):
    
    try:
        
        if hasattr(obj, "data"):
            arr = np.asarray(obj.data)
            return arr
        
        if isinstance(obj, np.ndarray):
            return obj

        if isinstance(obj, (list, tuple)):
            return np.asarray(obj)
        
        if isinstance(obj, dict):
            for key in ("embeddings", "X", "vectors", "arr"):
                if key in obj:
                    return np.asarray(obj[key])
            # fallback: if values are array-like, pick first array-like with ndim>=2
            for v in obj.values():
                try:
                    a = np.asarray(v)
                    if a.ndim >= 2:
                        return a
                except Exception:
                    pass
        # fallback: try direct conversion
        return np.asarray(obj)
    except Exception as e:
        raise RuntimeError(f"Could not convert loaded object to numpy array: {e}")

def main():
    print("Loading with joblib:", EMB_PATH)
    obj = joblib.load(EMB_PATH)
    print("Loaded object type:", type(obj))
    # inspect a bit
    try:
        arr = to_numpy(obj)
    except Exception as e:
        print("Error converting to numpy:", e)
        return

    print("Converted to numpy. dtype:", arr.dtype, "shape:", arr.shape)
    print("Example first row (first 10 values):", arr[0][:10].tolist() if arr.shape[0] > 0 else [])
    print("Memory size (MB):", arr.nbytes / (1024**2))

if __name__ == "__main__":
    main()
