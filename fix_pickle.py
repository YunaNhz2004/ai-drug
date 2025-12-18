import pickle
import sys
import numpy as np
import io

# ---- MAP MODULE CŨ ----
sys.modules['numpy._core'] = np
sys.modules['numpy._core.numeric'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy.core._multiarray_umath'] = np.core._multiarray_umath

# ---- CUSTOM UNPICKLER ----
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect numpy._core về numpy
        if module.startswith('numpy._core'):
            module = 'numpy'
        
        # Map các hàm cũ
        if name == '_frombuffer':
            return np.frombuffer
        if name == '_reconstruct':
            return np.core.multiarray._reconstruct
            
        return super().find_class(module, name)

print("Fixing TOX21.pkl...")

try:
    # Load bằng custom unpickler
    with open("data/processed/tox21.pkl", "rb") as f:
        unpickler = NumpyUnpickler(f)
        data = unpickler.load()

    print(f"✅ Loaded data: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

    # Save lại bằng pickle protocol 4
    with open("data/processed/tox21.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    print("✅ Successfully fixed TOX21.pkl")

except FileNotFoundError:
    print("⚠️ File not found, trying data/TOX21.pkl")
    try:
        with open("data/TOX21.pkl", "rb") as f:
            unpickler = NumpyUnpickler(f)
            data = unpickler.load()
        
        with open("data/TOX21.pkl", "wb") as f:
            pickle.dump(data, f, protocol=4)
        
        print("✅ Successfully fixed data/TOX21.pkl")
    except Exception as e2:
        print(f"❌ Error with fallback: {e2}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)