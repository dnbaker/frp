import jl
import numpy as np

if __name__ == "__main__":
    t = jl.ojlt(400, 25, 10)
    arr = np.array([i for i in range(t.from_size())], dtype=np.double)
    t.apply_inplace(arr)
    print(arr)
