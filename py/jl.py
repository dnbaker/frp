from _jl import *

if __name__ == "__main__":
    import numpy as np
    t = ojlt(400, 25, 10)
    arrs = [np.array([i for i in range(t.from_size())], dtype=np.double) for j in range(10)]
    assert len(arrs) == 10
    for i in range(len(arrs)):
        t.apply_inplace(arrs[i])
    print(arrs)


