from _jl import *

if __name__ == "__main__":
    NUM_SAMPLES = 5
    import numpy as np
    t = ojlt(80, 3, 10)
    arrs = np.array([[i + NUM_SAMPLES * j for i in range(t.from_size())] for j in range(NUM_SAMPLES)], dtype=np.double)
    # arrs = [np.array([i for i in range(t.from_size())], dtype=np.double) for j in range(10)]
    assert len(arrs) == NUM_SAMPLES
    outdata = t.matrix_apply_oop(arrs)
    print(outdata)


