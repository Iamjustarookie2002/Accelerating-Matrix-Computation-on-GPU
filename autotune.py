import numpy as np
from kernel_tuner import tune_kernel

# Problem size (matrix dimensions)
m = n = k = 512

# Create input and output arrays
A = np.random.randint(0, 10, size=(m, n)).astype(np.int32)
B = np.random.randint(0, 10, size=(n, k)).astype(np.int32)
C = np.zeros((m, k)).astype(np.int32)

# Arguments to be passed to the kernel
args = [A, B, C, np.int32(m), np.int32(n), np.int32(k)]

# Tunable parameters
tune_params = {
    "BN": [8, 16, 32, 64],
    "BM": [64, 128, 256],
    "BK": [64, 128, 256],
    "WM": [32, 64, 128, 256],
    "WK": [32, 64, 128, 256],
    "WKITER": [1, 2, 4, 8],
    "TM": [4, 8, 16, 32],
    "TK": [4, 8, 16, 32],
    "NUM_THREADS": [128, 256],
}

restrictions = [
    "BK % WK == 0",
    "BM % WM == 0",
    "(BK / WK) * (BM / WM) == (NUM_THREADS / 32)",
    "(WM * WK) % (32 * TM * TK * WKITER) == 0",
    "WM % ((WM * WK) // (32 * TM * TK * WKITER)) == 0",
    "WK % WKITER == 0",
    "(NUM_THREADS * 4) % BK == 0",
    "(NUM_THREADS * 4) % BN == 0",
    "BK % (16 * TK) == 0",
    "BM % (16 * TM) == 0",
    "(BM * BN) % (4 * NUM_THREADS) == 0",
    "(BN * BN) % (4 * NUM_THREADS) == 0"
]
problem_size = (m, k)
def grid_div_x(params):
    return params["BK"]

def grid_div_y(params):
    return params["BM"]


results, env = tune_kernel(
    kernel_name="matrix_mul",
    kernel_source="tune_matrix_mul.cu",
    problem_size=problem_size,
    arguments=args,
    tune_params=tune_params,
    restrictions=restrictions,
    lang="CUDA",
    block_size_names=["NUM_THREADS"],
    verbose=True,
    grid_div_x=grid_div_x,
    grid_div_y=grid_div_y,
)

# Print best configuration
print("Best config:\n", results[0])
