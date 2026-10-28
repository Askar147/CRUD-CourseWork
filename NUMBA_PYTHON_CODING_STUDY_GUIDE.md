# Python CUDA with Numba — Course-Only Coding Study Guide

**CSCI GPU Programming | Source-restricted study edition**

> **Scope:** This guide is built ONLY from the uploaded Python Numba materials: `Introduction+to+CUDA+Python+with+Numba.md`, `Custom+CUDA+Kernels+in+Python+with+Numba.md`, `Effective+Memory+Use.md`, `AC_CUDA_Python_3.pptx`, `AC_CUDA_Python_4.pptx`, `AC_CUDA_Python_5.pptx`, and the Python file `reduce0.py`. It does **not** use CUDA C/C++ presentations, other-language notebooks, or previous midterm PDFs. Worked solutions below are reconstructed from exercises or directly based on Python examples in these sources; they are not claims about an upcoming exam.

## Navigation / recommended preparation order

| Priority | Study unit | Material used | What to be able to code |
|---|---|---|---|
| 1 | Custom Numba kernels | *Custom CUDA Kernels*, “A First CUDA Kernel” | `@cuda.jit`, `cuda.grid(1)`, kernel launch, device arrays |
| 1 | Grid-stride loops | *Custom CUDA Kernels*, “A First Grid Stride Loop” | `cuda.gridsize(1)` and a correct loop covering all elements |
| 1 | Race conditions and atomics | *Custom CUDA Kernels*, “Atomic Operations” and histogram assessment | `cuda.atomic.add`, safe histogram |
| 1 | Global vs shared memory | *Effective Use of the Memory Subsystem*, “Shared Memory” | `cuda.shared.array`, `cuda.syncthreads()` |
| 1 | 2D kernels and coalescing | *Effective Memory Use*; `AC_CUDA_Python_3.pptx` | `cuda.grid(2)`, matrix addition, correct row/column indexing |
| 1 | Transpose / bank conflicts | *Effective Memory Use*; `AC_CUDA_Python_4/5.pptx` | shared-memory transpose, `(32, 33)` padding |
| 2 | GPU ufuncs and transfers | *Introduction to CUDA Python with Numba* | `@vectorize(target='cuda')`, device arrays, `out=` |
| 2 | CPU compilation, timing, verification | First two Numba Markdown lessons | `@njit`, `%timeit`, `cuda.synchronize()`, `np.testing` |
| 2 | Python reduction | `reduce0.py` | two-pass reduction using dynamic shared memory |
| 3 | Debugging and GPU RNG | *Custom CUDA Kernels*, appendices | simulator, `print`, seeded RNG state |

### What is deliberately NOT in the guide

The specific Numba Python Markdown lessons and Python slide decks listed above **do not teach a Python-stream workflow, pinned host-memory allocation, asynchronous pinned-memory transfers, CUDA event timing, or code to query the SM count**. Those topics must not be imported from the other course materials for this source-limited guide. The Python kernel lesson **does** explain what SMs and warps are and gives *general heuristics* for grid/block sizes; that is included below. `reduce0.py` is included because it is a provided **Python Numba** program, not because of the separate C-based reduction lecture.

---

# Part I — Numba fundamentals

## 1. Three different kinds of Numba function

Do not confuse the decorators:

| Decorator | Executes on | What it is used for | Result style |
|---|---|---|---|
| `@njit` (or the course's `@jit(nopython=True)`) | CPU | Compiled numerical Python | Ordinary Python return value |
| `@vectorize(['float32(float32, float32)'], target='cuda')` | GPU | Element-by-element ufunc, with automatically selected parallel configuration | Array output (or a provided `out=` array) |
| `@cuda.jit` | GPU | Explicit custom kernel with threads and blocks | No kernel return value: writes into an output array |
| `@cuda.jit(device=True)` | GPU, called by GPU code | Helper callable inside a GPU ufunc/kernel | Can return scalar or tuple values |

The course introduces CPU compilation first. A typical CPU example is:

```python
from numba import njit

@njit
def sum_squares(a):
    total = 0.0
    for i in range(a.size):
        total += a[i] * a[i]
    return total
```

**Important idea from the introduction notebook:** Numba uses just-in-time compilation and specializes compiled functions for argument types; first-call compilation time is different from steady-state execution time. The course teaches nopython mode as its preferred CPU compilation approach.

### GPU ufunc example (directly follows the course's pattern)

```python
import numpy as np
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y

x = np.arange(100000, dtype=np.float32)
y = 2 * x
result = add_ufunc(x, y)
np.testing.assert_allclose(result, x + y)
```

**Know why it works:** the function body is written for **one pair of scalar inputs**, but Numba applies it elementwise over entire arrays. When ordinary host arrays are passed, the example notebook describes Numba handling memory allocation and transfers automatically. This makes ufuncs convenient, although repeated host↔GPU transfers can dominate runtime.

### Device helper function

The introduction notebook's example converts polar coordinates to Cartesian coordinates using a device helper inside a vectorized GPU function:

```python
import math
from numba import cuda, vectorize

@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    return rho * math.cos(theta), rho * math.sin(theta)

@vectorize(['float32(float32, float32)'], target='cuda')
def distance_from_origin(rho, theta):
    x, y = polar_to_cartesian(rho, theta)
    return math.sqrt(x*x + y*y)
```

This is a **simplified worked variant** of the source's four-argument `polar_distance` example. `device=True` means the helper is invoked by GPU code, not directly by ordinary host Python.

### Supported Python inside kernels

The introduction material explicitly mentions `if`/`elif`/`else`, `for` and `while`, arithmetic, selected `math`/`cmath` functions, and tuples. **Do not assume an arbitrary Python or NumPy operation will work inside a device function.** For example, the notebook uses `math.exp` inside a CUDA ufunc rather than a NumPy array operation.

## 2. Host and device arrays: the memory pattern taught in Python

The first notebook devotes a full section to **minimizing unnecessary transfers**:

```python
from numba import cuda
import numpy as np

x = np.arange(100000, dtype=np.float32)
y = (2 * x).astype(np.float32)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array(x.shape, dtype=np.float32)

add_ufunc(d_x, d_y, out=d_out)
result = d_out.copy_to_host()

np.testing.assert_allclose(result, x + y)
```

Know this table:

| Numba expression | Course meaning |
|---|---|
| `cuda.to_device(x)` | Copy a NumPy host array to device memory |
| `cuda.device_array(shape, dtype=...)` | Allocate an uninitialized device array |
| `cuda.device_array_like(d_x)` | Allocate an uninitialized device array with shape/dtype like `d_x` |
| `d_out.copy_to_host()` | Copy output to a new host NumPy array |
| `add_ufunc(d_x, d_y, out=d_out)` | Reuse existing device input/output allocations |

The lesson's `make_pulses` exercise makes the same point: keep the pulse output on the GPU so the following addition ufunc consumes a device array, rather than transferring intermediate data to the CPU and back.

**Potential coding trap:** uninitialized `device_array` memory must be written before using its contents. A device array is not automatically a zero-filled array.

## 3. Your first custom kernel

The kernel lesson writes a simple vector addition kernel:

```python
import numpy as np
from numba import cuda

@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]

n = 4096
a = np.arange(n, dtype=np.int32)
b = np.ones(n, dtype=np.int32)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_out = cuda.device_array_like(d_a)

threads_per_block = 128
blocks = (n + threads_per_block - 1) // threads_per_block
add_kernel[blocks, threads_per_block](d_a, d_b, d_out)

np.testing.assert_array_equal(d_out.copy_to_host(), a + b)
```

**Kernel rules to memorize:**

1. `@cuda.jit` declares a GPU kernel. A kernel's output is written to an array; do not `return out`.
2. `kernel[blocks, threads_per_block](arguments)` is the launch syntax used in the Python lessons.
3. `i = cuda.grid(1)` equals `cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x`.
4. `if i < n` guards against extra threads beyond the end of the array.
5. Kernel launches are asynchronous relative to host Python. A synchronous `copy_to_host()` waits for the data; for timing, the lesson explicitly calls `cuda.synchronize()`.

### Be ready to calculate indices by hand

For `blocks=3`, `threads_per_block=4`:

| Block index | Thread IDs inside block | Global indices |
|---|---|---|
| 0 | 0, 1, 2, 3 | 0, 1, 2, 3 |
| 1 | 0, 1, 2, 3 | 4, 5, 6, 7 |
| 2 | 0, 1, 2, 3 | 8, 9, 10, 11 |

For an array of length `10`, threads with global indices `10` and `11` must not access the array. They are handled by the bounds check.

---

# Part II — Flexible kernels, performance, and correctness

## 4. Grid-stride loops — a central course technique

The Python kernel notebook explains that creating one thread for every input element becomes unsuitable for very large datasets. Use a **grid-stride loop** so each thread handles multiple elements:

```python
from numba import cuda

@cuda.jit
def add_grid_stride(a, b, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, a.shape[0], stride):
        out[i] = a[i] + b[i]
```

Example configuration taken from the notebook:

```python
n = 100000
threads_per_block = 128
blocks = 30
# Total launched threads = 3840, but all 100000 items are processed.
add_grid_stride[blocks, threads_per_block](d_a, d_b, d_out)
```

The mathematical definitions are:

- `start = blockIdx.x * blockDim.x + threadIdx.x`
- `stride = blockDim.x * gridDim.x`
- Each thread works at positions `start`, `start + stride`, `start + 2*stride`, etc. until the index reaches `n`.

For 2 blocks × 4 threads, total grid size is `8`: thread `3` processes `3, 11, 19, ...` while indices remain in range.

**Why a grid-stride loop can still coalesce:** adjacent threads in the same warp initially access adjacent array elements; on their next iteration they again access an adjacent group separated from the previous group by the entire grid size.

### Source-based worked exercise: array hypotenuse

The kernel notebook explicitly asks you to rewrite `hypot_stride` using a grid-stride loop. A worked implementation is:

```python
import math
from numba import cuda

@cuda.jit
def hypot_stride(a, b, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, a.size, stride):
        out[i] = math.sqrt(a[i] * a[i] + b[i] * b[i])
```

Launch using the notebook's example `hypot_stride[128, 64](d_a, d_b, d_c)` and verify with `np.testing.assert_allclose(d_c.copy_to_host(), np.hypot(a,b), rtol=..., atol=...)`.

## 5. Warp and streaming multiprocessor concepts — only what the Python lesson says

The *Custom CUDA Kernels* lesson explicitly teaches:

- A GPU has **streaming multiprocessors (SMs)** that execute scheduled thread blocks.
- A thread block is assigned to one SM, and more than one block may reside on an SM.
- CUDA uses **warps of 32 threads**; instructions are issued to warps.
- Having other ready warps lets an SM do useful work while a memory or arithmetic instruction is waiting: **latency hiding**.
- The lesson's *starting heuristics*, not rigid requirements: choose block sizes that are multiples of 32, often `128–512` threads; consider roughly `2–4` blocks per SM, with the lesson also suggesting `20–100` blocks as a starting range.

The Python sources **do not give a mandatory `warp_size * sm_count` formula or show a Python API for querying the SM count**. Do not memorize either as a course-only requirement.

## 6. Timing code as shown in the Numba notebooks

The Python lesson explicitly benchmarks CPU calculations with `%timeit`, and kernel work with a synchronization call:

```python
# In a Jupyter notebook, after preparing arrays and compiling the kernel:
%timeit np.hypot(a, b)
%time hypot_stride[128, 64](d_a, d_b, d_c); cuda.synchronize()
```

**Why synchronize?** The launch returns control to Python before GPU work necessarily finishes. `cuda.synchronize()` makes the host wait before reading the host-side timing measurement.

The *Effective Memory Use* notebook also uses `%timeit kernel[blocks, threads](...); cuda.synchronize()`. Several cells in the source show `cuda.synchronize` **without parentheses**; those should be written as `cuda.synchronize()` when reproducing an actual synchronization call. This is an explicit correction to the copied notebook syntax, not an additional course topic.

**Comparison must be interpreted correctly:** CPU NumPy timing and GPU execution timing may cover different work. `to_device` and `copy_to_host` have their own costs, so a GPU kernel that looks fast by itself does not necessarily make the entire application faster.

## 7. Race conditions and atomic operations

The course gives the simplest race example:

```python
@cuda.jit
def unsafe_counter(counter):
    counter[0] += 1      # Multiple threads race on the same location

@cuda.jit
def safe_counter(counter):
    cuda.atomic.add(counter, 0, 1)
```

With `64` blocks and `64` threads, the expected counter increment is `4096`; the non-atomic version can lose increments.

The lesson distinguishes read-after-write and write-after-write hazards. Where possible, arrange for each thread to own different output indices. If multiple threads **must** update the same result, use an appropriate atomic operation.

### Worked solution to the notebook's histogram assessment

The assessment specifically asks for a grid-stride loop **and** atomics. This example follows the exact problem shape from the notebook:

```python
import numpy as np
from numba import cuda

@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        value = x[i]
        if xmin <= value < xmax:
            bin_number = int((value - xmin) / bin_width)
            if 0 <= bin_number < nbins:
                cuda.atomic.add(histogram_out, bin_number, 1)

# Host setup in the style of the notebook
x = np.random.normal(size=10000).astype(np.float32)
xmin = np.float32(-4.0)
xmax = np.float32(4.0)
d_x = cuda.to_device(x)
d_hist = cuda.to_device(np.zeros(10, dtype=np.int32))

cuda_histogram[128, 64](d_x, xmin, xmax, d_hist)
result = d_hist.copy_to_host()
expected, _ = np.histogram(x, bins=10, range=(-4.0, 4.0))
np.testing.assert_array_equal(result, expected)
```

The code above is a **worked answer reconstructed from the assignment**, not a supplied official solution. Pay attention to two independent requirements: `grid(1)/gridsize(1)` divides work, while `atomic.add()` protects shared bin counts.

---

# Part III — Two-dimensional kernels and memory performance

## 8. 2D block and grid indexing

*Effective Memory Use* introduces 2D grids with tuples:

```python
blocks = (2, 2)
threads_per_block = (2, 2)

@cuda.jit
def store_coordinates(out):
    x, y = cuda.grid(2)
    if x < out.shape[1] and y < out.shape[0]:
        out[y, x] = x + y / 10.0
```

The notebook's introductory coordinate example happens to index `A[x][y]`, but the matrix coalescing and transpose sections use **column `x` and row `y`**, so `matrix[y, x]` is the most useful convention for the later exercises. The coordinate kernel above has been **reconstructed** using that convention.

The formulas are:

```python
x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
```

### Source-based worked exercise: coalesced matrix addition

```python
@cuda.jit
def matrix_add(a, b, out):
    x, y = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        out[y, x] = a[y, x] + b[y, x]

threads_per_block = (32, 32)
blocks = (64, 64)  # exactly covers the 2048x2048 example in the notebook
matrix_add[blocks, threads_per_block](d_a, d_b, d_out)
```

Remember that x is the **fast-changing column index** in a row-major NumPy array. Adjacent thread `x` indices should usually access adjacent memory addresses.

## 9. Memory coalescing — the lesson's performance model

**Sources:** *Effective Memory Use*, “The Problem: Uncoalesced Memory Access Hurts Performance”; `AC_CUDA_Python_3.pptx`, slides 2–19 and 20–48.

- In a **coalesced** access pattern, neighboring threads in a warp access nearby contiguous memory locations, so memory traffic is served with relatively few transactions.
- In an **uncoalesced** pattern, neighboring threads request addresses far apart, causing more memory segments to be fetched and less of each fetched segment to be useful.
- The lecture emphasizes a counterintuitive point: a thread reading a row **sequentially** can still result in **uncoalesced access across the warp** if neighboring threads are reading **different rows** at the same instant.

The notebook compares:

```python
@cuda.jit
def add_experiment(a, b, out, stride, coalesced):
    i = cuda.grid(1)
    if i < out.size:
        if coalesced:
            out[i] = a[i] + b[i]
        else:
            out[i] = a[stride * i] + b[stride * i]
```

The source chooses `stride = 16`. It reports a large performance loss for the strided version, and the `AC_CUDA_Python_3` slides visualize why.

### Row sums versus column sums

The slide deck explains why column sums can be better coalesced **across a warp**:

```python
@cuda.jit
def row_sums(a, out):
    row = cuda.grid(1)
    if row < a.shape[0]:
        total = 0.0
        for col in range(a.shape[1]):
            total += a[row, col]
        out[row] = total

@cuda.jit
def col_sums(a, out):
    col = cuda.grid(1)
    if col < a.shape[1]:
        total = 0.0
        for row in range(a.shape[0]):
            total += a[row, col]
        out[col] = total
```

Both are correct algorithms. In the row version, adjacent threads have different `row` values and read separated memory locations at the same loop iteration. In the column version, adjacent threads have different `col` values and access adjacent positions within one row. This is the concept illustrated by `AC_CUDA_Python_3.pptx` slides 28–48.

## 10. Shared memory and barriers

*Effective Memory Use* describes device global memory as large and accessible across blocks, while **shared memory** is small, on-chip, and shared **only by threads in the same block**.

The syntax is:

```python
from numba import cuda, types

@cuda.jit
def reverse_four(a, out):
    tile = cuda.shared.array(4, dtype=types.int32)
    tid = cuda.threadIdx.x
    tile[tid] = a[tid]
    cuda.syncthreads()
    out[tid] = tile[3 - tid]

# Launch with exactly one block of four threads
reverse_four[1, 4](d_a, d_out)
```

**Trace for input `[0, 1, 2, 3]`:** four threads write `tile[0]` through `tile[3]`; the barrier waits until all threads have written; thread 0 reads `tile[3]`, thread 1 reads `tile[2]`, etc.; output is `[3, 2, 1, 0]`.

The lesson emphasizes a **compile-time constant shape** for a fixed-size shared allocation (for instance `(32,32)`), and a synchronization point before reading values another thread may have written. Keep every thread in a block participating in the barrier; do not return from some threads and then have other threads wait at the barrier.

## 11. Matrix transpose: using shared memory for coalesced input and output

**Sources:** *Effective Memory Use*, “Exercise: Use Shared Memory for Coalesced Reads and Writes With Matrix Transpose”; `AC_CUDA_Python_4.pptx`, slides 5–41.

The simplest transpose is:

```python
@cuda.jit
def naive_transpose(a, out):
    x, y = cuda.grid(2)
    if y < a.shape[0] and x < a.shape[1]:
        out[x, y] = a[y, x]
```

The source explains that neighboring threads read contiguous elements of a source row, but write to different output rows, giving uncoalesced writes.

### Worked course-style shared-memory transpose

This is the *bank-conflict-free* variant derived from the Python course's final exercise (32×32 tiles with one extra shared-memory column). **Assume both matrix dimensions are divisible by 32**:

```python
from numba import cuda, types

@cuda.jit
def tile_transpose(a, out):
    tile = cuda.shared.array((32, 33), dtype=types.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    x = cuda.blockIdx.x * cuda.blockDim.x + tx
    y = cuda.blockIdx.y * cuda.blockDim.y + ty

    # Coalesced global reads into shared memory
    tile[ty, tx] = a[y, x]
    cuda.syncthreads()

    # Swap block coordinates; maintain contiguous writes along x
    out_x = cuda.blockIdx.y * cuda.blockDim.y + tx
    out_y = cuda.blockIdx.x * cuda.blockDim.x + ty

    out[out_y, out_x] = tile[tx, ty]

# For a 4096x4096 float32 array, as in the notebook:
threads = (32, 32)
blocks = (128, 128)
tile_transpose[blocks, threads](d_a, d_out)
# Verify: np.testing.assert_array_equal(d_out.copy_to_host(), a.T)
```

**Why synchronize after filling `tile`?** Each thread may need to read an element from `tile` written by a **different** thread. If all writes have not finished, the result is not reliable.

**Why are block indices swapped on output?** The transposed tile belongs to output coordinates corresponding to the opposite location in the block grid.

**Why is there an extra column?** The next section answers that.

## 12. Shared-memory bank conflicts and one-column padding

**Sources:** `AC_CUDA_Python_5.pptx`, slides 2–9 and 11–25; *Effective Memory Use*, “Assessment: Resolve Memory Bank Conflicts”.

The slides state that actual shared memory consists of **32 banks**, each **4 bytes wide** in the presentation's model. Consecutive 4-byte words are assigned to successive banks. Multiple threads of a warp accessing **different addresses in the same bank** can cause conflicts, serializing requests.

The transpose kernel needs a *columnar* read from shared memory:

```python
tile[tx, ty]  # these accesses can be bank-conflicting if row length is 32
```

Why does padding work for float32?

- With `tile = cuda.shared.array((32, 32), types.float32)`, the start of successive rows lies 32 words apart, which maps to the **same** bank.
- With `tile = cuda.shared.array((32, 33), types.float32)`, successive rows start 33 words apart, so a column traverses different banks.
- This adds only **one unused column** and makes the transpose shared-memory access pattern bank-conflict-free in the lecture's example.

**Critical exam reasoning:** Coalescing concerns **global memory transactions**; bank conflicts concern **shared memory**. They are different optimizations, and the course demonstrates both with transpose.

---

# Part IV — Python reduction, debugging, and optional material

## 13. Reduction using the supplied Python file `reduce0.py`

**Source:** `reduce0.py` is a Python Numba implementation of an interleaved-addressing reduction. Its core algorithm is:

```python
@cuda.jit
def reduce0(g_idata, g_odata):
    sdata = cuda.shared.array(shape=0, dtype=types.int32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    sdata[tid] = g_idata[i]
    cuda.syncthreads()

    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s *= 2

    if tid == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]
```

Here, `shape=0` denotes **dynamically sized shared memory**; the number of bytes is specified when launching the kernel. The original example launches one reduction stage and then reduces the per-block partial results in a second stage:

```python
shared_bytes = threads_per_block * np.dtype(np.int32).itemsize
reduce0[blocks, threads_per_block, 0, shared_bytes](d_a, d_partials)
```

**What you need to understand, not merely copy:**

1. Each thread loads one input element into the block's shared memory.
2. Every thread must finish its load before reduction starts (`syncthreads`).
3. The first iteration adds pairs for which `tid % 2 == 0`; the next works with stride `2`, then `4`, and so on.
4. Threads synchronize between iterations because later iterations depend on partial sums from the earlier ones.
5. Thread 0 writes the **one partial sum per block**.
6. Another kernel launch is used to combine the partial sums; a block-wide `syncthreads()` does not synchronize all blocks of the grid.

The supplied `reduce0.py` uses `n=1024` and `threads_per_block=1024`; that case fits its assumptions. As written, the kernel has no `if i < n` guard, and the unguarded shared-memory accesses assume suitably sized blocks (especially a power-of-two thread count). **Do not assume the original function safely handles arbitrary array sizes without adaptation.**

### Optional Python-only extension: arbitrary-size reduction

This extension is based on the same Numba shared-memory primitives and the supplied Python reduction, with a bounds-check and multiple passes. It uses a descending reduction stride and a fixed power-of-two block size:

```python
import numpy as np
from numba import cuda, types

@cuda.jit
def reduce_sum_pass(inp, partials):
    shared = cuda.shared.array(shape=0, dtype=types.float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    if i < inp.size:
        shared[tid] = inp[i]
    else:
        shared[tid] = 0.0
    cuda.syncthreads()

    offset = cuda.blockDim.x // 2
    while offset > 0:
        if tid < offset:
            shared[tid] += shared[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        partials[cuda.blockIdx.x] = shared[0]

# Host-side multi-pass reduction; input must be nonempty.
a = np.arange(1000, dtype=np.float32)
current = cuda.to_device(a)
threads = 256                 # power of two

while current.size > 1:
    blocks = (current.size + threads - 1) // threads
    partials = cuda.device_array(blocks, dtype=np.float32)
    shared_bytes = threads * np.dtype(np.float32).itemsize
    reduce_sum_pass[blocks, threads, 0, shared_bytes](current, partials)
    current = partials

answer = current.copy_to_host()[0]
np.testing.assert_allclose(answer, a.sum(dtype=np.float32), rtol=1e-5)
```

This implementation is **a teaching extension**, not copied from the Python file verbatim. The key source-based elements are dynamic shared memory, `cuda.syncthreads()`, tree accumulation, per-block outputs, and repeated kernel launches.

## 14. Debugging from the Numba kernel notebook

The *Custom CUDA Kernels* appendix covers:

- `print('in range', x[i], bin_number)` inside a kernel for simple scalar debugging. Printing every thread can produce huge output.
- `NUMBA_ENABLE_CUDASIM=1 python your_program.py` to execute CUDA kernels through the Numba simulator on the CPU for debugging logic. The notebook demonstrates using Python's debugger for a selected simulated thread.
- `@cuda.jit(debug=True)` and the debugging examples to help locate invalid memory access. Some external utility names in that appendix are historical, but this guide does not expand into unrelated native CUDA tooling.

### Common errors worth checking

| Error | Likely cause | Python lesson connection |
|---|---|---|
| Some output elements unchanged | Grid too small and no grid-stride loop | First kernel; grid-stride section |
| Out-of-range access | Too many threads, no bounds test | First kernel; debugging appendix |
| Wrong histogram counts | Non-atomic read–modify–write | Histogram assessment |
| Shared-memory result unstable | Missing `cuda.syncthreads()` | Shared memory / transpose |
| Slow row/matrix access | Neighboring threads access different distant regions | Coalescing slides |
| Slow tiled transpose | Shared columnar bank conflicts | Bank conflict slides |
| GPU time seems implausibly tiny | Kernel launch timed without synchronization | Timing section |
| GPU ufunc is slower on tiny inputs | Launch and host/device transfer costs dominate | Introductory ufunc comparison |

## 15. Optional appendix: random numbers on the GPU

The Python custom-kernels notebook includes an **optional** Monte Carlo appendix using Numba RNG state:

```python
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)

threads_per_block = 64
blocks = 24
rng_states = create_xoroshiro128p_states(
    threads_per_block * blocks, seed=1
)

@cuda.jit
def monte_carlo_mean(rng_states, iterations, out):
    tid = cuda.grid(1)
    total = 0.0
    for _ in range(iterations):
        total += xoroshiro128p_uniform_float32(rng_states, tid)
    out[tid] = total / iterations
```

This is relevant if an exercise asks for **GPU-generated** random values. The notebook initializes one independent RNG state per thread, then passes the state array into a kernel. It is distinct from generating random numbers on the CPU with NumPy and copying them to the GPU.

---

# Part V — Course-aligned practice problems

These are adapted from exercises **in the Python Numba course materials**, not inferred from old exam PDFs.

## Problem A — Write a grid-stride kernel (15–20 minute practice)

Given two arrays `a` and `b`, each with `N=100000`, compute `out[i] = sqrt(a[i]**2 + b[i]**2)` on the GPU. Use exactly **30 blocks × 128 threads**. Return a host array and compare it with `np.hypot(a, b)` using `np.testing.assert_allclose`.

**Must show:** a `@cuda.jit` kernel, `cuda.grid(1)`, `cuda.gridsize(1)`, device allocation, kernel launch, and correctness check.

**Based on:** *Custom CUDA Kernels*, grid-stride hypotenuse exercise.

## Problem B — Histogram with atomics (20–25 minute practice)

Generate 10,000 `float32` values, create 10 histogram bins for the range `[-4,4)`, and write a Numba CUDA kernel that counts values correctly with **128 blocks × 64 threads**. Start the histogram on the device filled with zeros and compare with the CPU.

**Must show:** grid-stride loop, bin-width computation, range checks, `cuda.atomic.add`.

**Based on:** *Custom CUDA Kernels*, histogram assessment.

## Problem C — Coalescing and 2D kernels (15–20 minute practice)

Given two `2048×2048` float32 arrays, add them with a 2D CUDA grid configured with **(32,32)** threads per block and **(64,64)** blocks. Explain why `a[y,x]` usually coalesces for consecutive x threads, then show a strided or transposed access pattern that changes performance without changing correctness.

**Must show:** `cuda.grid(2)`, row/column layout, matrix indexing, timing plus synchronization.

**Based on:** *Effective Memory Use*, 2D matrix-add exercise and `AC_CUDA_Python_3`.

## Problem D — Shared-memory transpose and padding (25–35 minute practice)

Write a transpose for a square 64×64 float32 matrix using **(32,32)** thread blocks. First write a naïve transpose. Next use shared memory `(32,32)`; finally change it to `(32,33)` and explain the reason. Check all three outputs against `a.T`.

**Must show:** shared tile, load from `a[y,x]`, `cuda.syncthreads()`, swapped output block indices, `tile[tx,ty]`, and padding.

**Based on:** *Effective Memory Use* transpose and bank conflict exercises; `AC_CUDA_Python_4/5`.

## Problem E — Trace a reduction (10–15 minute practice)

Using the interleaved-addressing algorithm from `reduce0.py`, trace the shared-memory contents for input `[1,2,3,4,5,6,7,8]`.

- At `s=1`, threads `0,2,4,6` add a value from `tid+s`.
- At `s=2`, threads `0,4` add a value from `tid+s`.
- At `s=4`, thread `0` adds from thread `4`.

**Expected final result:** `shared[0] == 36`. Explain what the barrier does after each level and why another kernel may be needed for a larger input.

**Based on:** `reduce0.py`.

### Mini reference answers (test yourself before revealing mentally)

- **A:** `for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):` and `out[i] = math.sqrt(a[i]*a[i]+b[i]*b[i])`.
- **B:** `cuda.atomic.add(histogram_out, bin_number, 1)`, not `histogram_out[bin_number] += 1`.
- **C:** `x,y = cuda.grid(2)` and `out[y,x] = a[y,x] + b[y,x]`; check that `x` maps to contiguous columns.
- **D:** use `tile = cuda.shared.array((32,33), types.float32)` and a barrier between writing/reading shared memory.
- **E:** after the first level, active partial sums are `3,7,11,15`; after the second, `10,26`; after the final, `36`.

---

# Part VI — One-page memorization sheet

```python
import numpy as np
import math
from numba import cuda, types, njit, vectorize

# CPU numerical compilation
@njit
def cpu_calc(x):
    return x*x

# GPU ufunc: written as a scalar operation
@vectorize(['float32(float32, float32)'], target='cuda')
def gpu_add(x, y):
    return x + y

# GPU device helper: callable from other GPU code
@cuda.jit(device=True)
def square(x):
    return x*x

# One-dimensional GPU kernel with grid-stride loop
@cuda.jit
def square_array(a, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, a.size, stride):
        out[i] = square(a[i])

# Two-dimensional kernel
@cuda.jit
def matrix_add(a, b, out):
    x, y = cuda.grid(2)
    if x < a.shape[1] and y < a.shape[0]:
        out[y, x] = a[y, x] + b[y, x]

# Shared memory inside a kernel
# tile = cuda.shared.array((32, 33), dtype=types.float32)
# cuda.syncthreads()

# Atomic update inside a kernel
# cuda.atomic.add(histogram, bin_number, 1)

# Host-side memory and launch
n = 100000
a = np.arange(n, dtype=np.float32)
d_a = cuda.to_device(a)
d_out = cuda.device_array_like(d_a)
square_array[30, 128](d_a, d_out)
result = d_out.copy_to_host()
np.testing.assert_allclose(result, a*a)

# In a Jupyter notebook: %timeit ... ; cuda.synchronize()
```

## Ten questions to answer without notes

1. Why does `@cuda.jit` normally write to `out` instead of returning an array?
2. Expand `cuda.grid(1)` using the CUDA indexing variables supplied by Numba.
3. What is `cuda.gridsize(1)`? Why is it essential for very large arrays?
4. Why can a grid-stride loop use fewer GPU threads than input elements?
5. Why should blocks often contain a multiple of 32 threads, according to the Numba lesson?
6. How do `cuda.to_device`, `cuda.device_array`, and `copy_to_host()` differ?
7. When is `cuda.atomic.add` needed? Why is normal `+=` unsafe in a histogram?
8. Why does `cuda.syncthreads()` appear after writing into a shared-memory tile?
9. Why is a warp's access to `a[y,x]` generally better coalesced when neighboring threads vary `x`?
10. Why does a shared-memory transpose often use a `(32,33)` tile instead of `(32,32)`?

## Suggested preparation sequence

- **Pass 1 — Code fluency:** type the 1D addition kernel, the grid-stride hypotenuse kernel, and the histogram kernel without reference material.
- **Pass 2 — Memory reasoning:** hand-trace `cuda.grid(2)`, identify coalesced and uncoalesced accesses, and write the shared-memory transpose.
- **Pass 3 — Optimization:** explain what changes between a `(32,32)` and `(32,33)` shared tile; time kernels with `cuda.synchronize()`.
- **Pass 4 — Python reduction:** trace `reduce0.py`, then run the generalization with arrays of several sizes.
- **Pass 5 — Exam rehearsal:** solve Problems A–E from blank Python cells; verify results before looking at the worked patterns.

---

# Exact source index

| File | Material actually used |
|---|---|
| `Introduction+to+CUDA+Python+with+Numba.md` | “First Steps: Compile for the CPU”; “Object and nopython Modes”; “Making ufuncs for the GPU”; “CUDA Device Functions”; “Allowed Python on the GPU”; “Managing GPU Memory” |
| `Custom+CUDA+Kernels+in+Python+with+Numba.md` | “A First CUDA Kernel”; “An Aside on Hiding Latency and Execution Configuration Choices”; “A First Grid Stride Loop”; “Timing the Kernel”; “Atomic Operations and Avoiding Race Conditions”; histogram assessment; debugging and random-number appendices |
| `Effective+Memory+Use.md` | “The Problem: Uncoalesced Memory Access Hurts Performance”; “Column and Row Sums”; “2 and 3 Dimensional Blocks and Grids”; “Coalesced 2-Dimensional Matrix Add”; “Shared Memory”; matrix transpose; bank conflicts |
| `AC_CUDA_Python_3.pptx` | Slides 2–19 (warp-level access and coalescing); 20–48 (row versus column sums) |
| `AC_CUDA_Python_4.pptx` | Slides 5–41 (naïve and shared-memory matrix transpose) |
| `AC_CUDA_Python_5.pptx` | Slides 2–9 (bank conflicts); 11–25 (padding shared-memory tiles) |
| `reduce0.py` | Dynamic shared memory, interleaved-addressing tree reduction, partial sums across two kernel launches |

**Source restrictions maintained:** no prior-midterm task requirements were used, and no implementations are taken from the C/C++ presentations or the other-language notebooks. This guide is explicitly oriented toward reproducing and understanding the **Python Numba exercises**.
