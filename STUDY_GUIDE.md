# CSCI 723 Python CUDA Coding Midterm — Study Guide

**Based on the uploaded Fall 2026 PhD exam, Fall 2025 exam, `reduce0.py`, Numba CUDA notebooks/Markdown tutorials, memory-coalescing/shared-memory slides, and the Week 6 reduction lecture.**

> **Submission-language warning:** Both uploaded papers explicitly say to submit a single `.cu` file (CUDA C++). This package deliberately translates the problems into Python with **Numba CUDA** because that is the proposed preparation scenario. The uploaded papers do **not** confirm that the instructor will permit a Python submission.

## A. Know exactly what the two exams ask

### Fall 2026 PhD midterm (higher priority)

Given arrays `a`, `b`, each of length `N`, initialize values between -5 and 5; produce independent results `c = a+b`, `a-b`, `a*b`, `a/b`, and reduce/sum each resulting array. The exam says to:

1. Write GPU kernels to initialize `a` and `b` on **independent streams**.
2. Write separate kernels for `+`, `-`, `*`, `/` and separate **device functions** for each operation.
3. Write a reduction kernel using the Week 6 parallel reduction algorithm.
4. Write CPU/host operation functions; check GPU results against CPU results.
5. Include a main function, report device name, output arrays for `N = 10`, all sums and CPU/GPU multiplication timings.
6. For the main kernel launch use **blocks = warp size × number of streaming multiprocessors** and select suitable threads per block.
7. Include grid-stride loops, streams, asynchronous data migration, and resource cleanup.

Exam grading: 10 percentage points for task requirements, 3 for reduction, and 1 each for grid-stride loops, launch configuration, streams, async transfers, cleanup, device functions, and reduction implementation/calling. Total 20 percentage points in course grading.

### Fall 2025 midterm (also practice)

Compute **matrix multiplication** `C = A @ B`, not element-wise `A*B`, for integer matrices of shape `(64,64)` (`N = 1 << 6`). GPU fill on independent streams; GPU matrix kernel; CPU matrix function; verification; main; grid stride; hardware-dependent blocks; CPU and GPU timing; proper transfers/cleanup. Test `4x4` first before `64x64`.

## B. CUDA Python essential syntax

```python
import numpy as np
from numba import cuda, float32

@cuda.jit
def add_kernel(a, b, c, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]

n = 1000
threads = 256
blocks = (n + threads - 1) // threads

a = np.arange(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array(n, dtype=np.float32)
add_kernel[blocks, threads](d_a, d_b, d_c, n)
result = d_c.copy_to_host()
np.testing.assert_allclose(result, a + b)
```

Remember:

- `@cuda.jit` decorates a **GPU kernel**; kernel results go into output arrays; don't `return` the output array from the kernel.
- `@cuda.jit(device=True)` decorates a **GPU device function** called from inside GPU kernels; it *may* return a scalar.
- `cuda.grid(1) = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x`.
- `cuda.gridsize(1) = cuda.gridDim.x * cuda.blockDim.x`.
- Launch syntax: `kernel[blocks_per_grid, threads_per_block, stream](arguments)`.
- CUDA C++ `kernel<<<blocks, threads, shared_bytes, stream>>>` maps conceptually to Numba `kernel[blocks, threads, stream, shared_bytes]`. For the supplied static-shared-memory solutions, shared bytes are not needed.
- The host CPU and the GPU use different memory spaces unless managed memory mechanisms are deliberately employed. Explicit `to_device`, `device_array`, and `copy_to_host` are easiest here.

### Why grid-stride loops matter

An ordinary 1-thread-per-element kernel fails to cover all elements if the launch grid has fewer than `N` threads. If too many threads launch and there is no bounds check, it can access out of bounds.

```python
@cuda.jit
def multiply_kernel(a, b, c, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        c[i] = a[i] * b[i]
        i += stride
```

If the grid has 6 threads, thread 2 processes indices 2, 8, 14, 20, ... until `i >= N`. This works for **both small and very large** `N` and naturally produces coalesced memory access for successive threads.

## C. Hardware configuration, streams, copies and timing

### Hardware-dependent blocks

```python
gpu = cuda.get_current_device()
warp_size = gpu.WARP_SIZE                    # normally 32
sm_count = gpu.MULTIPROCESSOR_COUNT
blocks = warp_size * sm_count               # explicit midterm instruction
threads = 256                               # benchmark 128/256/512
```

**Important distinction:** the exam's `32 * SMs` is an imposed grid rule, not an independently proven optimum for every workload. In actual CUDA programming the optimum depends on the input size, registers, shared memory, and GPU occupancy. For reduction, using `min(blocks, ceil(N/threads))` avoids producing huge numbers of unnecessary partial sums; the supplied solution does that for reduction while honoring the 2026 exam launch rule for fill and arithmetic kernels.

### Two independent streams for initialization

```python
stream_a = cuda.stream()
stream_b = cuda.stream()
fill_a[blocks, threads, stream_a](d_a, n)
fill_b[blocks, threads, stream_b](d_b, n)
stream_a.synchronize()
stream_b.synchronize()
# Safe now for a dependent kernel to consume both arrays in another stream.
```

Within one stream, operations are ordered. Operations on different streams may overlap if GPU resources allow. **Different streams do not implicitly establish a cross-stream dependency**, so synchronize both fill streams before launching dependent work on a third stream. For basic exams this is simpler than cross-stream CUDA event dependencies.

### Pinned host memory + async copy

```python
host = cuda.pinned_array(n, dtype=np.float32)
d_a.copy_to_host(host, stream=stream_a)   # async device-to-host enqueue
stream_a.synchronize()                   # only now is it safe to read host
```

Host pinned memory makes asynchronous host/device copies possible. `cuda.to_device(host_array, stream=stream_a)` is a corresponding **host-to-device** async transfer when the host buffer is pinned. The 2026 problem specifically requests **GPU fill kernels**, so the complete supplied solution uses GPU fill, followed by async GPU-to-CPU migration of input arrays and results.

### Timing GPU versus CPU correctly

```python
import time

t0 = time.perf_counter()
cpu_result = a * b
cpu_ms = 1000 * (time.perf_counter() - t0)

# Warm up the Numba JIT before timing.
multiply_kernel[blocks, threads, stream](d_a, d_b, d_c, n)
stream.synchronize()

start = cuda.event(timing=True)
end = cuda.event(timing=True)
start.record(stream)
multiply_kernel[blocks, threads, stream](d_a, d_b, d_c, n)
end.record(stream)
end.synchronize()
gpu_kernel_ms = cuda.event_elapsed_time(start, end)
```

The above times **only GPU kernel execution**, not PCIe transfers. For end-to-end timing, start a CPU wall clock before transfers and stop it *after* synchronizing the last transfer. Always identify which timing is reported. Small `N` is often **slower** on GPU after launch and transfer overhead.

## D. Parallel reduction: central 2026 skill

Goal: `sum(c)` where `c` has `N` values. A normal element-wise kernel cannot compute the sum of all values without some way to combine partial results. A single scalar accumulator written by all GPU threads has a **race condition** unless atomics or a coordinated reduction are used; the exam specifically requests **Week 6 reduction**.

### Visual example

Start with `[1,2,3,4,5,6,7,8]`. Using a **sequential-addressing** shared-memory tree:

- First round (stride 4): add pairs `(1+5), (2+6), (3+7), (4+8)` => `[6,8,10,12]`.
- Second round (stride 2): `(6+10), (8+12)` => `[16,20]`.
- Third round (stride 1): `(16+20)` => `36`.

For 8 threads, there are 3 shared-memory reduction rounds, or `log2(8)`.

### Minimal 256-thread reduction stage

```python
from numba import cuda, float32

@cuda.jit
def reduce_kernel(values, partial, n):
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    step = cuda.gridsize(1)

    local = float32(0.0)
    while i < n:
        local += values[i]
        i += step

    shared = cuda.shared.array(256, dtype=float32)
    shared[tid] = local
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = shared[0]
```

Key points: (1) each thread first grid-stride-accumulates its global-memory elements; (2) every thread places its local partial result in block-local shared memory; (3) `cuda.syncthreads()` ensures all writes are finished; (4) half of threads combine values in each iteration; (5) thread 0 writes **one partial sum per block**.

To sum large arrays, launch this kernel again on the **partial sums**. Continue until one scalar remains. CUDA blocks cannot use ordinary `cuda.syncthreads()` to synchronize globally; the *next kernel invocation in the same stream* is ordered after the previous one. Keeping all stages on the GPU avoids needless host/device copies.

### Relation to uploaded `reduce0.py`

Your sample `reduce0.py` is the Week 6 **interleaved-addressing reduction**, which uses `s = 1; while s < blockDim.x: if tid % (2*s) == 0: sdata[tid] += sdata[tid+s]`. Its essentials—shared memory, a barrier after every round, and one result per block—are the same. The exam solution provided uses **sequential-addressing, decreasing stride**, which is a faster common variation in the Week 6 reduction lecture and supports any positive `N` with bounds protection.

The original sample assumes favorable input/block sizes and does not bounds-check the initial load. Do **not** apply that unmodified to arbitrary `N`, because the final block may read past the end of the array.

### Why use shared memory instead of repeatedly writing global memory?

Shared memory is low-latency storage per block. It allows coordination of a block's threads and reduces global-memory traffic; however, using it correctly requires synchronization. For the reduction, the main bottleneck is usually **global-memory bandwidth**, because each input element contributes little arithmetic work.

## E. Matrix multiplication: second likely problem

**Do not confuse matrix multiplication with element-wise multiplication.** For 2D matrices `A` and `B`, matrix multiplication is:

`C[row, col] = sum(A[row, k] * B[k, col] for k in range(N))`.

For:

```
A = [[1,2], [3,4]]
B = [[5,6], [7,8]]
```

The correct result is `C = [[19,22], [43,50]]`, NOT `[[5,12],[21,32]]`.

A simple, robust CUDA mapping is one GPU thread per output cell, with a **flattened index**:

```python
@cuda.jit
def matrix_multiply(a, b, c, n):
    idx = cuda.grid(1)
    step = cuda.gridsize(1)
    while idx < n * n:
        row = idx // n
        col = idx % n
        total = 0
        for k in range(n):
            total += a[row * n + k] * b[k * n + col]
        c[idx] = total
        idx += step
```

The matrices are stored as 1D row-major buffers of length `N*N`, so their original shapes can be recovered with `.reshape(N,N)` after copying to host. Validate at `N=2` or `N=4`, and then scale to `N=64`.

As further optimization (not required by the 2025 paper), tiled matrix multiplication loads `A` and `B` fragments into shared memory to reuse data, rather than repeatedly rereading global memory. If asked specifically for an optimized matrix kernel, review the effective-memory lecture and shared-memory examples.

## F. Likely mistakes that lose points

1. **Wrong decorator:** CPU JIT `@jit` is not GPU kernel `@cuda.jit`.
2. **Missing launch dimensions:** `kernel(args)` instead of `kernel[blocks, threads](args)`.
3. **Out-of-bounds:** no `i < n` guard or broken grid-stride loop.
4. **Wrong stride:** increment by `cuda.blockDim.x` rather than the entire grid `cuda.gridsize(1)`.
5. **Kernel returns a scalar:** CUDA kernels should write to device output arrays; only device functions return scalars.
6. **Race in a sum:** `out[0] += value[i]` from every thread without an atomic or reduction.
7. **Bad barrier placement:** putting `cuda.syncthreads()` in a branch that not every block thread reaches can deadlock.
8. **Reduction stage too large/too small:** shared array size must support `threads`; final stage needs a correct launch size and boundary checks.
9. **Divide by zero:** initialize `b` with nonzero values when using `/`.
10. **Stream dependency bug:** use streams for independent fill, then explicitly synchronize before dependent computation in another stream.
11. **Incorrect GPU timing:** timing a launch with a CPU clock but forgetting GPU synchronization.
12. **JIT skew:** compiling during the timed run can dominate measured time.
13. **Unnecessary transfers:** repeatedly copying results back to CPU between GPU reduction stages.
14. **Precision mismatch:** use `np.testing.assert_allclose` for floating point reductions; order of additions differs.
15. **Wrong kind of matrix product:** element-wise `A*B` instead of `A @ B`.

## G. Practice sequence (start from a blank file)

1. **5–10 min:** Write `@cuda.jit` vector addition and CPU comparison for N=10.
2. **5–10 min:** Replace the single-index kernel with `cuda.grid(1)` plus a grid-stride loop. Test N=10, 257, 10000.
3. **10 min:** Add four device functions, four GPU arithmetic kernels, and four CPU reference functions.
4. **10 min:** Add two GPU fill kernels using two independent CUDA streams. Check `b` is nonzero.
5. **15–20 min:** Implement a 256-thread reduction kernel with static shared memory and `cuda.syncthreads()`; run multiple passes until one number remains.
6. **10 min:** Add hardware query, 32×SM block count, stream sync, pinned buffers and async device-to-host transfers.
7. **5 min:** Add GPU events and CPU wall-clock timing; verify all arrays/sums; print N=10 outputs; clean memory.
8. **15–25 min:** Implement naive NxN matrix multiplication with `idx//N`, `idx%N`, inner `k` loop. Verify on a 2×2 hand example then N=64.

## H. Three mock exam questions

### Mock 1: 1D arithmetic and reduction (2026 style)

Write a **single Python file** that generates arrays `A`, `B` of `N=100000`, each value in [-5,5], with nonzero `B` values; GPU initialize using two streams; compute four arithmetic outputs using separate device functions and kernels; sum each with shared-memory reduction; compare to CPU values; print GPU name and kernel-only CPU/GPU multiplication times; use grid-stride loops and `blocks = 32*SMs`; copy results asynchronously via pinned host arrays.

Test by switching `N` to `10`, `257`, `1023`, `1024`, `1025`, `1048576`. Explain why these edge sizes are useful.

### Mock 2: matrix multiplication (2025 style)

Implement `N = 1 << 6` int32 matrix multiplication `C = A @ B` using flattened 1D device arrays. Use separate GPU fill streams, a grid-stride multiplication kernel, CPU triple loop, exact equality check, device query and event timing. Explain why a 64x64 GPU kernel may run slower than a CPU reference despite using many GPU threads.

### Mock 3: modify a reduction (transfer task)

Given an array containing negative/positive float32 values, implement sum of squares `sum(x*x)` using the Week 6 reduction scheme (not just a one-thread loop and not NumPy). Avoid out-of-bounds reads when N is not divisible by 256. Compare CPU and GPU with a tolerance. Explain the roles of shared memory and `cuda.syncthreads()`.

## I. Quick self-check questions and answers

1. **What is a warp?** Normally 32 threads scheduled/executed as a group.
2. **What is an SM?** A hardware streaming multiprocessor that schedules/executes thread blocks.
3. **What is `cuda.grid(1)`?** Global one-dimensional thread index.
4. **What is `cuda.gridsize(1)`?** Total number of threads in the launched 1D grid.
5. **When do you need a grid-stride loop?** When you want the grid to process any N, including N greater than grid thread count.
6. **Why are CUDA launches asynchronous?** Host control normally returns before GPU work finishes.
7. **What does `stream.synchronize()` do?** Wait for queued work in that CUDA stream to complete.
8. **Why use pinned host memory?** Required to reliably support asynchronous CPU-GPU DMA transfers.
9. **Why not use `cuda.syncthreads()` for grid-wide synchronization?** It synchronizes threads in one block only.
10. **Why reduce in several kernel launches?** Each block produces a partial sum; a subsequent launch combines across blocks.
11. **How many rounds are in a 256-thread tree reduction?** `log2(256)=8` reduction rounds after data are in shared memory.
12. **Why does memory coalescing matter?** Neighboring threads accessing neighboring global-memory addresses uses fewer memory transactions.
13. **What is a bank conflict?** Different shared-memory words requested by threads map to the same bank and cannot be served in parallel; padding a 2D shared tile by one column often solves transpose-related conflicts.
14. **Can a device function return a value?** Yes. The GPU kernel itself writes outputs to device arrays.
15. **How do you compare floating point sums?** `np.isclose`/`assert_allclose` with tolerances, because addition order differs.
16. **What is the 2025 matrix formula?** `C[i,j]=Σ_k A[i,k]B[k,j]`.
17. **What is the 2026 operation formula?** `C[i]=op(A[i],B[i])`, then `Σ_i C[i]`.

## J. Running the solutions

On a CUDA-enabled NVIDIA GPU with NumPy, Numba and a compatible CUDA environment:

```bash
python midterm_2026_numba_solution.py 10
python midterm_2026_numba_solution.py 1048576
python midterm_2025_matrix_numba_solution.py
```

If using Linux/Windows and a compatible Numba CUDA installation, `python -c "from numba import cuda; print(cuda.is_available())"` must report `True` for normal execution. A Mac without an NVIDIA CUDA GPU cannot run the real GPU benchmark. Numba's CPU CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`) can test core kernels, but does **not** faithfully emulate real device properties, streams, GPU event timing, concurrency or performance. The two scripts are written for a real GPU; their core kernels were validated using the simulator with small grids.

The Python CUDA examples in the uploaded tutorials may reflect their original training environment, so API availability and support should be verified in the classroom's installed Numba/CUDA version.
