"""CSCI 723 (2026) practice solution, translated to Numba CUDA.

Usage (on a computer with NVIDIA CUDA GPU and Numba CUDA):
    python midterm_2026_numba_solution.py 10
    python midterm_2026_numba_solution.py 1048576

The exam specifies a CUDA C++ submission. This is a Python/Numba STUDY
translation, not a claim that Python submissions are permitted.
"""

import gc
import sys
import time

import numpy as np
from numba import cuda, float32


# Four GPU-only functions, one per arithmetic operation.
@cuda.jit(device=True)
def add_device(x, y):
    return x + y


@cuda.jit(device=True)
def subtract_device(x, y):
    return x - y


@cuda.jit(device=True)
def multiply_device(x, y):
    return x * y


@cuda.jit(device=True)
def divide_device(x, y):
    return x / y


# The midterm explicitly asks to INITIALIZE the arrays using GPU kernels,
# with a separate CUDA stream for each initialization.
@cuda.jit
def fill_a(a, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        a[i] = (i % 11) - 5                 # -5 through +5
        i += step


@cuda.jit
def fill_b(b, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        value = ((i * 7 + 3) % 11) - 5    # -5 through +5
        if value == 0:
            value = 1                     # Avoid division by zero.
        b[i] = value
        i += step


@cuda.jit
def add_kernel(a, b, out, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        out[i] = add_device(a[i], b[i])
        i += step


@cuda.jit
def subtract_kernel(a, b, out, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        out[i] = subtract_device(a[i], b[i])
        i += step


@cuda.jit
def multiply_kernel(a, b, out, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        out[i] = multiply_device(a[i], b[i])
        i += step


@cuda.jit
def divide_kernel(a, b, out, n):
    i = cuda.grid(1)
    step = cuda.gridsize(1)
    while i < n:
        out[i] = divide_device(a[i], b[i])
        i += step


@cuda.jit
def reduce_kernel(values, partial_sums, n):
    """Tree reduction: grid-stride accumulation + block-local shared memory.

    Each block writes ONE partial sum. The CPU launches this same kernel
    repeatedly on its partial sums until only one number remains.
    The block size MUST be 256 for this static shared-memory allocation.
    """
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    step = cuda.gridsize(1)

    local_sum = float32(0.0)
    while i < n:
        local_sum += values[i]
        i += step

    shared = cuda.shared.array(shape=256, dtype=float32)
    shared[tid] = local_sum
    cuda.syncthreads()                 # All threads have finished writing.

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()             # All threads finish this reduction step.
        stride //= 2

    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]


def gpu_sum(values, n, max_blocks, stream):
    """Launch as many reduction stages as needed, without host copies."""
    current = values
    intermediates = []  # Retain device buffers until all async work completes.
    size = n
    threads = 256
    while size > 1:
        # Reduction is intentionally allowed to use fewer blocks than the
        # fixed 32 * SM count requested for the element-wise kernels.
        blocks = min(max_blocks, (size + threads - 1) // threads)
        partial = cuda.device_array(blocks, dtype=np.float32, stream=stream)
        intermediates.append(partial)
        reduce_kernel[blocks, threads, stream](current, partial, size)
        current = partial
        size = blocks
    return current, intermediates


# Four CPU reference implementations (NumPy uses optimized CPU loops).
def add_host(a, b):
    return a + b


def subtract_host(a, b):
    return a - b


def multiply_host(a, b):
    return a * b


def divide_host(a, b):
    return a / b


def verify(cpu_results, gpu_results, cpu_sums, gpu_sums):
    for name in cpu_results:
        np.testing.assert_allclose(
            gpu_results[name], cpu_results[name], rtol=1e-5, atol=1e-5
        )
        # Reduction changes summation order, so float sums need tolerance.
        np.testing.assert_allclose(
            gpu_sums[name], cpu_sums[name], rtol=1e-3, atol=1e-2
        )
    print('CPU/GPU results and reductions: VERIFIED')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if n < 1:
        raise ValueError('N must be positive')
    if not cuda.is_available():
        raise RuntimeError('A CUDA-enabled NVIDIA GPU is required')

    device = cuda.get_current_device()
    warp = device.WARP_SIZE
    sm_count = device.MULTIPROCESSOR_COUNT
    threads = 256                 # Good starting point, NOT universally optimal.
    blocks = warp * sm_count      # EXACT launch-grid requirement in exam.

    print('GPU:', device.name)
    print('SMs:', sm_count, '| warp:', warp, '| blocks:', blocks, '| threads:', threads)
    print('N:', n)

    stream_a = cuda.stream()
    stream_b = cuda.stream()
    stream_compute = cuda.stream()

    d_a = cuda.device_array(n, dtype=np.float32)
    d_b = cuda.device_array(n, dtype=np.float32)

    # Independent stream initialization.
    fill_a[blocks, threads, stream_a](d_a, n)
    fill_b[blocks, threads, stream_b](d_b, n)
    stream_a.synchronize()
    stream_b.synchronize()

    # Page-locked buffers enable genuinely async device-to-host copies.
    a = cuda.pinned_array(n, dtype=np.float32)
    b = cuda.pinned_array(n, dtype=np.float32)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    stream_a.synchronize()
    stream_b.synchronize()

    # CPU reference and CPU multiply timing.
    cpu_results = {
        'addition': add_host(a, b),
        'subtraction': subtract_host(a, b),
    }
    t0 = time.perf_counter()
    cpu_results['multiplication'] = multiply_host(a, b)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    cpu_results['division'] = divide_host(a, b)
    cpu_sums = {key: float(np.sum(val, dtype=np.float64))
                for key, val in cpu_results.items()}

    d_results = {key: cuda.device_array(n, dtype=np.float32)
                 for key in ('addition', 'subtraction', 'multiplication', 'division')}

    # Warm-up to exclude one-time JIT compilation from the timed GPU kernel.
    multiply_kernel[blocks, threads, stream_compute](
        d_a, d_b, d_results['multiplication'], n
    )
    stream_compute.synchronize()

    add_kernel[blocks, threads, stream_compute](d_a, d_b, d_results['addition'], n)
    subtract_kernel[blocks, threads, stream_compute](d_a, d_b, d_results['subtraction'], n)

    start = cuda.event(timing=True)
    stop = cuda.event(timing=True)
    start.record(stream_compute)
    multiply_kernel[blocks, threads, stream_compute](
        d_a, d_b, d_results['multiplication'], n
    )
    stop.record(stream_compute)

    divide_kernel[blocks, threads, stream_compute](d_a, d_b, d_results['division'], n)

    reduction_jobs = {key: gpu_sum(val, n, blocks, stream_compute)
                      for key, val in d_results.items()}
    d_sums = {key: job[0] for key, job in reduction_jobs.items()}

    # Pinned host results allow asynchronous device-to-host transfer.
    gpu_results = {key: cuda.pinned_array(n, dtype=np.float32)
                   for key in d_results}
    gpu_sum_buffers = {key: cuda.pinned_array(1, dtype=np.float32)
                       for key in d_results}
    for key in d_results:
        d_results[key].copy_to_host(gpu_results[key], stream=stream_compute)
        d_sums[key].copy_to_host(gpu_sum_buffers[key], stream=stream_compute)

    stream_compute.synchronize()             # Wait before checking GPU results.
    gpu_ms = cuda.event_elapsed_time(start, stop)
    gpu_sums = {key: float(buffer[0]) for key, buffer in gpu_sum_buffers.items()}

    verify(cpu_results, gpu_results, cpu_sums, gpu_sums)
    if n <= 10:
        print('\nA =', a)
        print('B =', b)
    for key in cpu_results:
        shown = cpu_results[key] if n <= 10 else cpu_results[key][:10]
        print('CPU', key, '=', shown, '| sum =', cpu_sums[key])
    for key in gpu_results:
        shown = gpu_results[key] if n <= 10 else gpu_results[key][:10]
        print('GPU', key, '=', shown, '| sum =', gpu_sums[key])
    print(f'CPU multiplication: {cpu_ms:.6f} ms (NumPy only)')
    print(f'GPU multiplication: {gpu_ms:.6f} ms (kernel only, no copies)')

    # Device allocations are managed by Numba; drop references when done.
    del d_a, d_b, d_results, d_sums, reduction_jobs
    gc.collect()


if __name__ == '__main__':
    main()
