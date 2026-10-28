"""Fall 2025 CSCI GPU midterm matrix multiplication translated to Numba.

Usage: python midterm_2025_matrix_numba_solution.py
A and B are logically NxN matrices stored in 1D row-major buffers.
"""

import gc
import time

import numpy as np
from numba import cuda


@cuda.jit
def fill_a(a, count):
    idx = cuda.grid(1)
    step = cuda.gridsize(1)
    while idx < count:
        a[idx] = (idx * 3) % 7 - 3
        idx += step


@cuda.jit
def fill_b(b, count):
    idx = cuda.grid(1)
    step = cuda.gridsize(1)
    while idx < count:
        b[idx] = (idx * 5) % 7 - 3
        idx += step


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


def host_multiply(a, b, n):
    c = np.zeros(n * n, dtype=np.int32)
    for row in range(n):
        for col in range(n):
            total = 0
            for k in range(n):
                total += int(a[row * n + k]) * int(b[k * n + col])
            c[row * n + col] = total
    return c


def verify(cpu, gpu):
    np.testing.assert_array_equal(cpu, gpu)
    print('Matrix multiplication: VERIFIED')


def main():
    if not cuda.is_available():
        raise RuntimeError('A CUDA-enabled NVIDIA GPU is required')
    n = 1 << 6
    count = n * n
    device = cuda.get_current_device()
    blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
    threads = 256
    print('GPU:', device.name, '| N:', n, '| blocks:', blocks, '| threads:', threads)

    stream_a = cuda.stream()
    stream_b = cuda.stream()
    stream_compute = cuda.stream()

    d_a = cuda.device_array(count, dtype=np.int32)
    d_b = cuda.device_array(count, dtype=np.int32)
    d_c = cuda.device_array(count, dtype=np.int32)

    fill_a[blocks, threads, stream_a](d_a, count)
    fill_b[blocks, threads, stream_b](d_b, count)
    stream_a.synchronize()
    stream_b.synchronize()

    # Pinned host buffers support asynchronous device-to-host migration.
    a = cuda.pinned_array(count, dtype=np.int32)
    b = cuda.pinned_array(count, dtype=np.int32)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    stream_a.synchronize()
    stream_b.synchronize()

    t0 = time.perf_counter()
    cpu_c = host_multiply(a, b, n)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    # Warm-up; JIT compilation should not be included in kernel timing.
    matrix_multiply[blocks, threads, stream_compute](d_a, d_b, d_c, n)
    stream_compute.synchronize()

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record(stream_compute)
    matrix_multiply[blocks, threads, stream_compute](d_a, d_b, d_c, n)
    end.record(stream_compute)

    c = cuda.pinned_array(count, dtype=np.int32)
    d_c.copy_to_host(c, stream=stream_compute)
    stream_compute.synchronize()
    gpu_ms = cuda.event_elapsed_time(start, end)

    verify(cpu_c, c)
    print('A (first 4x4):\n', a.reshape(n, n)[:4, :4])
    print('B (first 4x4):\n', b.reshape(n, n)[:4, :4])
    print('C (first 4x4):\n', c.reshape(n, n)[:4, :4])
    print(f'CPU multiplication: {cpu_ms:.6f} ms (Python loops)')
    print(f'GPU multiplication: {gpu_ms:.6f} ms (kernel only)')

    del d_a, d_b, d_c
    gc.collect()


if __name__ == '__main__':
    main()
