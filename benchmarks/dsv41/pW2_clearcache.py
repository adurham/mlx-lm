#!/usr/bin/env python3
"""pW2 -- does mx.clear_cache() evict compiled graphs / JIT metal kernels?

If yes, the prefill driver's periodic clear_cache() (every 4 chunks) destroys
exactly the decode-side artifacts warmup() paid for, and the first decode step
recompiles them -> the 1.3-1.9 s spike. Tiny, allocation-light probe.
"""
import time
import mlx.core as mx

print("mlx", getattr(mx, "__version__", "?"), flush=True)


@mx.compile
def f(a, b):
    x = a
    for _ in range(40):
        x = mx.tanh(x * 1.0001 + b) + mx.sin(x) * 0.5
    return x.sum(axis=-1)


@mx.compile
def g(a, b):
    # a second, differently-shaped compiled fn (as the model has many)
    return (a @ b.T).sum(axis=-1)


def timeit(fn, *args, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        mx.eval(r)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


a = mx.random.normal((1, 5120)).astype(mx.bfloat16)
b = mx.random.normal((1, 5120)).astype(mx.bfloat16)
w = mx.random.normal((5120, 5120)).astype(mx.bfloat16)

print("f cold :", [f"{t:.1f}" for t in timeit(f, a, b)], flush=True)
print("f warm :", [f"{t:.1f}" for t in timeit(f, a, b)], flush=True)
print("g cold :", [f"{t:.1f}" for t in timeit(g, a, w)], flush=True)
print("g warm :", [f"{t:.1f}" for t in timeit(g, a, w)], flush=True)

mx.clear_cache()
print("after clear_cache():", flush=True)
print("f      :", [f"{t:.1f}" for t in timeit(f, a, b)], flush=True)
print("g      :", [f"{t:.1f}" for t in timeit(g, a, w)], flush=True)

mx.clear_compile_cache()
print("after clear_compile_cache():", flush=True)
print("f      :", [f"{t:.1f}" for t in timeit(f, a, b)], flush=True)
print("g      :", [f"{t:.1f}" for t in timeit(g, a, w)], flush=True)

# metal_kernel JIT: does it survive clear_cache?
src = """
    uint tid = thread_position_in_grid.x;
    out[tid] = (T)(inp[tid] * 2.0f);
"""
k = mx.fast.metal_kernel(name="pW2_double", input_names=["inp"], output_names=["out"], source=src)
big = mx.zeros((1 << 20,), dtype=mx.float16)


def run_k():
    return k(inputs=[big], template=[("T", mx.float16)],
             grid=(big.size, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[big.shape], output_dtypes=[mx.float16])[0]


print("metal cold:", [f"{t:.1f}" for t in timeit(run_k)], flush=True)
print("metal warm:", [f"{t:.1f}" for t in timeit(run_k)], flush=True)
mx.clear_cache()
print("metal after clear_cache:", [f"{t:.1f}" for t in timeit(run_k)], flush=True)

print("PW2_DONE", flush=True)
