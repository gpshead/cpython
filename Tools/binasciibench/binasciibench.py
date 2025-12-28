#!/usr/bin/env python3
"""Benchmark for binascii base64 encoding and decoding performance.

This benchmark measures the throughput of base64 encoding and decoding
operations using the binascii module's C implementation.

Usage:
    python Tools/binasciibench/binasciibench.py [--iterations N] [--sizes S1,S2,...]

The benchmark tests various data sizes to help identify:
- Small message overhead (function call, buffer allocation)
- Large message throughput (core algorithm performance)
- Scaling characteristics

For testing vectorization improvements, compile Python with optimizations:
    x86-64:  CFLAGS="-O3 -march=x86-64-v3" ./configure --enable-optimizations
    ARM:     CFLAGS="-O3 -march=armv8.2-a" ./configure --enable-optimizations

NOTE: Do NOT use --with-pydebug for benchmarking as debug builds include
assertions and other overhead that would skew performance measurements.

Then compare results before and after code changes.
"""

import argparse
import binascii
import os
import statistics
import sys
import time

# Default test parameters
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 3
DEFAULT_SIZES = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

# Number of operations per timing run (adjusted per size for consistent timing)
MIN_OPS_PER_RUN = 100
TARGET_TIME_MS = 50  # Target ~50ms per measurement for accuracy


def generate_test_data(size):
    """Generate random binary data of the specified size."""
    return os.urandom(size)


def generate_base64_data(size):
    """Generate valid base64-encoded data of approximately the specified decoded size."""
    binary = os.urandom(size)
    return binascii.b2a_base64(binary, newline=False)


def benchmark_encode(data, num_ops):
    """Benchmark base64 encoding."""
    b2a = binascii.b2a_base64
    start = time.perf_counter_ns()
    for _ in range(num_ops):
        b2a(data, newline=False)
    end = time.perf_counter_ns()
    return end - start


def benchmark_decode(data, num_ops):
    """Benchmark base64 decoding."""
    a2b = binascii.a2b_base64
    start = time.perf_counter_ns()
    for _ in range(num_ops):
        a2b(data)
    end = time.perf_counter_ns()
    return end - start


def benchmark_encode_with_newline(data, num_ops):
    """Benchmark base64 encoding with newline (default behavior)."""
    b2a = binascii.b2a_base64
    start = time.perf_counter_ns()
    for _ in range(num_ops):
        b2a(data)
    end = time.perf_counter_ns()
    return end - start


def benchmark_decode_strict(data, num_ops):
    """Benchmark base64 decoding in strict mode."""
    a2b = binascii.a2b_base64
    start = time.perf_counter_ns()
    for _ in range(num_ops):
        a2b(data, strict_mode=True)
    end = time.perf_counter_ns()
    return end - start


def calibrate_ops(bench_func, data, target_time_ms):
    """Determine number of operations needed for accurate timing."""
    # Start with a small number of ops
    num_ops = MIN_OPS_PER_RUN
    elapsed_ns = bench_func(data, num_ops)
    elapsed_ms = elapsed_ns / 1_000_000

    if elapsed_ms < 1:
        # Too fast, need many more ops
        num_ops = max(MIN_OPS_PER_RUN, int(num_ops * target_time_ms / max(0.1, elapsed_ms)))
        # Re-measure to verify
        elapsed_ns = bench_func(data, num_ops)
        elapsed_ms = elapsed_ns / 1_000_000

    # Adjust to hit target time
    if elapsed_ms > 0:
        num_ops = max(MIN_OPS_PER_RUN, int(num_ops * target_time_ms / elapsed_ms))

    return num_ops


def run_benchmark(bench_func, data, num_ops, iterations, warmup):
    """Run a benchmark with warmup and multiple iterations."""
    # Warmup runs
    for _ in range(warmup):
        bench_func(data, num_ops)

    # Timed runs
    times_ns = []
    for _ in range(iterations):
        elapsed_ns = bench_func(data, num_ops)
        times_ns.append(elapsed_ns)

    return times_ns


def format_throughput(bytes_per_second):
    """Format throughput in human-readable units."""
    if bytes_per_second >= 1_000_000_000:
        return f"{bytes_per_second / 1_000_000_000:.2f} GB/s"
    elif bytes_per_second >= 1_000_000:
        return f"{bytes_per_second / 1_000_000:.2f} MB/s"
    elif bytes_per_second >= 1_000:
        return f"{bytes_per_second / 1_000:.2f} KB/s"
    else:
        return f"{bytes_per_second:.2f} B/s"


def format_size(size):
    """Format size in human-readable units."""
    if size >= 1_048_576:
        return f"{size // 1_048_576}M"
    elif size >= 1024:
        return f"{size // 1024}K"
    else:
        return str(size)


def print_results(name, size, times_ns, num_ops, data_size):
    """Print benchmark results."""
    # Calculate statistics
    times_per_op_ns = [t / num_ops for t in times_ns]
    mean_ns = statistics.mean(times_per_op_ns)
    stdev_ns = statistics.stdev(times_per_op_ns) if len(times_per_op_ns) > 1 else 0

    # Calculate throughput
    bytes_per_ns = data_size / mean_ns
    bytes_per_second = bytes_per_ns * 1_000_000_000
    throughput = format_throughput(bytes_per_second)

    # Calculate coefficient of variation
    cv = (stdev_ns / mean_ns * 100) if mean_ns > 0 else 0

    size_str = format_size(size)
    print(f"{name:<20} {size_str:>8}  {mean_ns:>12.1f} ns  "
          f"(+/- {cv:>5.1f}%)  {throughput:>12}")


def run_all_benchmarks(sizes, iterations, warmup):
    """Run all benchmark variants for all sizes."""
    print(f"binascii base64 benchmark")
    print(f"Python: {sys.version}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    print()
    print(f"{'Benchmark':<20} {'Size':>8}  {'Time/op':>15}  "
          f"{'Variance':>10}  {'Throughput':>12}")
    print("-" * 75)

    for size in sizes:
        # Generate test data
        binary_data = generate_test_data(size)
        base64_data = generate_base64_data(size)

        # Benchmark encode
        num_ops = calibrate_ops(benchmark_encode, binary_data, TARGET_TIME_MS)
        times = run_benchmark(benchmark_encode, binary_data, num_ops,
                              iterations, warmup)
        print_results("b2a_base64", size, times, num_ops, size)

        # Benchmark encode with newline
        num_ops = calibrate_ops(benchmark_encode_with_newline, binary_data,
                                TARGET_TIME_MS)
        times = run_benchmark(benchmark_encode_with_newline, binary_data,
                              num_ops, iterations, warmup)
        print_results("b2a_base64(newline)", size, times, num_ops, size)

        # Benchmark decode
        num_ops = calibrate_ops(benchmark_decode, base64_data, TARGET_TIME_MS)
        times = run_benchmark(benchmark_decode, base64_data, num_ops,
                              iterations, warmup)
        print_results("a2b_base64", size, times, num_ops, size)

        # Benchmark decode strict
        num_ops = calibrate_ops(benchmark_decode_strict, base64_data,
                                TARGET_TIME_MS)
        times = run_benchmark(benchmark_decode_strict, base64_data, num_ops,
                              iterations, warmup)
        print_results("a2b_base64(strict)", size, times, num_ops, size)

        print()


def run_scaling_analysis(iterations, warmup):
    """Analyze how performance scales with data size."""
    print("\nScaling Analysis")
    print("=" * 75)
    print("Measuring bytes processed per nanosecond at different sizes")
    print("(Higher is better, ideal scaling shows constant rate for large sizes)")
    print()

    sizes = [2**i for i in range(4, 21)]  # 16 bytes to 1MB

    encode_rates = []
    decode_rates = []

    for size in sizes:
        binary_data = generate_test_data(size)
        base64_data = generate_base64_data(size)

        # Encode benchmark
        num_ops = calibrate_ops(benchmark_encode, binary_data, TARGET_TIME_MS)
        times = run_benchmark(benchmark_encode, binary_data, num_ops,
                              iterations, warmup)
        mean_ns = statistics.mean(t / num_ops for t in times)
        encode_rate = size / mean_ns
        encode_rates.append((size, encode_rate))

        # Decode benchmark
        num_ops = calibrate_ops(benchmark_decode, base64_data, TARGET_TIME_MS)
        times = run_benchmark(benchmark_decode, base64_data, num_ops,
                              iterations, warmup)
        mean_ns = statistics.mean(t / num_ops for t in times)
        decode_rate = size / mean_ns
        decode_rates.append((size, decode_rate))

    print(f"{'Size':>10}  {'Encode (B/ns)':>15}  {'Decode (B/ns)':>15}")
    print("-" * 45)
    for i, size in enumerate(sizes):
        print(f"{format_size(size):>10}  {encode_rates[i][1]:>15.3f}  "
              f"{decode_rates[i][1]:>15.3f}")

    # Report peak rates
    print()
    peak_encode = max(r[1] for r in encode_rates)
    peak_decode = max(r[1] for r in decode_rates)
    print(f"Peak encode rate: {format_throughput(peak_encode * 1e9)}")
    print(f"Peak decode rate: {format_throughput(peak_decode * 1e9)}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark binascii base64 encoding and decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of timed iterations (default: {DEFAULT_ITERATIONS})"
    )
    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "-s", "--sizes",
        type=str,
        default=None,
        help="Comma-separated list of sizes to test (e.g., '64,256,1024')"
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling analysis across many sizes"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer iterations and sizes"
    )

    args = parser.parse_args()

    if args.quick:
        sizes = [64, 1024, 65536]
        iterations = 3
        warmup = 1
    else:
        if args.sizes:
            sizes = [int(s.strip()) for s in args.sizes.split(",")]
        else:
            sizes = DEFAULT_SIZES
        iterations = args.iterations
        warmup = args.warmup

    run_all_benchmarks(sizes, iterations, warmup)

    if args.scaling:
        run_scaling_analysis(iterations, warmup)


if __name__ == "__main__":
    main()
