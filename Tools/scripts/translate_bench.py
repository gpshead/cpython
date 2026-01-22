"""
translate() performance benchmark for str, bytes, and bytearray.

This benchmark measures the performance of translate() methods under various
conditions to help evaluate optimizations.

To run:
    python Tools/scripts/translate_bench.py

Options:
    --sizes         Comma-separated list of string sizes to test
    --iterations    Number of iterations per benchmark
    --quiet         Only show summary results
    --str-only      Only benchmark str.translate()
    --bytes-only    Only benchmark bytes/bytearray.translate()

The benchmark tests several categories:

str.translate():
1. SIMD-accelerated cases (ASCII, no deletions, table populated):
   - ROT13 cipher
   - Lowercase to uppercase
   - Identity mapping (no-op)

2. Non-accelerated cases:
   - Character deletions
   - Sparse tables (few translations)
   - Non-ASCII input

bytes/bytearray.translate():
1. No-deletion fast path (unrolled loop with pipelining):
   - Full byte rotation
   - Identity mapping

2. Deletion cases:
   - Delete specific bytes
"""

from __future__ import annotations

import argparse
import sys
import time


def make_rot13_table() -> dict[int, int]:
    """ROT13 cipher - rotates letters by 13 positions."""
    table = {}
    for i in range(26):
        table[ord('a') + i] = ord('a') + (i + 13) % 26
        table[ord('A') + i] = ord('A') + (i + 13) % 26
    return table


def make_uppercase_table() -> dict[int, int]:
    """Lowercase to uppercase conversion."""
    return {ord('a') + i: ord('A') + i for i in range(26)}


def make_identity_table() -> dict[int, int]:
    """Identity mapping - characters map to themselves."""
    return {i: i for i in range(128)}


def make_delete_vowels_table() -> dict[int, None]:
    """Delete all vowels - tests deletion handling."""
    return {ord(c): None for c in 'aeiouAEIOU'}


def make_sparse_table() -> dict[int, int]:
    """Only translate a few characters."""
    return {ord('<'): ord('['), ord('>'): ord(']'), ord('&'): ord('+')}


def make_dense_table() -> dict[int, int]:
    """Translate many characters (pseudo-random permutation)."""
    return {i: (i * 7 + 13) % 128 for i in range(128)}


def generate_ascii_lowercase(length: int) -> str:
    """Generate repeating lowercase letters."""
    base = 'abcdefghijklmnopqrstuvwxyz'
    return (base * ((length // 26) + 1))[:length]


def generate_ascii_mixed(length: int) -> str:
    """Generate mixed printable ASCII."""
    base = 'The quick brown fox jumps over the lazy dog. 0123456789!@#$%'
    return (base * ((length // len(base)) + 1))[:length]


def generate_with_non_ascii(length: int) -> str:
    """Generate ASCII with some non-ASCII characters mixed in."""
    base = 'Hello world with émojis and ñ special chars: café, naïve'
    return (base * ((length // len(base)) + 1))[:length]


def generate_sparse_input(length: int) -> str:
    """Generate input with few characters needing translation."""
    base = 'This has <some> special & characters to translate'
    return (base * ((length // len(base)) + 1))[:length]


# --- bytes/bytearray helpers ---

def make_bytes_rot_table() -> bytes:
    """Full byte rotation by 13 positions (like ROT13 for all bytes)."""
    return bytes((b + 13) % 256 for b in range(256))


def make_bytes_identity_table() -> bytes:
    """Identity mapping - bytes map to themselves."""
    return bytes(range(256))


def make_bytes_swap_table() -> bytes:
    """Swap high/low nibbles of each byte."""
    return bytes(((b >> 4) | ((b & 0x0F) << 4)) for b in range(256))


def generate_bytes_data(length: int) -> bytes:
    """Generate repeating byte pattern."""
    base = bytes(range(256))
    return (base * ((length // 256) + 1))[:length]


def generate_ascii_bytes(length: int) -> bytes:
    """Generate ASCII-only bytes (printable range)."""
    base = b'The quick brown fox jumps over the lazy dog. 0123456789!@#'
    return (base * ((length // len(base)) + 1))[:length]


def benchmark_bytes_translate(
    data: bytes,
    table: bytes,
    delete: bytes | None,
    iterations: int,
    warmup: int = 5
) -> tuple[float, int]:
    """Benchmark bytes.translate() and return (ns_per_call, output_length)."""
    # Warmup
    for _ in range(warmup):
        result = data.translate(table, delete) if delete else data.translate(table)

    # Timed run
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = data.translate(table, delete) if delete else data.translate(table)
    end = time.perf_counter_ns()

    return (end - start) / iterations, len(result)


def benchmark_bytearray_translate(
    data: bytearray,
    table: bytes,
    delete: bytes | None,
    iterations: int,
    warmup: int = 5
) -> tuple[float, int]:
    """Benchmark bytearray.translate() and return (ns_per_call, output_length)."""
    # Warmup
    for _ in range(warmup):
        result = data.translate(table, delete) if delete else data.translate(table)

    # Timed run
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = data.translate(table, delete) if delete else data.translate(table)
    end = time.perf_counter_ns()

    return (end - start) / iterations, len(result)


def benchmark_translate(
    test_string: str,
    table: dict,
    iterations: int,
    warmup: int = 5
) -> tuple[float, int]:
    """
    Benchmark str.translate() and return (ns_per_call, output_length).
    """
    # Pre-create the translation table
    trans_table = str.maketrans(table)

    # Warmup
    for _ in range(warmup):
        result = test_string.translate(trans_table)

    # Timed run
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = test_string.translate(trans_table)
    end = time.perf_counter_ns()

    return (end - start) / iterations, len(result)


def format_throughput(ns_per_call: float, input_len: int) -> str:
    """Format throughput in GB/s."""
    if ns_per_call > 0:
        gbps = input_len / ns_per_call  # bytes per ns = GB/s
        return f"{gbps:.2f}"
    return "N/A"


def run_benchmark_suite(
    sizes: list[int],
    iterations: int,
    quiet: bool = False
) -> dict:
    """Run the full benchmark suite."""

    results = {}

    # Define benchmark categories
    benchmarks = [
        # Category: SIMD-accelerated (ASCII, no deletions)
        ("ROT13 (SIMD path)", make_rot13_table(), generate_ascii_lowercase),
        ("Uppercase (SIMD path)", make_uppercase_table(), generate_ascii_lowercase),
        ("Identity (SIMD path)", make_identity_table(), generate_ascii_lowercase),
        ("Dense table (SIMD path)", make_dense_table(), generate_ascii_lowercase),

        # Category: Non-accelerated
        ("Delete vowels (deletions)", make_delete_vowels_table(), generate_ascii_lowercase),
        ("Sparse table (few chars)", make_sparse_table(), generate_sparse_input),
        ("Non-ASCII input", make_rot13_table(), generate_with_non_ascii),

        # Category: Mixed content
        ("Mixed ASCII content", make_rot13_table(), generate_ascii_mixed),
    ]

    for name, table, generator in benchmarks:
        results[name] = {}

        if not quiet:
            print(f"\n{name}")
            print("-" * 60)
            print(f"{'Size':>10} {'ns/call':>12} {'GB/s':>10} {'Out len':>10}")
            print("-" * 60)

        for size in sizes:
            test_string = generator(size)
            # Adjust iterations for larger sizes
            iters = max(100, iterations * 1000 // size)

            ns_per_call, out_len = benchmark_translate(test_string, table, iters)
            throughput = format_throughput(ns_per_call, len(test_string))

            results[name][size] = {
                'ns_per_call': ns_per_call,
                'throughput_gbps': float(throughput) if throughput != "N/A" else 0,
                'output_len': out_len,
                'input_len': len(test_string),
            }

            if not quiet:
                print(f"{size:>10} {ns_per_call:>12.1f} {throughput:>10} {out_len:>10}")

    return results


def run_bytes_benchmark_suite(
    sizes: list[int],
    iterations: int,
    quiet: bool = False
) -> dict:
    """Run the bytes/bytearray benchmark suite."""

    results = {}

    # bytes.translate benchmarks
    bytes_benchmarks = [
        ("bytes: full rotation (no del)", make_bytes_rot_table(), None, generate_bytes_data),
        ("bytes: nibble swap (no del)", make_bytes_swap_table(), None, generate_bytes_data),
        ("bytes: identity (no del)", make_bytes_identity_table(), None, generate_bytes_data),
        ("bytes: ASCII data (no del)", make_bytes_rot_table(), None, generate_ascii_bytes),
        ("bytes: with deletions", make_bytes_identity_table(), b'\x00\x10\x20\x30', generate_bytes_data),
    ]

    for name, table, delete, generator in bytes_benchmarks:
        results[name] = {}

        if not quiet:
            print(f"\n{name}")
            print("-" * 60)
            print(f"{'Size':>10} {'ns/call':>12} {'GB/s':>10} {'Out len':>10}")
            print("-" * 60)

        for size in sizes:
            data = generator(size)
            iters = max(100, iterations * 1000 // size)

            ns_per_call, out_len = benchmark_bytes_translate(data, table, delete, iters)
            throughput = format_throughput(ns_per_call, len(data))

            results[name][size] = {
                'ns_per_call': ns_per_call,
                'throughput_gbps': float(throughput) if throughput != "N/A" else 0,
                'output_len': out_len,
                'input_len': len(data),
            }

            if not quiet:
                print(f"{size:>10} {ns_per_call:>12.1f} {throughput:>10} {out_len:>10}")

    # bytearray.translate benchmarks
    bytearray_benchmarks = [
        ("bytearray: full rotation", make_bytes_rot_table(), None, generate_bytes_data),
        ("bytearray: with deletions", make_bytes_identity_table(), b'\x00\x10\x20\x30', generate_bytes_data),
    ]

    for name, table, delete, generator in bytearray_benchmarks:
        results[name] = {}

        if not quiet:
            print(f"\n{name}")
            print("-" * 60)
            print(f"{'Size':>10} {'ns/call':>12} {'GB/s':>10} {'Out len':>10}")
            print("-" * 60)

        for size in sizes:
            data = bytearray(generator(size))
            iters = max(100, iterations * 1000 // size)

            ns_per_call, out_len = benchmark_bytearray_translate(data, table, delete, iters)
            throughput = format_throughput(ns_per_call, len(data))

            results[name][size] = {
                'ns_per_call': ns_per_call,
                'throughput_gbps': float(throughput) if throughput != "N/A" else 0,
                'output_len': out_len,
                'input_len': len(data),
            }

            if not quiet:
                print(f"{size:>10} {ns_per_call:>12.1f} {throughput:>10} {out_len:>10}")

    return results


def print_summary(results: dict, sizes: list[int]) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("SUMMARY: Throughput (GB/s) by string size")
    print("=" * 70)

    # Header
    header = f"{'Benchmark':<30}"
    for size in sizes:
        if size >= 1024:
            header += f" {size//1024:>6}KB"
        else:
            header += f" {size:>6}B"
    print(header)
    print("-" * 70)

    # Data rows
    for name, size_results in results.items():
        row = f"{name:<30}"
        for size in sizes:
            if size in size_results:
                tp = size_results[size]['throughput_gbps']
                row += f" {tp:>7.2f}"
            else:
                row += f" {'N/A':>7}"
        print(row)

    print("=" * 70)


def verify_correctness(test_str: bool = True, test_bytes: bool = True) -> bool:
    """Verify that translations produce correct results."""
    print("Verifying correctness...")

    if test_str:
        # Test ROT13
        table = str.maketrans(make_rot13_table())
        result = "Hello World".translate(table)
        expected = "Uryyb Jbeyq"
        if result != expected:
            print(f"FAIL: str ROT13 '{result}' != '{expected}'")
            return False

        # Test uppercase
        table = str.maketrans(make_uppercase_table())
        result = "hello world".translate(table)
        expected = "HELLO WORLD"
        if result != expected:
            print(f"FAIL: str Uppercase '{result}' != '{expected}'")
            return False

        # Test deletion
        table = str.maketrans(make_delete_vowels_table())
        result = "hello world".translate(table)
        expected = "hll wrld"
        if result != expected:
            print(f"FAIL: str Delete vowels '{result}' != '{expected}'")
            return False

        # Test sparse
        table = str.maketrans(make_sparse_table())
        result = "<hello> & world".translate(table)
        expected = "[hello] + world"
        if result != expected:
            print(f"FAIL: str Sparse '{result}' != '{expected}'")
            return False

    if test_bytes:
        # Test bytes rotation
        table = make_bytes_rot_table()
        data = bytes(range(16))
        result = data.translate(table)
        expected = bytes((b + 13) % 256 for b in range(16))
        if result != expected:
            print(f"FAIL: bytes rotation {result!r} != {expected!r}")
            return False

        # Test bytes identity
        table = make_bytes_identity_table()
        data = b"hello world"
        result = data.translate(table)
        if result != data:
            print(f"FAIL: bytes identity {result!r} != {data!r}")
            return False

        # Test bytes with deletion
        data = bytes(range(64))
        result = data.translate(None, b'\x00\x10\x20\x30')
        expected = bytes(b for b in range(64) if b not in (0, 16, 32, 48))
        if result != expected:
            print(f"FAIL: bytes deletion {result!r} != {expected!r}")
            return False

        # Test bytearray rotation
        table = make_bytes_rot_table()
        data = bytearray(range(16))
        result = data.translate(table)
        expected = bytearray((b + 13) % 256 for b in range(16))
        if result != expected:
            print(f"FAIL: bytearray rotation {result!r} != {expected!r}")
            return False

    print("All correctness tests passed.\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark translate() performance for str, bytes, bytearray"
    )
    parser.add_argument(
        '--sizes',
        type=str,
        default='64,100,127,256,500,1000,1024,4096,16384,65536,262144',
        help='Comma-separated list of string sizes to test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10000,
        help='Base number of iterations (adjusted by size)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary results'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip correctness verification'
    )
    parser.add_argument(
        '--str-only',
        action='store_true',
        help='Only benchmark str.translate()'
    )
    parser.add_argument(
        '--bytes-only',
        action='store_true',
        help='Only benchmark bytes/bytearray.translate()'
    )

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    run_str = not args.bytes_only
    run_bytes = not args.str_only

    print(f"translate() Benchmark")
    print(f"Python: {sys.version}")
    print(f"Sizes: {sizes}")
    print(f"Base iterations: {args.iterations}")
    types = []
    if run_str:
        types.append("str")
    if run_bytes:
        types.append("bytes")
        types.append("bytearray")
    print(f"Types: {', '.join(types)}")
    print("=" * 60)

    if not args.no_verify:
        if not verify_correctness(test_str=run_str, test_bytes=run_bytes):
            sys.exit(1)

    all_results = {}

    if run_str:
        print("\n" + "=" * 60)
        print("str.translate()")
        print("=" * 60)
        str_results = run_benchmark_suite(sizes, args.iterations, args.quiet)
        all_results.update(str_results)
        print_summary(str_results, sizes)

    if run_bytes:
        print("\n" + "=" * 60)
        print("bytes/bytearray.translate()")
        print("=" * 60)
        bytes_results = run_bytes_benchmark_suite(sizes, args.iterations, args.quiet)
        all_results.update(bytes_results)
        print_summary(bytes_results, sizes)

    # Print interpretation notes
    print("\nNotes:")
    if run_str:
        print("- str 'SIMD path' benchmarks use vectorized nibble-split lookup")
        print("- str 'Delete vowels' cannot use SIMD (output length changes)")
        print("- str 'Non-ASCII input' falls back to the slow path")
    if run_bytes:
        print("- bytes/bytearray use unrolled loop for better pipelining")
        print("- bytes/bytearray deletion cases use the slower path")
    print("- Sizes like 100, 127, 500 test non-aligned lengths")


if __name__ == '__main__':
    main()
