# binascii Base64 Benchmark

This directory contains a benchmark tool for measuring the performance of
the `binascii` module's base64 encoding and decoding functions.

## Usage

```bash
# Default benchmark (64, 1K, 64K, 1M sizes)
python Tools/binasciibench/binasciibench.py

# Custom sizes
python Tools/binasciibench/binasciibench.py --sizes 64,1024,65536

# Scaling analysis across many sizes
python Tools/binasciibench/binasciibench.py --scaling
```

## Vectorization Optimization

The base64 encoding/decoding in `Modules/binascii.c` has been optimized by
restructuring the loops to eliminate loop-carried dependencies, enabling
better compiler optimization.

### The Problem with the Original Code

The original encoding loop accumulated state across iterations:

```c
// Original code - has loop-carried dependencies
for (; bin_len > 0; bin_len--, bin_data++) {
    leftchar = (leftchar << 8) | *bin_data;  // Depends on previous iteration
    leftbits += 8;                            // Depends on previous iteration
    while (leftbits >= 6) {
        this_ch = (leftchar >> (leftbits-6)) & 0x3f;
        leftbits -= 6;
        *ascii_data++ = table_b2a_base64[this_ch];
    }
}
```

This pattern prevents the compiler from:
1. Unrolling the loop effectively
2. Reordering memory operations
3. Using instruction-level parallelism

### The Optimized Approach

The new code processes complete 3-byte groups (which produce exactly 4 base64
characters) without any loop-carried state:

```c
// Optimized code - each iteration is independent
static inline void
base64_encode_trio(const unsigned char *in, unsigned char *out,
                   const unsigned char *table)
{
    // Combine 3 bytes into a 24-bit value
    unsigned int combined = ((unsigned int)in[0] << 16) |
                            ((unsigned int)in[1] << 8) |
                            (unsigned int)in[2];

    // Extract four 6-bit groups - all independent operations
    out[0] = table[(combined >> 18) & 0x3f];
    out[1] = table[(combined >> 12) & 0x3f];
    out[2] = table[(combined >> 6) & 0x3f];
    out[3] = table[combined & 0x3f];
}

// Main loop - each iteration processes one complete group
for (i = 0; i < n_trios; i++) {
    base64_encode_trio(in + i * 3, out + i * 4, table);
}
```

### Why This Is Faster

1. **No Loop-Carried Dependencies**: Each iteration is completely independent,
   allowing the CPU to execute multiple iterations in parallel via
   instruction-level parallelism (ILP).

2. **Better Memory Access Pattern**: The compiler can see that we read 3 bytes
   and write 4 bytes per iteration, enabling better prefetching and pipelining.

3. **Table Lookups Remain Fast**: We keep the 64-byte lookup table (fits in L1
   cache) rather than trying arithmetic conversion which has unpredictable
   branches.

4. **Predictable Branch Behavior**: The main loop has a single predictable
   branch (the loop condition), unlike the original's inner while loop.

## Performance Results

Measured on Intel Icelake-server (x86-64-v4 with AVX-512) with GCC 13.3.0,
`-O3 -march=native`:

### Scalar Optimization (loop restructuring)

| Operation       | Size | Baseline      | Scalar Opt    | Speedup |
|-----------------|------|---------------|---------------|---------|
| b2a_base64      | 64   | 347 MB/s      | 391 MB/s      | 1.13x   |
| a2b_base64      | 64   | 297 MB/s      | 730 MB/s      | 2.46x   |
| b2a_base64      | 64K  | 1.10 GB/s     | 1.91 GB/s     | 1.74x   |
| a2b_base64      | 64K  | 393 MB/s      | 1.53 GB/s     | 3.89x   |

### AVX-512 VBMI SIMD (on CPUs with VBMI support)

CPUs with AVX-512 VBMI (Icelake, Zen4, etc.) get additional acceleration.
Small buffers use the scalar path; SIMD kicks in at ≥48 bytes (encode) or
≥64 bytes (decode):

| Operation       | Size | Scalar Opt    | AVX-512 VBMI  | Speedup |
|-----------------|------|---------------|---------------|---------|
| b2a_base64      | 64   | 391 MB/s      | 433 MB/s      | 1.11x   |
| a2b_base64      | 64   | 730 MB/s      | 918 MB/s      | 1.26x   |
| b2a_base64      | 64K  | 1.91 GB/s     | 22.4 GB/s     | 11.7x   |
| a2b_base64      | 64K  | 1.53 GB/s     | 19.4 GB/s     | 12.7x   |

The SIMD implementation uses:
- `vpermb` for byte reshuffling across all 64 bytes
- `vpmaddubsw`/`vpmaddwd` for 6-bit packing/unpacking
- `vpermb` again for the final table lookup (encoding) or byte packing (decoding)

## Generated Assembly Analysis

The scalar optimization produces efficient code even without explicit SIMD.
GCC 13 with `-O3 -march=native` generates a tight loop that processes one
3-byte group per iteration, interleaving independent operations for good
instruction-level parallelism. The compiler pre-increments pointers early
to allow memory prefetching, and uses indexed addressing for output writes
that coalesce in the store buffer. The core loop is branch-free except for
the loop condition itself.

On CPUs with AVX-512 VBMI, the SIMD path takes over for buffers ≥48 bytes.
Each iteration processes 48 input bytes to 64 output characters (encoding)
or vice versa (decoding), using just a handful of vector instructions:
one `vpermb` for input reshuffling, multiply-add for bit manipulation,
and a final `vpermb` for either the 64-entry table lookup (encoding) or
byte packing (decoding).
