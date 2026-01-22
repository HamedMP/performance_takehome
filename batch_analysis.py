#!/usr/bin/env python3
"""
Analyze different batch size strategies for selection phases.
"""

def analyze_batch_strategies():
    n_vectors = 32  # Total vectors to process

    print("=" * 70)
    print("BATCH SIZE STRATEGY ANALYSIS")
    print("=" * 70)

    # Current strategy: 6-vector batches
    print("\n=== Current: 6-vector batches ===")
    batches_6 = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 6), (30, 2)]
    print(f"Batches: {batches_6}")
    print(f"  5 full batches (6 vectors) + 1 partial (2 vectors)")
    print(f"  Full batch: 6 valu ops/cycle × 12 cycles = 72 ops/round")
    print(f"  Partial batch: 2 valu ops/cycle × 12 cycles = 24 ops/round (33% efficient)")
    total_cycles_current = 5 * 12 + 12  # Full batches + partial (same cycles due to dependencies)
    print(f"  Estimated cycles per round: {total_cycles_current} (actual may vary)")

    # Alternative 1: 4-vector batches
    print("\n=== Alternative 1: 4-vector batches ===")
    batches_4 = [(i*4, 4) for i in range(8)]
    print(f"Batches: 8 × 4 vectors")
    print(f"  All full batches")
    print(f"  4 valu ops/cycle × 9 cycles = 36 ops/batch")
    total_cycles_4 = 8 * 9
    print(f"  Estimated cycles per round: {total_cycles_4}")

    # Alternative 2: 5-vector batches
    print("\n=== Alternative 2: Mix of 5 and 4 vector batches ===")
    # 32 = 5*5 + 4 + 3 = 25 + 4 + 3 = 32
    # Or: 32 = 5*4 + 4*3 = 20 + 12 = 32
    # Or: 32 = 5*6 + 2 = 30 + 2 = 32 (current)
    batches_mix = [(0, 5), (5, 5), (10, 5), (15, 5), (20, 4), (24, 4), (28, 4)]
    print(f"Batches: 4×5 + 3×4 = {4*5 + 3*4}")
    print(f"  5-vector batch: 5 valu ops/cycle × ~10 cycles = 50 ops/batch")
    print(f"  4-vector batch: 4 valu ops/cycle × 9 cycles = 36 ops/batch")
    # total_cycles_mix = 4 * 10 + 3 * 9

    # Alternative 3: 8-vector batches
    print("\n=== Alternative 3: 8-vector batches ===")
    batches_8 = [(i*8, 8) for i in range(4)]
    print(f"Batches: 4 × 8 vectors")
    print(f"  8 valu ops > 6 slots, need 2 cycles for some ops")
    print(f"  Hash stages 0,2,4: 8 multiply_add → 2 cycles")
    print(f"  Hash stages 1,3,5: 24 ops → 4 cycles")
    print(f"  Per batch: 3×2 + 3×4 = 18 cycles")
    total_cycles_8 = 4 * 18
    print(f"  Estimated cycles per round: {total_cycles_8}")

    print("\n=== Comparison ===")
    print(f"Current (6-vec):     ~{5*12 + 12} cycles (with partial batch overhead)")
    print(f"Alternative 1 (4-vec): ~{8*9} cycles")
    print(f"Alternative 3 (8-vec): ~{4*18} cycles")

    print("\n=== Analysis ===")
    print("The 6-vector batches with a 2-vector tail is inefficient.")
    print("4-vector batches give consistent performance but more batches.")
    print("8-vector batches need more cycles per batch but fewer batches.")
    print("")
    print("For selection phases (no gather overlap), the key is VALU efficiency.")
    print("4-vector batches: 4 ops fit in 6 slots (67% efficient)")
    print("6-vector batches: 6 ops fit in 6 slots (100% efficient)")
    print("8-vector batches: 8 ops need 2 cycles (67% efficient)")
    print("")
    print("Best strategy: 5×6 + 1×2 with better handling of the 2-vector tail")
    print("Or: 4×6 + 2×4 = 24 + 8 = 32 vectors")

    print("\n=== Proposed: 4×6 + 2×4 ===")
    batches_new = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 4), (28, 4)]
    print(f"Batches: {batches_new}")
    print(f"  4 full 6-vector batches + 2 4-vector batches")
    print(f"  6-vec batch: 6 ops × 12 cycles (100% efficient)")
    print(f"  4-vec batch: 4 ops × 9 cycles (67% efficient but no waste)")
    total_cycles_new = 4 * 12 + 2 * 9
    print(f"  Estimated cycles per round: {total_cycles_new}")
    print(f"  Current with waste: ~72 cycles")
    print(f"  Savings: ~{72 - total_cycles_new} cycles per round")
    print(f"  6 selection rounds × 6 cycles = ~36 cycles total savings")

if __name__ == '__main__':
    analyze_batch_strategies()
