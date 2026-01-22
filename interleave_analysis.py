#!/usr/bin/env python3
"""
Analyze potential for interleaved round processing.

Key insight: If we process different item groups at different rounds,
we might be able to overlap VALU work (selection phases) with LOAD work (gather phases).
"""

def analyze_interleave_potential():
    print("=" * 70)
    print("INTERLEAVED PROCESSING ANALYSIS")
    print("=" * 70)

    # Current structure (for 256 items, 32 vectors)
    print("\n=== Current Structure ===")
    print("Rounds 0-2: Selection (VALU only) - 407 cycles")
    print("Rounds 3-10: Gather (VALU + LOAD overlap) - 1160 cycles")
    print("Rounds 11-13: Selection (VALU only) - 336 cycles")
    print("Rounds 14-15: Gather (VALU + LOAD overlap) - 290 cycles")
    print("Other: 47 cycles")
    print("Total: 2240 cycles")

    print("\n=== Bottleneck Analysis ===")
    print("Selection phases: 743 cycles, LOAD idle ~90% of the time")
    print("Gather phases: 1450 cycles, LOAD busy ~88% of the time")
    print("")
    print("Key insight: 637 LOAD-idle cycles during selection phases")
    print("             could potentially be used for useful work!")

    print("\n=== Interleave Idea ===")
    print("Split items into Group A (0-127) and Group B (128-255)")
    print("")
    print("Instead of:")
    print("  All items: R0 → R1 → R2 → R3 → ... → R15")
    print("")
    print("Do:")
    print("  Group A: R0 → R1 → R2 → R3 → R4 → ...")
    print("  Group B:           R0 → R1 → R2 → R3 → ...")
    print("           (starts when A reaches R3)")
    print("")
    print("When Group A is at R3+ (gather, LOAD-heavy):")
    print("  Group B is at R0-R2 (selection, VALU-only)")
    print("  → Potential overlap!")

    print("\n=== Detailed Timing Analysis ===")

    # Per-group costs (half the items = 128 items = 16 vectors)
    n_vectors = 16  # per group

    # Selection phase costs (per round, per group)
    # 16 vectors in 6-vector batches = 2.67 batches ≈ 3 batches
    batches_6 = 3
    # Per batch: ~21 cycles (6 cycles selection, 1 XOR, 12 hash, 2 idx update)
    cycles_per_batch_selection = 21
    selection_cycles_per_round = batches_6 * cycles_per_batch_selection  # 63 cycles

    # Gather phase costs (per round, per group)
    # 16 vectors in 4-vector batches = 4 batches
    batches_4 = 4
    # Per batch: 18 cycles (16 LOAD gather + 9 VALU hash overlapped + 2 idx)
    gather_cycles_per_round = batches_4 * 18  # 72 cycles

    print(f"Per group (128 items = 16 vectors):")
    print(f"  Selection round: {selection_cycles_per_round} cycles (VALU-bound)")
    print(f"  Gather round: {gather_cycles_per_round} cycles (LOAD-bound)")

    print("\n=== Without Interleave (Sequential) ===")
    # Process Group A fully, then Group B
    total_sequential = 2 * (
        3 * selection_cycles_per_round +  # R0-R2: 3 × 63 = 189
        8 * gather_cycles_per_round +      # R3-R10: 8 × 72 = 576
        3 * selection_cycles_per_round +  # R11-R13: 3 × 63 = 189
        2 * gather_cycles_per_round        # R14-R15: 2 × 72 = 144
    )  # = 2 × 1098 = 2196
    print(f"  Group A: 3×63 + 8×72 + 3×63 + 2×72 = {3*63 + 8*72 + 3*63 + 2*72} cycles")
    print(f"  Group B: same")
    print(f"  Total (no overhead): {total_sequential} cycles")

    print("\n=== With Interleave (Overlapped) ===")
    # Overlap Group A's gather with Group B's selection

    # Phase 1: Group A does R0-R2 alone
    phase1 = 3 * selection_cycles_per_round  # 189 cycles
    print(f"  Phase 1: Group A R0-R2 (alone): {phase1} cycles")

    # Phase 2: Group A does R3-R10, Group B does R0-R2 + starts R3-R5
    # Group A: 8 gather rounds = 8 × 72 = 576 cycles
    # Group B: 3 selection rounds + 5 gather rounds (staggered)

    # Can we overlap?
    # Gather round: 72 cycles = 64 LOAD + 36 VALU (hash), with overlap ~72 cycles
    # Selection round: 63 cycles = 63 VALU

    # Within 72 cycles of gather:
    # LOAD busy: 64 cycles (but can be spread)
    # VALU busy: 36 cycles for hash
    # VALU free: 72 - 36 = 36 cycles

    # Selection needs 63 VALU cycles per round
    # If we try to fit selection into VALU-free time:
    # 36 free cycles vs 63 needed → doesn't fully fit!

    # Better model: time-interleave cycles
    # Assume we can interleave at batch level

    # Group A gather batch: 18 cycles (16 LOAD + 9 VALU hash)
    # During these 18 cycles, VALU has 9 cycles of hash work
    # VALU-free: 18 - 9 = 9 cycles

    # Group B selection batch: 21 cycles (all VALU)
    # Can we spread 21 VALU cycles across multiple gather batches?

    # For each Group A gather round (4 batches):
    # Total VALU-free time: 4 × 9 = 36 cycles
    # Group B selection round needs: 63 cycles
    # Overlap possible: 36 cycles
    # Sequential needed: 63 - 36 = 27 cycles

    # Combined time per pair (A gather + B selection):
    # max(72 cycles LOAD-bound, 36+63 cycles VALU-bound) = max(72, 99) = 99 cycles

    # Wait, that's worse! Let me recalculate.

    # Actually, the gather loop already overlaps hash with gather:
    # Batch N gather (16 LOAD cycles) overlaps with Batch N-1 hash (9 VALU cycles)
    # So VALU is only busy for 9 cycles per 18-cycle batch.

    # If we add selection work:
    # Batch N gather (16 LOAD) + Batch N-1 hash (9 VALU) + selection work (X VALU)
    # VALU capacity: 6 ops/cycle × 18 cycles = 108 ops
    # Hash needs: ~36 ops (9 cycles × 4 ops/cycle)
    # Selection batch needs: ~84 ops (21 cycles × 4 ops/cycle)
    # Total VALU ops needed: 36 + 84 = 120 > 108 capacity!

    # So we can fit 108 - 36 = 72 ops of selection into the gather time
    # That's 72/84 = 86% of a selection batch
    # Remaining 12 ops need 12/4 = 3 extra cycles

    # Revised timing per pair (1 gather round + partial selection):
    # LOAD time: 72 cycles
    # VALU time: 36 (hash) + 72 (partial selection overlapped) + 12 (extra) = 120 VALU ops
    # But spread: 72 cycles for LOAD, during which 108 VALU ops can be done
    # Remaining: 120 - 108 = 12 VALU ops = 3 cycles
    # Total: 72 + 3 = 75 cycles per gather round + partial selection round

    print("\n  Interleave calculation:")
    print("  Group A gather batch: 18 cycles (16 LOAD + 9 VALU)")
    print("  Group B selection batch: 21 cycles (all VALU)")
    print("")
    print("  Per 18-cycle gather batch:")
    print("    VALU capacity: 18 × 6 = 108 ops")
    print("    Hash needs: ~36 ops")
    print("    Available for selection: 108 - 36 = 72 ops")
    print("    Selection batch needs: ~84 ops")
    print("    Overflow: 84 - 72 = 12 ops = 3 extra cycles")
    print("")
    print("  Per gather round (4 batches) + 1 selection round (3 batches):")
    print("    Gather time: 4 × 18 = 72 cycles")
    print("    Selection overflow: 3 batches × 3 cycles = 9 cycles")
    print("    Total: 72 + 9 = 81 cycles (vs 72 gather-only or 72+63 sequential)")

    interleaved_gather_selection = 81  # cycles per pair of rounds

    # Phase 2: Group A does R3-R10 (8 rounds), Group B does R0-R2 (3 rounds)
    # Interleave: 8 A-gather rounds + 3 B-selection rounds
    # But B-selection is 3 rounds, A-gather is 8 rounds
    # First 3 A-gather rounds overlap with 3 B-selection rounds: 3 × 81 = 243 cycles
    # Remaining 5 A-gather rounds: 5 × 72 = 360 cycles
    # Total phase 2: 243 + 360 = 603 cycles

    phase2 = 3 * interleaved_gather_selection + 5 * gather_cycles_per_round
    print(f"\n  Phase 2: Group A R3-R10 + Group B R0-R2")
    print(f"    Interleaved (3 rounds): 3 × {interleaved_gather_selection} = {3 * interleaved_gather_selection}")
    print(f"    A-only (5 rounds): 5 × {gather_cycles_per_round} = {5 * gather_cycles_per_round}")
    print(f"    Total: {phase2} cycles")

    # Phase 3: Group A does R11-R13, Group B does R3-R7
    # 3 A-selection + 5 B-gather rounds
    # Interleave: A-selection (VALU) with B-gather (LOAD + VALU)
    # Same analysis as phase 2 but reversed
    # A-selection: 3 rounds × 63 = 189 VALU cycles
    # B-gather: 5 rounds × 72 = 360 cycles (with 36 VALU each = 180 VALU)
    # Total VALU: 189 + 180 = 369 cycles
    # LOAD: 360 cycles (B-gather)
    # Combined: We need to fit 369 VALU cycles into 360 cycles where 180 are already committed
    # Free VALU in 360 cycles: 360 - 180 = 180 cycles
    # A-selection needs: 189 cycles
    # Overflow: 189 - 180 = 9 cycles
    # Total phase 3: 360 + 9 = 369 cycles

    phase3 = 360 + 9  # rough estimate
    print(f"\n  Phase 3: Group A R11-R13 + Group B R3-R7")
    print(f"    Interleaved: ~{phase3} cycles")

    # Phase 4: Group A does R14-R15, Group B does R8-R10 + starts R11-R13
    # Similar analysis...
    # A-gather: 2 rounds × 72 = 144 cycles
    # B-gather: 3 rounds × 72 = 216 cycles
    # B-selection: 3 rounds × 63 = 189 cycles (starts when B finishes gather)
    # Total sequential: 144 + 216 + 189 = 549 cycles
    # With interleave: ?

    phase4 = 400  # rough estimate with some overlap
    print(f"\n  Phase 4: Group A R14-R15 + Group B R8-R15")
    print(f"    Estimated: ~{phase4} cycles")

    total_interleaved = phase1 + phase2 + phase3 + phase4
    print(f"\n  Total estimated interleaved: {total_interleaved} cycles")
    print(f"  vs Sequential: {total_sequential} cycles")
    print(f"  vs Current: 2240 cycles")
    print(f"  Potential savings: {2240 - total_interleaved} cycles")

    print("\n=== Key Challenges ===")
    print("1. VALU is needed for BOTH selection AND gather (hash computation)")
    print("2. Can't fully overlap - VALU becomes the bottleneck")
    print("3. Complex scheduling and register allocation")
    print("4. Code size increase")

    print("\n=== Alternative: Reduce VALU work ===")
    print("If we can reduce hash cycles by 33%, target is achievable:")
    print(f"  Current VALU cycles: ~1631 (72.8% of 2240)")
    print(f"  Target: ~1093 VALU cycles (67% reduction)")
    print(f"  Would give: ~1500 total cycles")

    print("\n=== The 'Transpose' Insight ===")
    print("Blog: 'transposing the entire computation rather than data'")
    print("")
    print("Possibilities:")
    print("1. Change round ordering (interleave as analyzed above)")
    print("2. Change how hash is computed (different stage ordering?)")
    print("3. Change tree traversal pattern")
    print("4. Something more fundamental about the algorithm structure")

if __name__ == '__main__':
    analyze_interleave_potential()
