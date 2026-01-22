#!/usr/bin/env python3
"""
Compute theoretical efficiency limits and identify where cycles go.
"""

from perf_takehome import KernelBuilder
from problem import Tree, Input, build_mem_image, Machine, VLEN, HASH_STAGES

def analyze_theoretical_limits():
    print("=" * 70)
    print("THEORETICAL EFFICIENCY ANALYSIS")
    print("=" * 70)

    # Problem parameters
    n_items = 256
    n_rounds = 16
    n_vectors = n_items // VLEN  # 32 vectors

    # Hash computation cost per vector
    # 6 stages: 3 multiply_add (1 cycle each), 3 shift+xor (3 cycles each)
    hash_cycles_per_vector = 3 * 1 + 3 * 3  # = 12 VALU cycles
    # But with 6 VALU slots, can process multiple vectors in parallel
    # Actually: 6 vectors can share cycles
    hash_cycles_per_batch_6 = 12  # For 6 vectors processed together

    # Gather cost
    # 4 vectors × 8 elements = 32 elements
    # 2 loads per cycle = 16 cycles
    gather_cycles_per_4_vectors = 16

    print(f"\nHash cost analysis:")
    print(f"  Per 6-vector batch (multiply_add stages): ~3 cycles")
    print(f"  Per 6-vector batch (shift+xor stages): ~9 cycles")
    print(f"  Total per 6-vector batch: ~12 cycles")
    print(f"  For 32 vectors (5.33 batches): ~64 cycles")

    print(f"\nGather cost analysis:")
    print(f"  Per 4-vector batch: 16 LOAD cycles (8 elements × 4 / 2)")
    print(f"  For 32 vectors (8 batches): 128 LOAD cycles")
    print(f"  With hash overlap: 128 cycles total (hash fits in gather time)")

    print(f"\n" + "=" * 70)
    print("CURRENT STRUCTURE ANALYSIS")
    print("=" * 70)

    # Rounds 0-2: No gather needed
    # - Round 0: broadcast tree[0], XOR, hash, idx update
    # - Round 1: arithmetic tree value, XOR, hash, idx update
    # - Round 2: 4-way selection, XOR, hash, idx update

    # For 32 vectors in 6-vector batches:
    batches = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 6), (30, 2)]  # 6 batches

    print(f"\nRounds 0-2 (no gather):")
    print(f"  Processing: {len(batches)} batches of 6 vectors")

    # Per round, per batch:
    # - Node value computation: 1-6 cycles (broadcast/arithmetic/selection)
    # - XOR: 1 cycle
    # - Hash: ~12 cycles
    # - Index update: ~2 cycles
    # Total: ~16-20 cycles per batch per round

    cycles_per_round_0 = 16 * len(batches)  # ~96
    cycles_per_round_1 = 17 * len(batches)  # ~102
    cycles_per_round_2 = 22 * len(batches)  # ~132 (more complex selection)

    print(f"  Estimated round 0: {cycles_per_round_0} cycles")
    print(f"  Estimated round 1: {cycles_per_round_1} cycles")
    print(f"  Estimated round 2: {cycles_per_round_2} cycles")
    print(f"  Plus setup/overhead: ~80 cycles")
    print(f"  Total rounds 0-2: ~{cycles_per_round_0 + cycles_per_round_1 + cycles_per_round_2 + 80}")

    # Rounds 3-10: Gather + hash overlap
    print(f"\nRounds 3-10 (gather, 8 rounds):")
    # 8 batches of 4 vectors each
    # Per batch: 16 gather cycles overlapped with 12 hash cycles
    # Total per round: 8 × 16 = 128 cycles
    # But with pipelining across batches, overlaps nicely
    gather_loop_per_round = 18  # From profiler
    print(f"  Per round (with overlap): ~{gather_loop_per_round} cycles")
    print(f"  Total 8 rounds: ~{gather_loop_per_round * 8} cycles")

    # Rounds 11-13: Same as 0-2
    print(f"\nRounds 11-13 (no gather, 3 rounds):")
    print(f"  Similar to rounds 0-2: ~336 cycles")

    # Rounds 14-15: Gather + hash
    print(f"\nRounds 14-15 (gather, 2 rounds):")
    print(f"  Similar to 3-10: ~{gather_loop_per_round * 2} cycles")

    print(f"\n" + "=" * 70)
    print("EFFICIENCY GAP ANALYSIS")
    print("=" * 70)

    print(f"\nCurrent breakdown:")
    print(f"  Rounds 0-2:   407 cycles (VALU: 72%, LOAD: 10%)")
    print(f"  Rounds 3-10:  145 cycles (VALU: 75%, LOAD: 88%)")
    print(f"  Rounds 11-13: 336 cycles (VALU: 80%, LOAD: 5%)")
    print(f"  Rounds 14-15: 145 cycles (VALU: 75%, LOAD: 88%)")
    print(f"  Other:        ~1207 cycles")

    wait_im_wrong = """
Actually let me recalculate:
init: 13 cycles
rounds_0-2_setup: 407 cycles (13-419)
loop1_body: 145 cycles (420-564)
rounds_11-13_setup: 336 cycles (565-900)
loop2_body: 145 cycles (901-1045)
store_phase: 34 cycles (1046-1079)
Total accounted: 13 + 407 + 145 + 336 + 145 + 34 = 1080 cycles

But machine.cycle = 2240?
The loop bodies execute multiple times!
"""

    print(f"\nWait - loop bodies execute multiple times!")
    print(f"  loop1_body (rounds 3-10): 145 static cycles × ~8 iterations = ~1160 dynamic cycles")
    print(f"  loop2_body (rounds 14-15): 145 static cycles × ~2 iterations = ~290 dynamic cycles")

    print(f"\nActual breakdown (estimated):")
    actual_rounds_0_2 = 407
    actual_rounds_3_10 = 145 * 8  # 8 rounds, 145 per iteration
    actual_rounds_11_13 = 336
    actual_rounds_14_15 = 145 * 2  # 2 rounds
    other = 13 + 34  # init + store

    # But loop1 and loop2 aren't 145 cycles each, that's the static size
    # The loops iterate based on the round counter

    print(f"\n*** KEY INSIGHT ***")
    print(f"The 145-cycle 'loop body' is PER ITERATION, not per round!")
    print(f"Loop1 iterates 8 times (rounds 3-10)")
    print(f"Loop2 iterates 2 times (rounds 14-15)")
    print(f"")
    print(f"So the actual cycle distribution is approximately:")
    print(f"  Init: 13")
    print(f"  Rounds 0-2 setup: 407")
    print(f"  Loop1 (8 iterations): 145 × 8 = 1160 (but rounds 3-10 are embedded in loop)")
    print(f"  Rounds 11-13 setup: 336")
    print(f"  Loop2 (2 iterations): 145 × 2 = 290 (but rounds 14-15 are embedded)")
    print(f"  Store: 34")
    print(f"  Total: 13 + 407 + 336 + 34 = 790 cycles non-loop + ??? loop")
    print(f"  But actual is 2240...")

    print(f"\nLet me reconsider the loop structure:")
    print(f"  Loop1 processes rounds 3-10 for ALL 256 items")
    print(f"  Inside the loop: process one batch through one round")
    print(f"  Iterations: 8 rounds × 8 batches = 64 iterations? Or 8 rounds only?")

def verify_loop_structure():
    """Run the kernel and trace the actual loop behavior."""
    kb = KernelBuilder()
    tree = Tree.generate(height=10)
    inp = Input.generate(forest=tree, rounds=16, batch_size=256)
    mem = build_mem_image(tree, inp)
    kb.build_kernel(tree.height, len(tree.values), len(inp.values), inp.rounds)

    # Count jumps
    jump_instrs = []
    for i, instr in enumerate(kb.instrs):
        if 'flow' in instr:
            for slot in instr['flow']:
                if slot[0] == 'cond_jump':
                    jump_instrs.append((i, slot[2], slot[1]))  # from, to, condition

    print(f"\n" + "=" * 70)
    print("LOOP STRUCTURE VERIFICATION")
    print("=" * 70)
    print(f"\nTotal static instructions: {len(kb.instrs)}")
    print(f"\nJump instructions:")
    for from_idx, to_idx, cond in jump_instrs:
        loop_size = from_idx - to_idx + 1
        print(f"  Instruction {from_idx} jumps to {to_idx} (loop body: {loop_size} instrs)")
        # Estimate iterations based on the loop structure
        if loop_size > 100:
            print(f"    This is likely the main gather loop")

    machine = Machine(mem, kb.instrs, kb.debug_info())
    machine.run()  # Stops at pause
    machine.run()  # Completes

    print(f"\nTotal dynamic cycles: {machine.cycle}")
    print(f"Total static instructions: {len(kb.instrs)}")
    print(f"Implied loop expansion: {machine.cycle / len(kb.instrs):.2f}x")

if __name__ == '__main__':
    analyze_theoretical_limits()
    verify_loop_structure()
