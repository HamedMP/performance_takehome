"""
Debugging and Analysis Tools for Performance Take-home

Usage:
    python debug_tools.py profile    # Cycle breakdown analysis
    python debug_tools.py indices    # Index range analysis
    python debug_tools.py preload    # Preload feasibility analysis
"""

from perf_takehome import KernelBuilder
from problem import VLEN


def profile_cycles():
    """Analyze where cycles are spent in the kernel"""
    kb = KernelBuilder()
    kb.build_kernel(10, 1023, 256, 16)

    print("=== INSTRUCTION STREAM ANALYSIS ===\n")

    # Find all jumps
    jumps = []
    pause_idx = None
    for i, instr in enumerate(kb.instrs):
        if 'flow' in instr:
            for slot in instr['flow']:
                if slot[0] == 'pause' and pause_idx is None:
                    pause_idx = i
                    print(f"Pause at instruction {i}")
                elif slot[0] == 'cond_jump':
                    jumps.append((i, slot[2]))
                    print(f"Jump at instruction {i}, target: {slot[2]}")

    print(f"\nTotal instructions: {len(kb.instrs)}")

    # Calculate sections properly
    init = pause_idx
    rounds_0_2_and_prologue = jumps[0][1] - pause_idx - 1
    loop1_body = jumps[0][0] - jumps[0][1] + 1
    between_loops = jumps[1][1] - jumps[0][0] - 1
    loop2_body = jumps[1][0] - jumps[1][1] + 1
    store_phase = len(kb.instrs) - jumps[1][0] - 1

    print(f"\n=== CYCLE BREAKDOWN ===")
    print(f"Init (before pause): {init} cycles")
    print(f"Rounds 0-2 + Loop1 prologue: {rounds_0_2_and_prologue} cycles")
    print(f"Loop1 body: {loop1_body} cycles × 8 iterations = {loop1_body * 8} cycles")
    print(f"Rounds 11-13 + Loop2 prologue: {between_loops} cycles")
    print(f"Loop2 body: {loop2_body} cycles × 2 iterations = {loop2_body * 2} cycles")
    print(f"Store phase: {store_phase} cycles")

    total = init + rounds_0_2_and_prologue + loop1_body * 8 + between_loops + loop2_body * 2 + store_phase
    print(f"\nTotal: {total} cycles")

    # Gap analysis
    target = 1487
    gap = total - target
    print(f"\n=== GAP ANALYSIS ===")
    print(f"Current: {total} cycles")
    print(f"Target: {target} cycles")
    print(f"Gap: {gap} cycles ({100*gap/total:.1f}% reduction needed)")


def analyze_index_ranges():
    """Analyze index ranges for each round"""
    print("=== INDEX RANGE ANALYSIS ===\n")

    def get_index_range(round_num):
        if round_num == 0:
            return [0]

        prev_indices = get_index_range(round_num - 1)
        new_indices = set()
        n_nodes = 2047

        for idx in prev_indices:
            new_idx_0 = idx * 2 + 1
            new_idx_1 = idx * 2 + 2

            if new_idx_0 >= n_nodes:
                new_indices.add(0)
            else:
                new_indices.add(new_idx_0)

            if new_idx_1 >= n_nodes:
                new_indices.add(0)
            else:
                new_indices.add(new_idx_1)

        return sorted(new_indices)

    for r in range(16):
        indices = get_index_range(r)
        min_idx = min(indices)
        max_idx = max(indices)
        count = len(indices)

        if count == 1:
            opt = "BROADCAST"
        elif count == 2:
            opt = "ARITHMETIC (2 values)"
        elif count <= 4:
            opt = "2-BIT SELECTION (4 values)"
        elif count <= 8:
            opt = "3-BIT SELECTION (8 values)"
        elif count <= 16:
            opt = "4-BIT SELECTION (16 values)"
        elif count <= 32:
            opt = "5-BIT SELECTION (32 values)"
        elif count <= 64:
            opt = "6-BIT SELECTION (64 values)"
        else:
            opt = f"GATHER ({count} values)"

        print(f"Round {r:2d}: indices {min_idx:4d}-{max_idx:4d} ({count:3d} unique) -> {opt}")


def analyze_preload_feasibility():
    """Analyze which rounds could use preloaded tree values"""
    print("=== PRELOAD FEASIBILITY ANALYSIS ===\n")

    def get_max_idx(round_num):
        if round_num == 0:
            return 0
        indices = []
        def get_indices(r):
            if r == 0:
                return [0]
            prev = get_indices(r - 1)
            result = set()
            for idx in prev:
                n0, n1 = idx * 2 + 1, idx * 2 + 2
                if n0 >= 2047:
                    result.add(0)
                else:
                    result.add(n0)
                if n1 >= 2047:
                    result.add(0)
                else:
                    result.add(n1)
            return sorted(result)
        return max(get_indices(round_num))

    # Analyze each round
    gather_needed = []
    preload_feasible = []

    for r in range(16):
        max_idx = get_max_idx(r)
        if max_idx <= 62:
            preload_feasible.append(r)
            print(f"Round {r:2d}: max_idx={max_idx:4d} - PRELOAD FEASIBLE")
        else:
            gather_needed.append(r)
            print(f"Round {r:2d}: max_idx={max_idx:4d} - NEEDS GATHER")

    print(f"\n=== SUMMARY ===")
    print(f"Preload feasible rounds: {preload_feasible}")
    print(f"Gather needed rounds: {gather_needed}")
    print(f"Potential gather elimination: {len(preload_feasible) - 6} more rounds")
    print(f"  (Currently optimizing: 0, 1, 2, 11, 12, 13)")
    print(f"  (Could also optimize: 3, 4, 5, 14, 15)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "profile":
        profile_cycles()
    elif cmd == "indices":
        analyze_index_ranges()
    elif cmd == "preload":
        analyze_preload_feasibility()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
