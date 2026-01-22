#!/usr/bin/env python3
"""
Custom CLI Profiler for VLIW SIMD Kernel
Similar to ncu-cli for analyzing simulator traces

Usage:
    python profiler.py summary     # High-level cycle breakdown
    python profiler.py slots       # Slot utilization analysis
    python profiler.py timeline    # Timeline of operations
    python profiler.py bottleneck  # Identify bottlenecks
    python profiler.py pipeline    # Pipeline efficiency analysis
    python profiler.py compare     # Compare engine utilization
"""

import sys
from collections import defaultdict
from perf_takehome import KernelBuilder
from problem import Tree, Input, build_mem_image, Machine, SLOT_LIMITS, VLEN

def build_and_run():
    """Build kernel and run simulation."""
    kb = KernelBuilder()
    tree = Tree.generate(height=10)
    inp = Input.generate(forest=tree, rounds=16, batch_size=256)
    mem = build_mem_image(tree, inp)
    kb.build_kernel(tree.height, len(tree.values), len(inp.values), inp.rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info())
    # Run twice - first run stops at pause, second completes
    machine.run()  # Stops at pause
    machine.run()  # Completes

    return kb, machine

def get_instruction_regions(instrs):
    """Identify code regions based on instruction patterns."""
    regions = []

    # Find pause (marks end of init)
    pause_idx = None
    for i, instr in enumerate(instrs):
        if 'flow' in instr and any(s[0] == 'pause' for s in instr['flow']):
            pause_idx = i
            break

    # Find jumps (mark loop boundaries)
    jumps = []
    for i, instr in enumerate(instrs):
        if 'flow' in instr:
            for slot in instr['flow']:
                if slot[0] == 'cond_jump':
                    jumps.append((i, slot[2]))  # (jump_from, jump_to)

    # Build regions
    if pause_idx is not None:
        regions.append(('init', 0, pause_idx))

    # Identify loops and non-loop regions
    if jumps:
        loop1_end, loop1_start = jumps[0]
        if pause_idx is not None:
            regions.append(('rounds_0-2_setup', pause_idx + 1, loop1_start - 1))
        regions.append(('loop1_body', loop1_start, loop1_end))

        if len(jumps) > 1:
            loop2_end, loop2_start = jumps[1]
            regions.append(('rounds_11-13_setup', loop1_end + 1, loop2_start - 1))
            regions.append(('loop2_body', loop2_start, loop2_end))
            regions.append(('store_phase', loop2_end + 1, len(instrs) - 1))

    return regions

def cmd_summary():
    """High-level cycle breakdown."""
    kb, machine = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - SUMMARY")
    print("=" * 60)
    print(f"\nTotal Cycles: {machine.cycle}")
    print(f"Total Instructions: {len(kb.instrs)}")
    print(f"Target: < 1,487 cycles")
    print(f"Gap: {machine.cycle - 1487} cycles ({(machine.cycle - 1487) / 1487 * 100:.1f}% over)")

    # Identify regions
    regions = get_instruction_regions(kb.instrs)

    print("\n" + "-" * 60)
    print("REGION BREAKDOWN")
    print("-" * 60)
    print(f"{'Region':<25} {'Start':>8} {'End':>8} {'Cycles':>8} {'%':>6}")
    print("-" * 60)

    for name, start, end in regions:
        cycles = end - start + 1
        pct = cycles / machine.cycle * 100
        print(f"{name:<25} {start:>8} {end:>8} {cycles:>8} {pct:>5.1f}%")

    # Engine utilization summary
    print("\n" + "-" * 60)
    print("ENGINE UTILIZATION (avg slots/cycle)")
    print("-" * 60)

    engine_totals = defaultdict(int)
    for instr in kb.instrs:
        for engine, slots in instr.items():
            if engine in SLOT_LIMITS:
                engine_totals[engine] += len(slots)

    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        if engine in engine_totals:
            avg = engine_totals[engine] / len(kb.instrs)
            max_slots = SLOT_LIMITS.get(engine, 1)
            util = avg / max_slots * 100
            bar = '█' * int(util / 5) + '░' * (20 - int(util / 5))
            print(f"{engine:>6}: {avg:>5.2f}/{max_slots} [{bar}] {util:>5.1f}%")

def cmd_slots():
    """Detailed slot utilization analysis."""
    kb, _ = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - SLOT UTILIZATION")
    print("=" * 60)

    # Per-engine histograms
    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        hist = defaultdict(int)
        for instr in kb.instrs:
            slots_used = len(instr.get(engine, []))
            hist[slots_used] += 1

        if sum(hist.values()) > hist[0]:  # If engine is ever used
            max_slots = SLOT_LIMITS.get(engine, 1)
            print(f"\n{engine.upper()} Engine (max {max_slots} slots):")
            print("-" * 40)
            for i in range(max_slots + 1):
                count = hist[i]
                pct = count / len(kb.instrs) * 100
                bar = '█' * int(pct / 2)
                print(f"  {i} slots: {count:>4} cycles ({pct:>5.1f}%) {bar}")

def cmd_timeline():
    """Show timeline of operations."""
    kb, _ = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - TIMELINE (first 50 cycles)")
    print("=" * 60)
    print(f"\n{'Cycle':>5} {'VALU':>4} {'LOAD':>4} {'STOR':>4} {'ALU':>4} {'FLOW':>4}  Operations")
    print("-" * 70)

    for i, instr in enumerate(kb.instrs[:50]):
        valu = len(instr.get('valu', []))
        load = len(instr.get('load', []))
        store = len(instr.get('store', []))
        alu = len(instr.get('alu', []))
        flow = len(instr.get('flow', []))

        # Summarize operations
        ops = []
        if 'valu' in instr:
            op_types = set(s[0] for s in instr['valu'])
            ops.append(f"valu:{','.join(op_types)}")
        if 'load' in instr:
            op_types = set(s[0] for s in instr['load'])
            ops.append(f"load:{','.join(op_types)}")
        if 'flow' in instr:
            op_types = set(s[0] for s in instr['flow'])
            ops.append(f"flow:{','.join(op_types)}")

        ops_str = ' '.join(ops)[:35]
        print(f"{i:>5} {valu:>4} {load:>4} {store:>4} {alu:>4} {flow:>4}  {ops_str}")

def cmd_bottleneck():
    """Identify bottleneck cycles."""
    kb, _ = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Categorize cycles by bottleneck type
    categories = {
        'valu_only': 0,      # VALU busy, LOAD idle
        'load_only': 0,      # LOAD busy, VALU idle
        'both_busy': 0,      # Both busy (good overlap)
        'both_idle': 0,      # Both idle (waste)
        'valu_saturated': 0, # VALU at max (6 slots)
        'load_saturated': 0, # LOAD at max (2 slots)
    }

    valu_idle_load_busy = []
    load_idle_valu_busy = []

    for i, instr in enumerate(kb.instrs):
        valu = len(instr.get('valu', []))
        load = len(instr.get('load', []))

        valu_busy = valu >= 3
        load_busy = load >= 1

        if valu_busy and not load_busy:
            categories['valu_only'] += 1
            load_idle_valu_busy.append(i)
        elif load_busy and not valu_busy:
            categories['load_only'] += 1
            valu_idle_load_busy.append(i)
        elif valu_busy and load_busy:
            categories['both_busy'] += 1
        else:
            categories['both_idle'] += 1

        if valu == 6:
            categories['valu_saturated'] += 1
        if load == 2:
            categories['load_saturated'] += 1

    print("\nCycle Categories:")
    print("-" * 50)
    total = len(kb.instrs)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {cat:<20}: {count:>4} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 50)
    print("OPTIMIZATION OPPORTUNITIES")
    print("-" * 50)

    print(f"\nLOAD idle while VALU busy: {len(load_idle_valu_busy)} cycles")
    print("  -> Could preload tree values or do other LOAD work")

    print(f"\nVALU idle while LOAD busy: {len(valu_idle_load_busy)} cycles")
    print("  -> Could do more hash/index computation")

    # Show where these occur
    if load_idle_valu_busy:
        regions = get_instruction_regions(kb.instrs)
        region_counts = defaultdict(int)
        for cycle in load_idle_valu_busy:
            for name, start, end in regions:
                if start <= cycle <= end:
                    region_counts[name] += 1
                    break

        print("\n  LOAD-idle cycles by region:")
        for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
            print(f"    {region}: {count} cycles")

def cmd_pipeline():
    """Analyze pipeline efficiency in gather loops."""
    kb, _ = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - PIPELINE ANALYSIS")
    print("=" * 60)

    # Find the main gather loop (Loop1)
    regions = get_instruction_regions(kb.instrs)
    loop1_region = None
    for name, start, end in regions:
        if name == 'loop1_body':
            loop1_region = (start, end)
            break

    if not loop1_region:
        print("Could not find loop1_body region")
        return

    start, end = loop1_region
    loop_instrs = kb.instrs[start:end+1]

    print(f"\nLoop1 body: instructions {start}-{end} ({len(loop_instrs)} cycles)")
    print("-" * 50)

    # Analyze overlap in the loop
    gather_cycles = 0
    hash_cycles = 0
    overlap_cycles = 0

    for instr in loop_instrs:
        valu = len(instr.get('valu', []))
        load = len(instr.get('load', []))

        has_gather = 'load' in instr and any(s[0] == 'load_offset' for s in instr['load'])
        has_hash = valu >= 3

        if has_gather:
            gather_cycles += 1
        if has_hash:
            hash_cycles += 1
        if has_gather and has_hash:
            overlap_cycles += 1

    print(f"Cycles with gather (load_offset): {gather_cycles}")
    print(f"Cycles with hash (VALU >= 3): {hash_cycles}")
    print(f"Cycles with BOTH (overlap): {overlap_cycles}")

    if gather_cycles > 0:
        overlap_pct = overlap_cycles / gather_cycles * 100
        print(f"\nOverlap efficiency: {overlap_pct:.1f}% of gather cycles have hash overlap")

    # Theoretical analysis
    print("\n" + "-" * 50)
    print("THEORETICAL ANALYSIS")
    print("-" * 50)

    items = 256
    rounds_in_loop = 8  # rounds 3-10
    vectors = items // VLEN  # 32 vectors
    batches_per_round = vectors // 4  # 8 batches of 4 vectors

    gather_per_batch = 16  # 4 vectors × 8 elements / 2 loads per cycle
    hash_per_batch = 12    # overlapped

    theoretical_min = batches_per_round * gather_per_batch  # LOAD-bound
    print(f"Items: {items}, Rounds: {rounds_in_loop}")
    print(f"Batches/round: {batches_per_round}, Gather cycles/batch: {gather_per_batch}")
    print(f"Theoretical minimum (LOAD-bound): {theoretical_min} cycles/round")
    print(f"Actual loop body: {len(loop_instrs)} cycles")
    print(f"Overhead: {len(loop_instrs) - theoretical_min} cycles ({(len(loop_instrs) - theoretical_min) / theoretical_min * 100:.1f}%)")

def cmd_compare():
    """Compare VALU vs LOAD utilization cycle by cycle."""
    kb, _ = build_and_run()

    print("=" * 60)
    print("KERNEL PROFILER - VALU vs LOAD COMPARISON")
    print("=" * 60)

    # Create utilization arrays
    valu_util = []
    load_util = []

    for instr in kb.instrs:
        valu_util.append(len(instr.get('valu', [])) / 6 * 100)
        load_util.append(len(instr.get('load', [])) / 2 * 100)

    # Summary stats
    import statistics
    print(f"\nVALU Utilization:")
    print(f"  Mean: {statistics.mean(valu_util):.1f}%")
    print(f"  Median: {statistics.median(valu_util):.1f}%")
    print(f"  Stdev: {statistics.stdev(valu_util):.1f}%")

    print(f"\nLOAD Utilization:")
    print(f"  Mean: {statistics.mean(load_util):.1f}%")
    print(f"  Median: {statistics.median(load_util):.1f}%")
    print(f"  Stdev: {statistics.stdev(load_util):.1f}%")

    # Find correlation
    n = len(valu_util)
    mean_v = sum(valu_util) / n
    mean_l = sum(load_util) / n

    cov = sum((v - mean_v) * (l - mean_l) for v, l in zip(valu_util, load_util)) / n
    std_v = (sum((v - mean_v) ** 2 for v in valu_util) / n) ** 0.5
    std_l = (sum((l - mean_l) ** 2 for l in load_util) / n) ** 0.5

    if std_v > 0 and std_l > 0:
        correlation = cov / (std_v * std_l)
        print(f"\nVALU-LOAD Correlation: {correlation:.3f}")
        if correlation > 0.3:
            print("  -> Positive: VALU and LOAD tend to be busy together (good overlap)")
        elif correlation < -0.3:
            print("  -> Negative: VALU and LOAD alternate (poor overlap)")
        else:
            print("  -> Weak: No strong pattern")

    # Distribution of combined utilization
    print("\n" + "-" * 50)
    print("COMBINED UTILIZATION DISTRIBUTION")
    print("-" * 50)

    bins = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 200)]
    for lo, hi in bins:
        count = sum(1 for v, l in zip(valu_util, load_util) if lo <= v + l < hi)
        pct = count / n * 100
        bar = '█' * int(pct / 2)
        print(f"  {lo:>3}-{hi:<3}%: {count:>4} cycles ({pct:>5.1f}%) {bar}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    commands = {
        'summary': cmd_summary,
        'slots': cmd_slots,
        'timeline': cmd_timeline,
        'bottleneck': cmd_bottleneck,
        'pipeline': cmd_pipeline,
        'compare': cmd_compare,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)

if __name__ == '__main__':
    main()
