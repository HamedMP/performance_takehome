#!/usr/bin/env python3
"""
Micro-analysis to find small optimization opportunities.
"""

from collections import defaultdict
from perf_takehome import KernelBuilder
from problem import Tree, Input, build_mem_image, Machine, SLOT_LIMITS, VLEN

def build_and_run():
    kb = KernelBuilder()
    tree = Tree.generate(height=10)
    inp = Input.generate(forest=tree, rounds=16, batch_size=256)
    mem = build_mem_image(tree, inp)
    kb.build_kernel(tree.height, len(tree.values), len(inp.values), inp.rounds)
    machine = Machine(mem, kb.instrs, kb.debug_info())
    machine.run()
    machine.run()
    return kb, machine

def find_idle_cycles():
    """Find cycles where both VALU and LOAD are idle."""
    kb, _ = build_and_run()

    print("=" * 70)
    print("IDLE CYCLE ANALYSIS")
    print("=" * 70)

    idle_cycles = []
    low_util_cycles = []

    for i, instr in enumerate(kb.instrs):
        valu = len(instr.get('valu', []))
        load = len(instr.get('load', []))
        store = len(instr.get('store', []))
        alu = len(instr.get('alu', []))
        flow = len(instr.get('flow', []))

        total_util = valu/6 + load/2 + store/2 + alu/12 + flow/1

        if valu == 0 and load == 0 and store == 0:
            idle_cycles.append((i, instr))
        elif total_util < 0.5:
            low_util_cycles.append((i, total_util, instr))

    print(f"\nCycles with NO valu, load, or store: {len(idle_cycles)}")
    for i, instr in idle_cycles[:20]:
        print(f"  Cycle {i}: {instr}")
    if len(idle_cycles) > 20:
        print(f"  ... and {len(idle_cycles) - 20} more")

    print(f"\nCycles with < 50% total utilization: {len(low_util_cycles)}")
    for i, util, instr in low_util_cycles[:20]:
        print(f"  Cycle {i} ({util*100:.1f}%): {instr}")
    if len(low_util_cycles) > 20:
        print(f"  ... and {len(low_util_cycles) - 20} more")

def find_valu_inefficiencies():
    """Find cycles where VALU has 1-3 slots used (could pack more)."""
    kb, _ = build_and_run()

    print("\n" + "=" * 70)
    print("VALU PACKING OPPORTUNITIES")
    print("=" * 70)

    light_valu = []
    for i, instr in enumerate(kb.instrs):
        valu = len(instr.get('valu', []))
        if 1 <= valu <= 3:
            light_valu.append((i, valu, instr))

    print(f"\nCycles with 1-3 VALU ops (could potentially pack more): {len(light_valu)}")

    # Group by region
    regions = {
        'init': (0, 12),
        'rounds_0-2': (13, 419),
        'loop1': (420, 564),
        'rounds_11-13': (565, 900),
        'loop2': (901, 1045),
        'store': (1046, 1079),
    }

    region_counts = defaultdict(int)
    for i, valu, instr in light_valu:
        for name, (start, end) in regions.items():
            if start <= i <= end:
                region_counts[name] += 1
                break

    print("\nBy region:")
    for name, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} cycles")

    print("\nExamples (first 15):")
    for i, valu, instr in light_valu[:15]:
        ops = instr.get('valu', [])
        op_types = [op[0] for op in ops]
        print(f"  Cycle {i}: {valu} VALU ops: {op_types}, other: {[k for k in instr if k != 'valu']}")

def find_load_inefficiencies():
    """Find cycles where LOAD has 1 slot used (could pack more)."""
    kb, _ = build_and_run()

    print("\n" + "=" * 70)
    print("LOAD PACKING OPPORTUNITIES")
    print("=" * 70)

    light_load = []
    for i, instr in enumerate(kb.instrs):
        load = len(instr.get('load', []))
        if load == 1:
            light_load.append((i, instr))

    print(f"\nCycles with exactly 1 LOAD op (could pack 2): {len(light_load)}")

    # Check if adjacent cycles could be merged
    merge_candidates = []
    for i, instr in light_load:
        if i + 1 < len(kb.instrs):
            next_instr = kb.instrs[i + 1]
            next_load = len(next_instr.get('load', []))
            if next_load == 1:
                # Both have 1 load, could potentially merge
                # Check if they conflict on other resources
                merge_candidates.append((i, instr, next_instr))

    print(f"\nPairs of adjacent 1-load cycles: {len(merge_candidates)}")
    print("Examples (first 10):")
    for i, instr, next_instr in merge_candidates[:10]:
        load_op = instr.get('load', [])
        next_load_op = next_instr.get('load', [])
        valu_conflict = len(instr.get('valu', [])) + len(next_instr.get('valu', [])) > 6
        print(f"  Cycles {i},{i+1}: {load_op[0][0]}, {next_load_op[0][0]} (VALU conflict: {valu_conflict})")

def analyze_selection_overhead():
    """Analyze overhead in selection phases."""
    kb, _ = build_and_run()

    print("\n" + "=" * 70)
    print("SELECTION PHASE OVERHEAD ANALYSIS")
    print("=" * 70)

    # Rounds 0-2: cycles 13-419
    # Rounds 11-13: cycles 565-900

    phases = [
        ('rounds_0-2', 13, 419),
        ('rounds_11-13', 565, 900),
    ]

    for name, start, end in phases:
        phase_instrs = kb.instrs[start:end+1]

        print(f"\n{name} ({end-start+1} cycles):")

        # Count operation types
        op_counts = defaultdict(lambda: defaultdict(int))
        for instr in phase_instrs:
            for engine, ops in instr.items():
                if engine in ['valu', 'load', 'store', 'alu', 'flow']:
                    for op in ops:
                        op_counts[engine][op[0]] += 1

        print(f"  VALU operations:")
        for op, count in sorted(op_counts['valu'].items(), key=lambda x: -x[1]):
            print(f"    {op}: {count}")

        print(f"  LOAD operations:")
        for op, count in sorted(op_counts['load'].items(), key=lambda x: -x[1]):
            print(f"    {op}: {count}")

        # Find the most common patterns
        print(f"\n  Most common instruction patterns:")
        pattern_counts = defaultdict(int)
        for instr in phase_instrs:
            valu = len(instr.get('valu', []))
            load = len(instr.get('load', []))
            alu = len(instr.get('alu', []))
            flow = len(instr.get('flow', []))
            pattern = f"V{valu}L{load}A{alu}F{flow}"
            pattern_counts[pattern] += 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {pattern}: {count} cycles")

if __name__ == '__main__':
    find_idle_cycles()
    find_valu_inefficiencies()
    find_load_inefficiencies()
    analyze_selection_overhead()
