#!/usr/bin/env python3
"""
Detailed analysis of cycles in setup phases (rounds 0-2, 11-13).
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
    machine.run()  # Stops at pause
    machine.run()  # Completes
    return kb, machine

def analyze_setup_phases():
    kb, _ = build_and_run()

    print("=" * 70)
    print("DETAILED ANALYSIS: SETUP PHASES (rounds 0-2, 11-13)")
    print("=" * 70)

    # Find phase boundaries
    regions = {}
    # From profiler: rounds_0-2_setup: 13-419, rounds_11-13_setup: 565-900
    regions['rounds_0-2'] = (13, 419)
    regions['rounds_11-13'] = (565, 900)

    for phase_name, (start, end) in regions.items():
        print(f"\n{'='*70}")
        print(f"PHASE: {phase_name} (cycles {start}-{end}, {end-start+1} cycles)")
        print("="*70)

        phase_instrs = kb.instrs[start:end+1]

        # Count operations by type
        op_counts = defaultdict(lambda: defaultdict(int))
        valu_idle = 0
        load_idle = 0
        both_idle = 0
        both_busy = 0

        for i, instr in enumerate(phase_instrs):
            valu_ops = instr.get('valu', [])
            load_ops = instr.get('load', [])

            valu_used = len(valu_ops)
            load_used = len(load_ops)

            if valu_used == 0 and load_used == 0:
                both_idle += 1
            elif valu_used == 0:
                valu_idle += 1
            elif load_used == 0:
                load_idle += 1
            else:
                both_busy += 1

            for op in valu_ops:
                op_counts['valu'][op[0]] += 1
            for op in load_ops:
                op_counts['load'][op[0]] += 1

        print(f"\nCycle breakdown:")
        print(f"  LOAD idle, VALU busy: {load_idle} cycles")
        print(f"  VALU idle, LOAD busy: {valu_idle} cycles")
        print(f"  Both busy: {both_busy} cycles")
        print(f"  Both idle: {both_idle} cycles")

        print(f"\nVALU operations ({sum(op_counts['valu'].values())} total):")
        for op, count in sorted(op_counts['valu'].items(), key=lambda x: -x[1]):
            print(f"  {op}: {count}")

        print(f"\nLOAD operations ({sum(op_counts['load'].values())} total):")
        for op, count in sorted(op_counts['load'].items(), key=lambda x: -x[1]):
            print(f"  {op}: {count}")

        # Analyze valu slot utilization distribution
        print(f"\nVALU slot utilization distribution:")
        valu_hist = defaultdict(int)
        for instr in phase_instrs:
            slots = len(instr.get('valu', []))
            valu_hist[slots] += 1
        for slots in range(7):
            count = valu_hist[slots]
            pct = count / len(phase_instrs) * 100
            bar = '█' * int(pct / 2)
            print(f"  {slots} slots: {count:>4} cycles ({pct:>5.1f}%) {bar}")

        # Find operations that could theoretically be overlapped
        print(f"\nPotential overlap analysis:")
        print(f"  LOAD has {load_idle} idle cycles while VALU is working")
        print(f"  During these {load_idle} cycles, LOAD could do:")
        print(f"    - {load_idle * 2} load operations (2 per cycle)")
        print(f"    - Or {load_idle * 2} vload operations")

    # Compare to gather loop
    print(f"\n{'='*70}")
    print("COMPARISON: GATHER LOOP (rounds 3-10)")
    print("="*70)

    loop_start, loop_end = 420, 564
    loop_instrs = kb.instrs[loop_start:loop_end+1]

    op_counts = defaultdict(lambda: defaultdict(int))
    load_idle = 0
    both_busy = 0

    for instr in loop_instrs:
        valu_ops = instr.get('valu', [])
        load_ops = instr.get('load', [])

        if len(valu_ops) > 0 and len(load_ops) > 0:
            both_busy += 1
        elif len(load_ops) == 0 and len(valu_ops) > 0:
            load_idle += 1

        for op in valu_ops:
            op_counts['valu'][op[0]] += 1
        for op in load_ops:
            op_counts['load'][op[0]] += 1

    print(f"\nCycle breakdown ({len(loop_instrs)} cycles total):")
    print(f"  Both VALU and LOAD busy: {both_busy} cycles ({both_busy/len(loop_instrs)*100:.1f}%)")
    print(f"  LOAD idle while VALU busy: {load_idle} cycles ({load_idle/len(loop_instrs)*100:.1f}%)")

    print(f"\nLOAD operations: {sum(op_counts['load'].values())}")
    for op, count in sorted(op_counts['load'].items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    print(f"\nKey insight:")
    print(f"  Gather loop achieves {both_busy/len(loop_instrs)*100:.1f}% overlap")
    print(f"  Setup phases achieve much lower overlap")
    print(f"  Total setup cycles: {407 + 336} = 743")
    print(f"  If we could match gather loop's efficiency...")
    print(f"  Target: < 1487 cycles, Current: 2240, Gap: 753")

if __name__ == '__main__':
    analyze_setup_phases()
