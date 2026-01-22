# Optimization Log

## Target
- **Recruiting threshold**: < 1,487 cycles
- **Best known**: < 1,363 cycles

## Baseline
- **147,734 cycles**

---

## Experiment 1: Loops + Basic VLIW Packing

**What I did:**
- Replaced full unrolling (4096 iterations at code-gen time) with actual runtime loops
- Packed independent operations into single VLIW bundles:
  - Two address calculations in parallel
  - Two loads in parallel (idx and val)
  - Two stores in parallel
  - Hash function: first two ops of each stage in parallel

**Result:** 114,784 cycles (1.29x speedup)

**Analysis:**
- Modest improvement from better instruction packing
- Still far from target - need much bigger gains
- The inner loop runs 256 × 16 = 4096 times
- Each iteration is ~28 cycles, which is reasonable for scalar

**Next steps:**
- Vectorization (SIMD) should be the big win - process 8 items at once
- Expected theoretical speedup: ~8x from vectorization

---

## Experiment 2: SIMD Vectorization

**What I did:**
- Process 8 batch items at once using VLEN=8 vector operations
- vload/vstore for contiguous index and value arrays
- Gather pattern for tree node values (4 cycles for 8 loads)
- valu for all arithmetic and hash computation
- vselect for conditional operations

**Result:** 18,027 cycles (8.2x speedup from baseline, 6.4x from Exp 1)

**Analysis:**
- Inner loop runs 512 times (32 per round × 16 rounds)
- ~35 cycles per iteration
- Gather operation (4 cycles) is a significant bottleneck
- Some redundant address calculations

**Next steps:**
- Remove redundant address calculations
- Process more items per iteration (16 or 32) to amortize loop overhead
- Software pipelining to overlap loads with computation

---

## Experiment 3: Process 16 Items Per Iteration

**What I did:**
- Use 2 sets of vector registers (A and B) to process 16 items per iteration
- Pack valu operations: 4-6 valu ops per cycle where possible
- Removed redundant address calculations (reuse for stores)
- Pre-broadcast forest_values_p outside loop

**Result:** 10,090 cycles (14.6x speedup from baseline, 1.79x from Exp 2)

**Analysis:**
- 256 iterations (16 items × 16 rounds)
- ~39 cycles per iteration
- Bottlenecks identified:
  - Gather: 8 cycles (limited by 2 loads/cycle)
  - vselect: 4 cycles (limited by 1 flow/cycle)
  - Hash: 12 cycles
  - Loop overhead: ~3 cycles

**Key insight:** The 4 vselect operations cost 4 cycles due to flow limit=1. Can we replace them with ALU ops?
- `1 if val%2==0 else 2` → `1 + (val & 1)` (eliminates 2 vselects!)
- `0 if idx>=n else idx` → `idx * (idx < n)` (eliminates 2 vselects!)

---

## Experiment 4: Eliminate vselect Bottleneck

**What I did:**
- Replaced `1 if val%2==0 else 2` → `1 + (val & 1)` using ALU ops
- Replaced `0 if idx>=n else idx` → `idx * (idx < n)` using multiply
- Eliminated all 4 vselect operations

**Result:** 9,322 cycles (15.8x speedup, 1.08x from Exp 3)

**Analysis:**
- Saved ~768 cycles (4 vselects × 256 iterations × ~0.75 cycles each)
- ~36 cycles per iteration now
- Breakdown: gather(8) + hash(12) + loads(2) + stores(2) + address(3) + index(4) + loop(3) ≈ 34-36 cycles

---

## Experiment 5: Process 32+ Items with Pipelining

**Status:** In progress

---

## Future Plan

1. ✅ **Experiment 2:** SIMD vectorization → 18,027 cycles
2. **Experiment 3:** Remove redundancies, better packing
3. **Experiment 4:** Process 16+ items per iteration
4. **Experiment 5:** Software pipelining / loop unrolling
5. **Experiment 6:** Fine-tune for hardware limits

---

## Results Summary

| Experiment | Cycles | Speedup vs Baseline | Speedup vs Previous |
|------------|--------|---------------------|---------------------|
| Baseline   | 147,734 | 1.00x | - |
| Exp 1: Loops + VLIW | 114,784 | 1.29x | 1.29x |
| Exp 2: SIMD | 18,027 | 8.20x | 6.37x |
| Exp 3: 16 items/iter | 10,090 | 14.6x | 1.79x |
| Exp 4: Eliminate vselect | 9,322 | 15.8x | 1.08x |
| Exp 5: Pack store+alu | 8,809 | 16.8x | 1.06x |
| Exp 6: Register persistence | 7,385 | 20.0x | 1.19x |
| Exp 7: Optimized load/store | 7,158 | 20.6x | 1.03x |
| Exp 8: Gather/hash pipelining | 4,998 | 29.6x | 1.43x |
| Exp 9: Round 0 optimization | 4,988 | 29.6x | 1.00x |
| Exp 10: 4-vector parallel hash | 4,892 | 30.2x | 1.02x |

---

## Experiment 10: 4-Vector Parallel Hash for Round 0

**What I did:**
- Process 4 vectors (32 items) at a time instead of 2
- Better utilize 6 valu slots per cycle:
  - XOR: 4 ops in 1 cycle
  - Hash: Pipeline stages across 4 vectors (6 ops/cycle for tmp1,tmp2)
  - Index: 6 ops/cycle where possible

**Result:** 4,892 cycles (30.2x speedup, 1.02x from previous)

**Analysis:**
- Saved ~96 cycles in round 0
- Better valu slot utilization
- Still limited by dependencies in hash stages

---

## Experiment 9: Round 0 Single Load

**What I did:**
- All items start with idx=0, so they all access tree[0]
- Instead of 256 gathers, load tree[0] once and broadcast to all items
- Process 2 vectors at a time (like main loop) for round 0 separately
- Main loop now runs 15 rounds (1-15) instead of 16

**Result:** 4,988 cycles (29.6x speedup, tiny 0.2% improvement)

**Analysis:**
- Saved ~10 cycles by eliminating gather for round 0
- The savings are minimal because the non-pipelined processing for round 0 still takes significant time
- Round 0 without pipelining: ~290 cycles
- Original pipelined round: ~312 cycles
- Net savings: ~22 cycles minus setup overhead

---

## Experiment 9b: Round 1 Arithmetic Computation (FAILED)

**What I did:**
- After round 0, indices are 1 or 2 (only 2 unique values)
- Loaded tree[1] and tree[2], computed node_val = tree1 + (idx-1) * (tree2-tree1)
- This eliminates gather for round 1

**Result:** 5,030 cycles (WORSE than before!)

**Analysis:**
- The arithmetic to compute node_val adds 3 extra cycles per batch
- Lost the pipelining benefit (gather was overlapped with hash of previous batch)
- Net result: 3 extra cycles × 16 batches = 48 cycles added
- Gather savings: ~5 cycles (8 gather cycles - 3 setup cycles) × 16 batches = not enough to compensate
- **Reverted this change**

**Key Learning:** Eliminating gather operations only helps if we can maintain pipelining overlap. The gather-hash overlap is critical to current performance.

---

## Analysis: Theoretical Minimum

For 256 items × 16 rounds:
- 4096 gather loads at 2/cycle = **2048 cycles minimum**
- Target is 1,487 cycles - LESS than gather minimum!

This means Claude must have found a way to:
1. Reduce gather operations
2. Process more loads per cycle somehow
3. Use algorithmic shortcuts

Current bottleneck analysis per 16-item batch:
- Gather: 8 cycles (limited by 2 loads/cycle)
- Hash: 12 cycles (limited by valu dependencies)
- Index: 5 cycles
- With current pipeline: ~17 cycles per batch × 16 batches × 16 rounds ≈ 4352 cycles

---

## Key Discovery: Index Pattern Repetition

**Analysis of unique indices per round:**
```
Round | Unique Indices | Max Index
------|----------------|----------
   0  |        1       |       0
   1  |        2       |       2
   2  |        4       |       6
   3  |        8       |      14
   4  |       16       |      30
   5  |       32       |      62
   6  |       63       |     126
   7  |      112       |     254
   8  |      162       |     510
   9  |      206       |    1021
  10  |        1       |       0   <-- ALL wrap to 0!
  11  |        2       |       2
  12  |        4       |       6
  13  |        8       |      14
  14  |       16       |      30
  15  |       32       |      62
```

**Key insight:** After round 9, all items have indices that exceed n_nodes (1023) after the idx*2+1/2 formula, causing ALL 256 items to wrap back to idx=0. This means:
- Rounds 10-15 repeat the same pattern as rounds 0-5
- Rounds 0 and 10: Only 1 unique index → could use 1 load + broadcast
- Rounds 1-5 and 11-15: 2-32 unique indices → could preload and select

**Potential savings:** If we could exploit this for all affected rounds, we could eliminate ~2946 of 4096 gather operations, saving ~1473 cycles theoretically.

---

## Experiment 11: Round 10 Special Handling (FAILED)

**What I did:**
- Attempted to restructure kernel to handle rounds 0, 1-9, 10, 11-15 separately
- Created helper function `run_pipelined_rounds(start, end)` to avoid code duplication
- Added special handling for round 10 (same as round 0)

**Result:** Correctness failure on round 1

**Analysis:**
- The refactoring introduced a bug related to loop control or variable scoping
- Complexity of nested functions with closures made debugging difficult
- Reverted to working version (4998 cycles)

**Key Learning:** Complex refactoring needs more careful testing. The current pipelined structure is fragile and hard to modify without introducing bugs.
