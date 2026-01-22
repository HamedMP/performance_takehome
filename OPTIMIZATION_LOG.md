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
