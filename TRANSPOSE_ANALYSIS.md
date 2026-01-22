# Transpose Computation Analysis

The Anthropic blog mentioned that Claude found the solution by "transposing the entire computation rather than figuring out how to transpose the data."

## Current State
- **Cycles**: 2,240
- **Target**: < 1,487 cycles
- **Gap**: 753 cycles (33.6% reduction needed)

## Current Computation Structure

```
for round in [0..15]:           # Outer loop: rounds
    for batch in [0..7]:        # Middle: batches of 4 vectors (32 items)
        gather tree[idx]        # LOAD: 16 cycles
        XOR val with tree_val   # VALU: overlapped
        hash(val)               # VALU: overlapped
        update idx              # VALU: overlapped
```

Key insight: Gather uses LOAD, hash uses VALU - they OVERLAP for efficiency.

## Interpretation 1: Swap Round and Batch Order

**Transposed structure:**
```
for batch in [0..7]:            # Outer: batches
    for round in [0..15]:       # Inner: rounds
        process(batch, round)
```

**Analysis:**
- Each batch would go through all 16 rounds before next batch
- Doesn't help: still need same number of gathers
- Loses inter-batch pipelining within rounds
- **Verdict: WORSE**

## Interpretation 2: Item-Major Processing (One Item Through All Rounds)

**Structure:**
```
for item in [0..255]:
    for round in [0..15]:
        val = hash(val ^ tree[idx])
        idx = update(idx)
```

**Analysis:**
- Loses SIMD parallelism completely (processing 1 item at a time)
- Would be ~256x slower
- **Verdict: TERRIBLE**

## Interpretation 3: Round-Staggered Processing

**Structure:**
Split items into groups at different rounds, process interleaved:
- Group A at round k
- Group B at round k-1

**Analysis:**
- Could potentially share LOAD slots between groups
- But VALU is still the constraint for hash
- Complexity increases significantly
- **Verdict: UNCLEAR, needs more analysis**

## Interpretation 4: Preload During LOAD-Idle Rounds

**Key Observation:**
Rounds 0-2 and 11-13 don't use LOAD for gather (broadcast/arithmetic/selection).
During these ~600 cycles, LOAD engine is mostly IDLE!

**Opportunity:**
- Preload tree values for rounds 3-5 during rounds 0-2
- Preload tree values for rounds 14-15 during rounds 11-13

**Tree values needed:**
- Round 3: tree[7-14] = 8 values
- Round 4: tree[15-30] = 16 values
- Round 5: tree[31-62] = 32 values
- Total: 56 values for rounds 3-5

**Preload cost:** 56 values / 2 loads per cycle = 28 cycles

**Problem:** Selection still uses VALU, competing with hash!
- Gather: 16 LOAD cycles + 12 VALU cycles (overlapped) = 16 cycles
- Selection: 0 LOAD cycles + 18+ VALU cycles (sequential with hash) = 18+ cycles

**Verdict: Selection still WORSE than pipelined gather**

## Interpretation 5: Speculative Execution

**Idea:** Compute both possible paths through tree, select correct one later.

After round 0, items go to idx=1 or idx=2. We could:
1. Compute hash(val ^ tree[1]) speculatively
2. Compute hash(val ^ tree[2]) speculatively
3. Select correct result based on round 0 outcome

**Analysis:**
- Doubles VALU work (2x hash)
- VALU is already bottleneck
- **Verdict: WORSE**

## Interpretation 6: Sort Items by Tree Path

**Idea:** After round 2, sort items by their index (3, 4, 5, or 6).
Sorted items with same index are contiguous → vload instead of gather!

**After sorting:**
- Group 0: all items with idx=3 (access tree[3])
- Group 1: all items with idx=4 (access tree[4])
- etc.

**Benefits:**
- Round 3: broadcast per group (no gather!)
- Round 4: 2-way selection per group
- Round 5: 4-way selection per group
- etc.

**Costs:**
- Sorting 256 items: ~200+ cycles (scatter operations)
- Re-sorting needed? Partially - groups remain grouped but subdivide

**Key insight:** Items that share an index in round k will have indices that differ by at most 2 in round k+1!
- idx=3 → children are 7, 8
- idx=4 → children are 9, 10
- etc.

So sorting once preserves partial order!

**Potential savings:**
- Rounds 3-9: 7 rounds × (128 gather - ~80 selection) = ~336 cycles saved?
- Minus sorting cost: ~200 cycles
- Net: ~136 cycles saved?

**Verdict: PROMISING but complex to implement**

## Interpretation 7: Transpose the Data Layout

**Idea:** Store data differently for better access patterns.

Current: `values[item]`, `indices[item]`
Transposed: Group by tree path?

**Analysis:**
- Tree path depends on hash results
- Hash results depend on input values
- Can't predict paths ahead of time
- **Verdict: NOT APPLICABLE**

## Interpretation 8: Multi-Round Fusion

**Idea:** Fuse the computation of multiple rounds together.

For rounds 0-2, we already do this implicitly (no gather, sequential processing).

What if we fused rounds 3-4?
- Round 3: needs tree[7-14]
- Round 4: needs tree[15-30]

If we knew which round-4 values we'd need based on round-3 outcomes...
We could load tree[7] and tree[15-16] in the same batch!

**Structure:**
```
For items at idx=3 after round 2:
  - Round 3 uses tree[3], goes to idx=7 or 8
  - Round 4 uses tree[7] or tree[8], goes to idx=15-18

Preload: tree[3], tree[7], tree[8], tree[15-18]
Process both rounds with selection from preloaded values
```

**Analysis:**
- Preload 7 values per "super-batch"
- Process 2 rounds with selection
- Still uses VALU for selection

**Verdict: Still limited by VALU contention**

## The Core Problem

**Selection vs Gather Trade-off:**
- Gather: Uses LOAD (16 cycles), VALU free for hash overlap
- Selection: Uses VALU (10+ cycles), LOAD free but VALU serialized with hash

The gather-hash overlap is SO efficient that alternatives struggle to beat it.

## Potential Breakthrough Ideas

### Idea A: Use ALU for Hash
The scalar ALU has 12 slots/cycle. Could we de-vectorize some hash computation?
- VALU: 6 slots, processes 8 elements at once
- ALU: 12 slots, processes 12 scalars at once

If we could split hash work between VALU and ALU...
But hash operates on vectors; splitting would require extract/insert operations.

### Idea B: Pipeline Across Rounds
Currently: gather(round k) overlapped with hash(round k-1)
Extended: gather(round k) overlapped with hash(round k-1) AND hash(round k-2)?

If we had 3 rounds in flight, we could better utilize VALU during gather.

### Idea C: Reduce Hash Cycles
The hash function takes 12 VALU cycles per batch.
Can we reduce this?
- multiply_add already used for stages 0, 2, 4
- Stages 1, 3, 5 each need 3 cycles (tmp1, tmp2, combine)

What if we restructured stages 1, 3, 5?
```
Stage 1: val = (val >> 12) ^ (val >> 17) ^ val
       = val ^ (val >> 12) ^ (val >> 17)
```
This needs 2 shifts and 2 XORs = 4 ops per vector.
With 4 vectors: 16 ops = 3 cycles (6 VALU slots).

Can we parallelize the shifts and XORs better?

### Idea D: Preload + Scatter Optimization
During rounds 0-2 (LOAD idle), preload ALL tree values we'll need for rounds 3-9.
Tree[7-1022] = 1016 values. Too many to store in scratch (1536 words total)!

But we could preload tree[7-126] (120 values) for rounds 3-6.
Then rounds 3-6 could use selection, and LOAD is free for... something else?

## Next Steps

1. Implement preload during rounds 0-2
2. Measure actual cycle savings from selection with preloaded values
3. Explore ALU hash offloading
4. Consider sorting approach if selection proves viable
