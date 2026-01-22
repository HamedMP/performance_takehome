"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def build_packed(self, slots: list[tuple[str, tuple]]):
        """
        Pack slots into VLIW bundles respecting slot limits.
        Uses greedy packing - adds each slot to the current bundle if possible,
        otherwise starts a new bundle.
        """
        instrs = []
        current_bundle = {}

        for engine, slot in slots:
            if engine == "debug":
                # Debug slots don't count towards cycle, just append
                if current_bundle:
                    instrs.append(current_bundle)
                    current_bundle = {}
                instrs.append({engine: [slot]})
                continue

            # Check if we can add to current bundle
            current_count = len(current_bundle.get(engine, []))
            if current_count < SLOT_LIMITS.get(engine, 1):
                if engine not in current_bundle:
                    current_bundle[engine] = []
                current_bundle[engine].append(slot)
            else:
                # Start new bundle
                if current_bundle:
                    instrs.append(current_bundle)
                current_bundle = {engine: [slot]}

        if current_bundle:
            instrs.append(current_bundle)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Heavily pipelined SIMD kernel. Overlaps gather with hash computation
        and keeps data in scratch across the entire loop.
        """
        # For 256 items, we need 32 vectors. Process 2 at a time (16 items).
        NUM_BATCHES = batch_size // (VLEN * 2)  # 16 batches of 16 items
        ITEMS_PER_ITER = VLEN * 2

        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load memory layout params
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        eleven_const = self.scratch_const(11)
        thirteen_const = self.scratch_const(13)
        vlen_const = self.scratch_const(VLEN)
        items_const = self.scratch_const(ITEMS_PER_ITER)

        # Persistent vectors for ALL 256 items (32 vectors = 256 scratch words each for idx and val)
        # This allows us to avoid loading/storing between rounds!
        all_idx = [self.alloc_scratch(f"all_idx_{i}", VLEN) for i in range(batch_size // VLEN)]
        all_val = [self.alloc_scratch(f"all_val_{i}", VLEN) for i in range(batch_size // VLEN)]

        # Working vectors
        v_node_val_a = self.alloc_scratch("v_node_val_a", VLEN)
        v_addr_a = self.alloc_scratch("v_addr_a", VLEN)
        v_tmp1_a = self.alloc_scratch("v_tmp1_a", VLEN)
        v_tmp2_a = self.alloc_scratch("v_tmp2_a", VLEN)
        v_tmp3_a = self.alloc_scratch("v_tmp3_a", VLEN)

        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)
        v_addr_b = self.alloc_scratch("v_addr_b", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_tmp3_b = self.alloc_scratch("v_tmp3_b", VLEN)

        # Broadcast constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        v_hash_consts = []
        for hi in range(len(HASH_STAGES)):
            v_c1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            v_hash_consts.append((v_c1, v_c3))

        # For multiply_add optimization: stages 0, 2, 4 use (val + c1) + (val << shift) = val * (1 + 2^shift) + c1
        # Stage 0: 1 + 2^12 = 4097, Stage 2: 1 + 2^5 = 33, Stage 4: 1 + 2^3 = 9
        v_mul_4097 = self.alloc_scratch("v_mul_4097", VLEN)
        v_mul_33 = self.alloc_scratch("v_mul_33", VLEN)
        v_mul_9 = self.alloc_scratch("v_mul_9", VLEN)
        mul_consts = [(0, v_mul_4097, 4097), (2, v_mul_33, 33), (4, v_mul_9, 9)]

        round_counter = self.alloc_scratch("round_counter")
        loop_cond = self.alloc_scratch("loop_cond")
        addr_tmp = self.alloc_scratch("addr_tmp")
        # Additional address registers for pipelined stores
        addr_tmp3 = self.alloc_scratch("addr_tmp3")
        addr_tmp4 = self.alloc_scratch("addr_tmp4")

        self.add("flow", ("pause",))

        # Initialize broadcast vectors
        self.instrs.append({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_addr = self.scratch_const(val1)
            c3_addr = self.scratch_const(val3)
            self.instrs.append({"valu": [
                ("vbroadcast", v_hash_consts[hi][0], c1_addr),
                ("vbroadcast", v_hash_consts[hi][1], c3_addr),
            ]})

        # Broadcast multiply_add constants
        for hi, v_mul, mul_val in mul_consts:
            mul_addr = self.scratch_const(mul_val)
            self.instrs.append({"valu": [("vbroadcast", v_mul, mul_addr)]})

        # Map stages to multiply_add vectors: stages 0, 2, 4 can use multiply_add
        mul_add_stages = {0: v_mul_4097, 2: v_mul_33, 4: v_mul_9}

        # Pre-compute all load/store addresses as constants
        addr_consts = [self.scratch_const(i * VLEN) for i in range(batch_size // VLEN)]
        addr_tmp2 = self.alloc_scratch("addr_tmp2")

        # Load ALL indices and values into scratch (pipelined: overlap addr compute with loads)
        # Indices: compute first pair of addresses
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["inp_indices_p"], addr_consts[0]),
            ("+", addr_tmp2, self.scratch["inp_indices_p"], addr_consts[1]),
        ]})
        # Pipelined: compute next addresses while loading current
        for i in range(2, batch_size // VLEN, 2):
            self.instrs.append({"alu": [
                ("+", addr_tmp3, self.scratch["inp_indices_p"], addr_consts[i]),
                ("+", addr_tmp4, self.scratch["inp_indices_p"], addr_consts[i+1]),
            ], "load": [
                ("vload", all_idx[i-2], addr_tmp),
                ("vload", all_idx[i-1], addr_tmp2),
            ]})
            addr_tmp, addr_tmp3 = addr_tmp3, addr_tmp
            addr_tmp2, addr_tmp4 = addr_tmp4, addr_tmp2
        # Final loads for last pair
        self.instrs.append({"load": [
            ("vload", all_idx[-2], addr_tmp),
            ("vload", all_idx[-1], addr_tmp2),
        ]})

        # Values: same pattern
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["inp_values_p"], addr_consts[0]),
            ("+", addr_tmp2, self.scratch["inp_values_p"], addr_consts[1]),
        ]})
        for i in range(2, batch_size // VLEN, 2):
            self.instrs.append({"alu": [
                ("+", addr_tmp3, self.scratch["inp_values_p"], addr_consts[i]),
                ("+", addr_tmp4, self.scratch["inp_values_p"], addr_consts[i+1]),
            ], "load": [
                ("vload", all_val[i-2], addr_tmp),
                ("vload", all_val[i-1], addr_tmp2),
            ]})
            addr_tmp, addr_tmp3 = addr_tmp3, addr_tmp
            addr_tmp2, addr_tmp4 = addr_tmp4, addr_tmp2
        self.instrs.append({"load": [
            ("vload", all_val[-2], addr_tmp),
            ("vload", all_val[-1], addr_tmp2),
        ]})

        # We need separate node_val registers for pipelining (double buffering)
        v_node_val_a2 = self.alloc_scratch("v_node_val_a2", VLEN)
        v_node_val_b2 = self.alloc_scratch("v_node_val_b2", VLEN)
        v_addr_a2 = self.alloc_scratch("v_addr_a2", VLEN)
        v_addr_b2 = self.alloc_scratch("v_addr_b2", VLEN)

        # ============================================
        # ROUND 0: All items have idx=0
        # Use broadcast instead of gather (big optimization!)
        # ============================================
        v_tree0 = self.alloc_scratch("v_tree0", VLEN)
        tree0_scalar = self.alloc_scratch("tree0_scalar")

        # Load tree[0] and broadcast
        self.instrs.append({"load": [("load", tree0_scalar, self.scratch["forest_values_p"])]})
        self.instrs.append({"valu": [("vbroadcast", v_tree0, tree0_scalar)]})

        # Process all batches for round 0 (no gathers needed!)
        # Pack 6 vectors at a time for better valu utilization (6 valu slots)
        v_tmp1_c = self.alloc_scratch("v_tmp1_c", VLEN)
        v_tmp1_d = self.alloc_scratch("v_tmp1_d", VLEN)
        v_tmp1_e = self.alloc_scratch("v_tmp1_e", VLEN)
        v_tmp1_f = self.alloc_scratch("v_tmp1_f", VLEN)
        v_tmp2_c = self.alloc_scratch("v_tmp2_c", VLEN)
        v_tmp2_d = self.alloc_scratch("v_tmp2_d", VLEN)
        v_tmp2_e = self.alloc_scratch("v_tmp2_e", VLEN)
        v_tmp2_f = self.alloc_scratch("v_tmp2_f", VLEN)

        # Process 6 vectors (48 items) at a time for better valu utilization
        # 32 vectors / 6 = 5.33, so process in chunks: 6, 6, 6, 6, 6, 2
        vec_batches = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 6), (30, 2)]

        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            # XOR with tree[0]
            xor_ops = [("^", v_val, v_val, v_tree0) for v_idx, v_val in vecs]
            self.instrs.append({"valu": xor_ops})

            # Hash (6 stages) - use multiply_add for stages 0, 2, 4
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi]

                if hi in mul_add_stages:
                    # Stages 0, 2, 4: val = val * multiplier + c1 (single multiply_add)
                    v_mul = mul_add_stages[hi]
                    mul_add_ops = [("multiply_add", v_val, v_val, v_mul, v_c1) for (v_idx, v_val) in vecs]
                    self.instrs.append({"valu": mul_add_ops[:6]})
                    if len(mul_add_ops) > 6:
                        self.instrs.append({"valu": mul_add_ops[6:]})
                else:
                    # Stages 1, 3, 5: use standard tmp1/tmp2 pattern
                    ops1 = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if len(ops1) < 6:
                            ops1.append((op1, t1, v_val, v_c1))
                        if len(ops1) < 6:
                            ops1.append((op3, t2, v_val, v_c3))
                    self.instrs.append({"valu": ops1})

                    remaining_tmp = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if i * 2 >= 6:
                            remaining_tmp.append((op1, t1, v_val, v_c1))
                        if i * 2 + 1 >= 6:
                            remaining_tmp.append((op3, t2, v_val, v_c3))
                    if remaining_tmp:
                        self.instrs.append({"valu": remaining_tmp})

                    combine_ops = [(op2, v_val, t1, t2) for (v_idx, v_val), t1, t2 in zip(vecs, tmp1_list, tmp2_list)]
                    self.instrs.append({"valu": combine_ops})

            # Index computation: idx=0, so new_idx = 0*2 + (1 or 2) = 1 + (val & 1)
            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_ops = [("+", v_idx, v_one, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_ops})
            # No wrapping check needed since 1 and 2 are < n_nodes

        # ============================================
        # ROUND 1: All indices are 1 or 2 (use arithmetic instead of gather)
        # ============================================
        tree1_scalar = self.alloc_scratch("tree1_scalar")
        tree2_scalar = self.alloc_scratch("tree2_scalar")
        diff_scalar = self.alloc_scratch("diff_scalar")
        v_tree1 = self.alloc_scratch("v_tree1", VLEN)
        v_diff = self.alloc_scratch("v_diff", VLEN)
        v_offset = self.alloc_scratch("v_offset", VLEN)
        v_node_tmp = self.alloc_scratch("v_node_tmp", VLEN)

        # Load tree[1] and tree[2]
        self.instrs.append({"alu": [("+", addr_tmp, self.scratch["forest_values_p"], one_const)]})
        self.instrs.append({"load": [("load", tree1_scalar, addr_tmp)]})
        self.instrs.append({"alu": [("+", addr_tmp, self.scratch["forest_values_p"], two_const)]})
        self.instrs.append({"load": [("load", tree2_scalar, addr_tmp)]})

        # Compute diff = tree2 - tree1 and broadcast
        self.instrs.append({"alu": [("-", diff_scalar, tree2_scalar, tree1_scalar)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree1, tree1_scalar),
            ("vbroadcast", v_diff, diff_scalar),
        ]})

        # Process all vectors for round 1
        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            # Compute node_val = tree1 + (idx - 1) * diff for each item
            # First: offset = idx - 1
            offset_ops = [("-", t1, v_idx, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": offset_ops})

            # node_val = tree1 + offset * diff = tree1 + tmp1 * diff
            mul_ops = [("*", t2, t1, v_diff) for t1, t2 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs])]
            self.instrs.append({"valu": mul_ops})

            add_node_ops = [("+", t1, v_tree1, t2) for t1, t2 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs])]
            self.instrs.append({"valu": add_node_ops})

            # XOR: val = val ^ node_val (node_val is in tmp1_list now)
            xor_ops = [("^", v_val, v_val, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": xor_ops})

            # Hash (6 stages) - use multiply_add for stages 0, 2, 4
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi]

                if hi in mul_add_stages:
                    v_mul = mul_add_stages[hi]
                    mul_add_ops = [("multiply_add", v_val, v_val, v_mul, v_c1) for (v_idx, v_val) in vecs]
                    self.instrs.append({"valu": mul_add_ops[:6]})
                    if len(mul_add_ops) > 6:
                        self.instrs.append({"valu": mul_add_ops[6:]})
                else:
                    ops1 = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if len(ops1) < 6:
                            ops1.append((op1, t1, v_val, v_c1))
                        if len(ops1) < 6:
                            ops1.append((op3, t2, v_val, v_c3))
                    self.instrs.append({"valu": ops1})

                    remaining_tmp = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if i * 2 >= 6:
                            remaining_tmp.append((op1, t1, v_val, v_c1))
                        if i * 2 + 1 >= 6:
                            remaining_tmp.append((op3, t2, v_val, v_c3))
                    if remaining_tmp:
                        self.instrs.append({"valu": remaining_tmp})

                    combine_ops = [(op2, v_val, t1, t2) for (v_idx, v_val), t1, t2 in zip(vecs, tmp1_list, tmp2_list)]
                    self.instrs.append({"valu": combine_ops})

            # Index computation: new_idx = idx * 2 + (1 + (val & 1))
            # idx is 1 or 2, so idx*2 is 2 or 4
            mul_idx_ops = [("*", v_idx, v_idx, v_two) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_one_ops = [("+", t1, v_one, t1) for t1 in tmp1_list[:num_vecs]]
            self.instrs.append({"valu": add_one_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})

            # Check idx < n_nodes and wrap if needed
            cmp_ops = [("<", t1, v_idx, v_n_nodes) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": cmp_ops})

            wrap_ops = [("*", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": wrap_ops})

        # ============================================
        # ROUNDS 2 to rounds-1: Regular pipelined loop
        # ============================================
        self.instrs.append({"load": [("const", round_counter, 2)]})  # Start at round 2
        outer_loop_start = len(self.instrs)

        # PIPELINED INNER LOOP
        # Overlap gather of batch N with hash/index computation of batch N-1

        # Prologue: Gather batch 0
        ia0, ib0 = 0, 1
        self.instrs.append({"valu": [
            ("+", v_addr_a, all_idx[ia0], v_forest_p),
            ("+", v_addr_b, all_idx[ib0], v_forest_p),
        ]})
        for i in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load_offset", v_node_val_a, v_addr_a, i),
                ("load_offset", v_node_val_a, v_addr_a, i + 1),
            ]})
        for i in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load_offset", v_node_val_b, v_addr_b, i),
                ("load_offset", v_node_val_b, v_addr_b, i + 1),
            ]})

        # Steady state: batches 1 to NUM_BATCHES-1
        for batch in range(1, NUM_BATCHES):
            # Previous batch indices
            ia_prev, ib_prev = (batch - 1) * 2, (batch - 1) * 2 + 1
            v_idx_prev_a, v_val_prev_a = all_idx[ia_prev], all_val[ia_prev]
            v_idx_prev_b, v_val_prev_b = all_idx[ib_prev], all_val[ib_prev]

            # Current batch indices
            ia_cur, ib_cur = batch * 2, batch * 2 + 1
            v_idx_cur_a, v_val_cur_a = all_idx[ia_cur], all_val[ia_cur]
            v_idx_cur_b, v_val_cur_b = all_idx[ib_cur], all_val[ib_cur]

            # Alternate node_val registers for double buffering
            if batch % 2 == 1:
                nv_a, nv_b = v_node_val_a2, v_node_val_b2
                nv_a_prev, nv_b_prev = v_node_val_a, v_node_val_b
                addr_cur_a, addr_cur_b = v_addr_a2, v_addr_b2
            else:
                nv_a, nv_b = v_node_val_a, v_node_val_b
                nv_a_prev, nv_b_prev = v_node_val_a2, v_node_val_b2
                addr_cur_a, addr_cur_b = v_addr_a, v_addr_b

            # XOR for previous batch (uses nv_a_prev, nv_b_prev)
            # Compute addresses for current batch (can be in parallel)
            self.instrs.append({"valu": [
                ("^", v_val_prev_a, v_val_prev_a, nv_a_prev),
                ("^", v_val_prev_b, v_val_prev_b, nv_b_prev),
                ("+", addr_cur_a, v_idx_cur_a, v_forest_p),
                ("+", addr_cur_b, v_idx_cur_b, v_forest_p),
            ]})

            # Gather current batch (8 cycles) overlapped with hash stages 0-4 for previous batch
            # Using multiply_add for stages 0, 2, 4 allows fitting more hash work in gather window
            # Schedule:
            # gi=0: gather A[0,1], stage 0 (multiply_add)
            # gi=1: gather A[2,3], stage 1 part 1 (tmp1, tmp2)
            # gi=2: gather A[4,5], stage 1 part 2 (combine)
            # gi=3: gather A[6,7], stage 2 (multiply_add)
            # gi=4: gather B[0,1], stage 3 part 1 (tmp1, tmp2)
            # gi=5: gather B[2,3], stage 3 part 2 (combine)
            # gi=6: gather B[4,5], stage 4 (multiply_add) + idx*2
            # gi=7: gather B[6,7], stage 5 part 1 (tmp1, tmp2)

            v_c1_0 = v_hash_consts[0][0]
            v_c1_2 = v_hash_consts[2][0]
            v_c1_4 = v_hash_consts[4][0]
            h_stage1 = HASH_STAGES[1]
            v_c1_1, v_c3_1 = v_hash_consts[1]
            h_stage3 = HASH_STAGES[3]
            v_c1_3, v_c3_3 = v_hash_consts[3]
            h_stage5 = HASH_STAGES[5]
            v_c1_5, v_c3_5 = v_hash_consts[5]

            # gi=0: gather A[0,1], stage 0 (multiply_add)
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 0),
                ("load_offset", nv_a, addr_cur_a, 1),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_4097, v_c1_0),
            ]})
            # gi=1: gather A[2,3], stage 1 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 2),
                ("load_offset", nv_a, addr_cur_a, 3),
            ], "valu": [
                (h_stage1[0], v_tmp1_a, v_val_prev_a, v_c1_1),
                (h_stage1[3], v_tmp2_a, v_val_prev_a, v_c3_1),
                (h_stage1[0], v_tmp1_b, v_val_prev_b, v_c1_1),
                (h_stage1[3], v_tmp2_b, v_val_prev_b, v_c3_1),
            ]})
            # gi=2: gather A[4,5], stage 1 part 2
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 4),
                ("load_offset", nv_a, addr_cur_a, 5),
            ], "valu": [
                (h_stage1[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage1[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})
            # gi=3: gather A[6,7], stage 2 (multiply_add)
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 6),
                ("load_offset", nv_a, addr_cur_a, 7),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_33, v_c1_2),
            ]})
            # gi=4: gather B[0,1], stage 3 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 0),
                ("load_offset", nv_b, addr_cur_b, 1),
            ], "valu": [
                (h_stage3[0], v_tmp1_a, v_val_prev_a, v_c1_3),
                (h_stage3[3], v_tmp2_a, v_val_prev_a, v_c3_3),
                (h_stage3[0], v_tmp1_b, v_val_prev_b, v_c1_3),
                (h_stage3[3], v_tmp2_b, v_val_prev_b, v_c3_3),
            ]})
            # gi=5: gather B[2,3], stage 3 part 2
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 2),
                ("load_offset", nv_b, addr_cur_b, 3),
            ], "valu": [
                (h_stage3[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage3[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})
            # gi=6: gather B[4,5], stage 4 (multiply_add) + idx*2
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 4),
                ("load_offset", nv_b, addr_cur_b, 5),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_9, v_c1_4),
                ("*", v_idx_prev_a, v_idx_prev_a, v_two),
                ("*", v_idx_prev_b, v_idx_prev_b, v_two),
            ]})
            # gi=7: gather B[6,7], stage 5 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 6),
                ("load_offset", nv_b, addr_cur_b, 7),
            ], "valu": [
                (h_stage5[0], v_tmp1_a, v_val_prev_a, v_c1_5),
                (h_stage5[3], v_tmp2_a, v_val_prev_a, v_c3_5),
                (h_stage5[0], v_tmp1_b, v_val_prev_b, v_c1_5),
                (h_stage5[3], v_tmp2_b, v_val_prev_b, v_c3_5),
            ]})

            # Stage 5 part 2 (no longer overlapped with gather)
            self.instrs.append({"valu": [
                (h_stage5[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage5[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})

            # Index computation (idx*2 already done above)
            self.instrs.append({"valu": [
                ("&", v_tmp1_a, v_val_prev_a, v_one),
                ("&", v_tmp1_b, v_val_prev_b, v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp3_a, v_one, v_tmp1_a), ("+", v_tmp3_b, v_one, v_tmp1_b),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx_prev_a, v_idx_prev_a, v_tmp3_a), ("+", v_idx_prev_b, v_idx_prev_b, v_tmp3_b),
            ]})
            self.instrs.append({"valu": [
                ("<", v_tmp1_a, v_idx_prev_a, v_n_nodes), ("<", v_tmp1_b, v_idx_prev_b, v_n_nodes),
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx_prev_a, v_idx_prev_a, v_tmp1_a), ("*", v_idx_prev_b, v_idx_prev_b, v_tmp1_b),
            ]})

        # Epilogue: Process last batch (NUM_BATCHES-1)
        ia_last, ib_last = (NUM_BATCHES - 1) * 2, (NUM_BATCHES - 1) * 2 + 1
        v_idx_last_a, v_val_last_a = all_idx[ia_last], all_val[ia_last]
        v_idx_last_b, v_val_last_b = all_idx[ib_last], all_val[ib_last]
        if (NUM_BATCHES - 1) % 2 == 1:
            nv_last_a, nv_last_b = v_node_val_a2, v_node_val_b2
        else:
            nv_last_a, nv_last_b = v_node_val_a, v_node_val_b

        self.instrs.append({"valu": [
            ("^", v_val_last_a, v_val_last_a, nv_last_a),
            ("^", v_val_last_b, v_val_last_b, nv_last_b),
        ]})
        # Hash (6 stages) - use multiply_add for stages 0, 2, 4
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            if hi in mul_add_stages:
                v_mul = mul_add_stages[hi]
                self.instrs.append({"valu": [
                    ("multiply_add", v_val_last_a, v_val_last_a, v_mul, v_c1),
                    ("multiply_add", v_val_last_b, v_val_last_b, v_mul, v_c1),
                ]})
            else:
                self.instrs.append({"valu": [
                    (op1, v_tmp1_a, v_val_last_a, v_c1), (op3, v_tmp2_a, v_val_last_a, v_c3),
                    (op1, v_tmp1_b, v_val_last_b, v_c1), (op3, v_tmp2_b, v_val_last_b, v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val_last_a, v_tmp1_a, v_tmp2_a),
                    (op2, v_val_last_b, v_tmp1_b, v_tmp2_b),
                ]})
        self.instrs.append({"valu": [
            ("&", v_tmp1_a, v_val_last_a, v_one), ("*", v_idx_last_a, v_idx_last_a, v_two),
            ("&", v_tmp1_b, v_val_last_b, v_one), ("*", v_idx_last_b, v_idx_last_b, v_two),
        ]})
        self.instrs.append({"valu": [
            ("+", v_tmp3_a, v_one, v_tmp1_a), ("+", v_tmp3_b, v_one, v_tmp1_b),
        ]})
        self.instrs.append({"valu": [
            ("+", v_idx_last_a, v_idx_last_a, v_tmp3_a), ("+", v_idx_last_b, v_idx_last_b, v_tmp3_b),
        ]})
        # Merge loop control with epilogue index computation
        self.instrs.append({"valu": [
            ("<", v_tmp1_a, v_idx_last_a, v_n_nodes), ("<", v_tmp1_b, v_idx_last_b, v_n_nodes),
        ], "alu": [("+", round_counter, round_counter, one_const)]})
        self.instrs.append({"valu": [
            ("*", v_idx_last_a, v_idx_last_a, v_tmp1_a), ("*", v_idx_last_b, v_idx_last_b, v_tmp1_b),
        ], "alu": [("<", loop_cond, round_counter, eleven_const)]})
        self.instrs.append({"flow": [("cond_jump", loop_cond, outer_loop_start)]})

        # ============================================
        # ROUND 11: All items have idx=0 (after round 10 wrapped all to 0)
        # Use broadcast instead of gather
        # ============================================
        self.instrs.append({"load": [("load", tree0_scalar, self.scratch["forest_values_p"])]})
        self.instrs.append({"valu": [("vbroadcast", v_tree0, tree0_scalar)]})

        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            xor_ops = [("^", v_val, v_val, v_tree0) for v_idx, v_val in vecs]
            self.instrs.append({"valu": xor_ops})

            # Hash (6 stages) - use multiply_add for stages 0, 2, 4
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi]

                if hi in mul_add_stages:
                    v_mul = mul_add_stages[hi]
                    mul_add_ops = [("multiply_add", v_val, v_val, v_mul, v_c1) for (v_idx, v_val) in vecs]
                    self.instrs.append({"valu": mul_add_ops[:6]})
                    if len(mul_add_ops) > 6:
                        self.instrs.append({"valu": mul_add_ops[6:]})
                else:
                    ops1 = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if len(ops1) < 6:
                            ops1.append((op1, t1, v_val, v_c1))
                        if len(ops1) < 6:
                            ops1.append((op3, t2, v_val, v_c3))
                    self.instrs.append({"valu": ops1})

                    remaining_tmp = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if i * 2 >= 6:
                            remaining_tmp.append((op1, t1, v_val, v_c1))
                        if i * 2 + 1 >= 6:
                            remaining_tmp.append((op3, t2, v_val, v_c3))
                    if remaining_tmp:
                        self.instrs.append({"valu": remaining_tmp})

                    combine_ops = [(op2, v_val, t1, t2) for (v_idx, v_val), t1, t2 in zip(vecs, tmp1_list, tmp2_list)]
                    self.instrs.append({"valu": combine_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_ops = [("+", v_idx, v_one, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_ops})

        # ============================================
        # ROUND 12: All indices are 1 or 2 (use arithmetic instead of gather)
        # ============================================
        self.instrs.append({"alu": [("+", addr_tmp, self.scratch["forest_values_p"], one_const)]})
        self.instrs.append({"load": [("load", tree1_scalar, addr_tmp)]})
        self.instrs.append({"alu": [("+", addr_tmp, self.scratch["forest_values_p"], two_const)]})
        self.instrs.append({"load": [("load", tree2_scalar, addr_tmp)]})
        self.instrs.append({"alu": [("-", diff_scalar, tree2_scalar, tree1_scalar)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree1, tree1_scalar),
            ("vbroadcast", v_diff, diff_scalar),
        ]})

        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            offset_ops = [("-", t1, v_idx, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": offset_ops})

            mul_ops = [("*", t2, t1, v_diff) for t1, t2 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs])]
            self.instrs.append({"valu": mul_ops})

            add_node_ops = [("+", t1, v_tree1, t2) for t1, t2 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs])]
            self.instrs.append({"valu": add_node_ops})

            xor_ops = [("^", v_val, v_val, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": xor_ops})

            # Hash (6 stages) - use multiply_add for stages 0, 2, 4
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v_c1, v_c3 = v_hash_consts[hi]

                if hi in mul_add_stages:
                    v_mul = mul_add_stages[hi]
                    mul_add_ops = [("multiply_add", v_val, v_val, v_mul, v_c1) for (v_idx, v_val) in vecs]
                    self.instrs.append({"valu": mul_add_ops[:6]})
                    if len(mul_add_ops) > 6:
                        self.instrs.append({"valu": mul_add_ops[6:]})
                else:
                    ops1 = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if len(ops1) < 6:
                            ops1.append((op1, t1, v_val, v_c1))
                        if len(ops1) < 6:
                            ops1.append((op3, t2, v_val, v_c3))
                    self.instrs.append({"valu": ops1})

                    remaining_tmp = []
                    for i, ((v_idx, v_val), t1, t2) in enumerate(zip(vecs, tmp1_list, tmp2_list)):
                        if i * 2 >= 6:
                            remaining_tmp.append((op1, t1, v_val, v_c1))
                        if i * 2 + 1 >= 6:
                            remaining_tmp.append((op3, t2, v_val, v_c3))
                    if remaining_tmp:
                        self.instrs.append({"valu": remaining_tmp})

                    combine_ops = [(op2, v_val, t1, t2) for (v_idx, v_val), t1, t2 in zip(vecs, tmp1_list, tmp2_list)]
                    self.instrs.append({"valu": combine_ops})

            mul_idx_ops = [("*", v_idx, v_idx, v_two) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_one_ops = [("+", t1, v_one, t1) for t1 in tmp1_list[:num_vecs]]
            self.instrs.append({"valu": add_one_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})

            cmp_ops = [("<", t1, v_idx, v_n_nodes) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": cmp_ops})

            wrap_ops = [("*", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": wrap_ops})

        # ============================================
        # ROUNDS 13-15: Third pipelined loop
        # ============================================
        self.instrs.append({"load": [("const", round_counter, 13)]})
        outer_loop_start_2 = len(self.instrs)

        # Prologue: Gather batch 0
        self.instrs.append({"valu": [
            ("+", v_addr_a, all_idx[0], v_forest_p),
            ("+", v_addr_b, all_idx[1], v_forest_p),
        ]})
        for i in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load_offset", v_node_val_a, v_addr_a, i),
                ("load_offset", v_node_val_a, v_addr_a, i + 1),
            ]})
        for i in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load_offset", v_node_val_b, v_addr_b, i),
                ("load_offset", v_node_val_b, v_addr_b, i + 1),
            ]})

        # Steady state
        for batch in range(1, NUM_BATCHES):
            ia_prev, ib_prev = (batch - 1) * 2, (batch - 1) * 2 + 1
            v_idx_prev_a, v_val_prev_a = all_idx[ia_prev], all_val[ia_prev]
            v_idx_prev_b, v_val_prev_b = all_idx[ib_prev], all_val[ib_prev]

            ia_cur, ib_cur = batch * 2, batch * 2 + 1
            v_idx_cur_a, v_val_cur_a = all_idx[ia_cur], all_val[ia_cur]
            v_idx_cur_b, v_val_cur_b = all_idx[ib_cur], all_val[ib_cur]

            if batch % 2 == 1:
                nv_a, nv_b = v_node_val_a2, v_node_val_b2
                nv_a_prev, nv_b_prev = v_node_val_a, v_node_val_b
                addr_cur_a, addr_cur_b = v_addr_a2, v_addr_b2
            else:
                nv_a, nv_b = v_node_val_a, v_node_val_b
                nv_a_prev, nv_b_prev = v_node_val_a2, v_node_val_b2
                addr_cur_a, addr_cur_b = v_addr_a, v_addr_b

            self.instrs.append({"valu": [
                ("^", v_val_prev_a, v_val_prev_a, nv_a_prev),
                ("^", v_val_prev_b, v_val_prev_b, nv_b_prev),
                ("+", addr_cur_a, v_idx_cur_a, v_forest_p),
                ("+", addr_cur_b, v_idx_cur_b, v_forest_p),
            ]})

            # Restructured pipelined hash with multiply_add for stages 0, 2, 4
            v_c1_0 = v_hash_consts[0][0]
            v_c1_2 = v_hash_consts[2][0]
            v_c1_4 = v_hash_consts[4][0]
            h_stage1 = HASH_STAGES[1]
            v_c1_1, v_c3_1 = v_hash_consts[1]
            h_stage3 = HASH_STAGES[3]
            v_c1_3, v_c3_3 = v_hash_consts[3]
            h_stage5 = HASH_STAGES[5]
            v_c1_5, v_c3_5 = v_hash_consts[5]

            # gi=0: gather A[0,1], stage 0 (multiply_add)
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 0),
                ("load_offset", nv_a, addr_cur_a, 1),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_4097, v_c1_0),
            ]})
            # gi=1: gather A[2,3], stage 1 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 2),
                ("load_offset", nv_a, addr_cur_a, 3),
            ], "valu": [
                (h_stage1[0], v_tmp1_a, v_val_prev_a, v_c1_1),
                (h_stage1[3], v_tmp2_a, v_val_prev_a, v_c3_1),
                (h_stage1[0], v_tmp1_b, v_val_prev_b, v_c1_1),
                (h_stage1[3], v_tmp2_b, v_val_prev_b, v_c3_1),
            ]})
            # gi=2: gather A[4,5], stage 1 part 2
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 4),
                ("load_offset", nv_a, addr_cur_a, 5),
            ], "valu": [
                (h_stage1[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage1[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})
            # gi=3: gather A[6,7], stage 2 (multiply_add)
            self.instrs.append({"load": [
                ("load_offset", nv_a, addr_cur_a, 6),
                ("load_offset", nv_a, addr_cur_a, 7),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_33, v_c1_2),
            ]})
            # gi=4: gather B[0,1], stage 3 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 0),
                ("load_offset", nv_b, addr_cur_b, 1),
            ], "valu": [
                (h_stage3[0], v_tmp1_a, v_val_prev_a, v_c1_3),
                (h_stage3[3], v_tmp2_a, v_val_prev_a, v_c3_3),
                (h_stage3[0], v_tmp1_b, v_val_prev_b, v_c1_3),
                (h_stage3[3], v_tmp2_b, v_val_prev_b, v_c3_3),
            ]})
            # gi=5: gather B[2,3], stage 3 part 2
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 2),
                ("load_offset", nv_b, addr_cur_b, 3),
            ], "valu": [
                (h_stage3[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage3[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})
            # gi=6: gather B[4,5], stage 4 (multiply_add) + idx*2
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 4),
                ("load_offset", nv_b, addr_cur_b, 5),
            ], "valu": [
                ("multiply_add", v_val_prev_a, v_val_prev_a, v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev_b, v_val_prev_b, v_mul_9, v_c1_4),
                ("*", v_idx_prev_a, v_idx_prev_a, v_two),
                ("*", v_idx_prev_b, v_idx_prev_b, v_two),
            ]})
            # gi=7: gather B[6,7], stage 5 part 1
            self.instrs.append({"load": [
                ("load_offset", nv_b, addr_cur_b, 6),
                ("load_offset", nv_b, addr_cur_b, 7),
            ], "valu": [
                (h_stage5[0], v_tmp1_a, v_val_prev_a, v_c1_5),
                (h_stage5[3], v_tmp2_a, v_val_prev_a, v_c3_5),
                (h_stage5[0], v_tmp1_b, v_val_prev_b, v_c1_5),
                (h_stage5[3], v_tmp2_b, v_val_prev_b, v_c3_5),
            ]})

            # Stage 5 part 2
            self.instrs.append({"valu": [
                (h_stage5[2], v_val_prev_a, v_tmp1_a, v_tmp2_a),
                (h_stage5[2], v_val_prev_b, v_tmp1_b, v_tmp2_b),
            ]})

            self.instrs.append({"valu": [
                ("&", v_tmp1_a, v_val_prev_a, v_one),
                ("&", v_tmp1_b, v_val_prev_b, v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp3_a, v_one, v_tmp1_a), ("+", v_tmp3_b, v_one, v_tmp1_b),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx_prev_a, v_idx_prev_a, v_tmp3_a), ("+", v_idx_prev_b, v_idx_prev_b, v_tmp3_b),
            ]})
            self.instrs.append({"valu": [
                ("<", v_tmp1_a, v_idx_prev_a, v_n_nodes), ("<", v_tmp1_b, v_idx_prev_b, v_n_nodes),
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx_prev_a, v_idx_prev_a, v_tmp1_a), ("*", v_idx_prev_b, v_idx_prev_b, v_tmp1_b),
            ]})

        # Epilogue for third loop
        ia_last2, ib_last2 = (NUM_BATCHES - 1) * 2, (NUM_BATCHES - 1) * 2 + 1
        v_idx_last2_a, v_val_last2_a = all_idx[ia_last2], all_val[ia_last2]
        v_idx_last2_b, v_val_last2_b = all_idx[ib_last2], all_val[ib_last2]
        if (NUM_BATCHES - 1) % 2 == 1:
            nv_last2_a, nv_last2_b = v_node_val_a2, v_node_val_b2
        else:
            nv_last2_a, nv_last2_b = v_node_val_a, v_node_val_b

        self.instrs.append({"valu": [
            ("^", v_val_last2_a, v_val_last2_a, nv_last2_a),
            ("^", v_val_last2_b, v_val_last2_b, nv_last2_b),
        ]})
        # Hash (6 stages) - use multiply_add for stages 0, 2, 4
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash_consts[hi]
            if hi in mul_add_stages:
                v_mul = mul_add_stages[hi]
                self.instrs.append({"valu": [
                    ("multiply_add", v_val_last2_a, v_val_last2_a, v_mul, v_c1),
                    ("multiply_add", v_val_last2_b, v_val_last2_b, v_mul, v_c1),
                ]})
            else:
                self.instrs.append({"valu": [
                    (op1, v_tmp1_a, v_val_last2_a, v_c1), (op3, v_tmp2_a, v_val_last2_a, v_c3),
                    (op1, v_tmp1_b, v_val_last2_b, v_c1), (op3, v_tmp2_b, v_val_last2_b, v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val_last2_a, v_tmp1_a, v_tmp2_a),
                    (op2, v_val_last2_b, v_tmp1_b, v_tmp2_b),
                ]})
        self.instrs.append({"valu": [
            ("&", v_tmp1_a, v_val_last2_a, v_one), ("*", v_idx_last2_a, v_idx_last2_a, v_two),
            ("&", v_tmp1_b, v_val_last2_b, v_one), ("*", v_idx_last2_b, v_idx_last2_b, v_two),
        ]})
        self.instrs.append({"valu": [
            ("+", v_tmp3_a, v_one, v_tmp1_a), ("+", v_tmp3_b, v_one, v_tmp1_b),
        ]})
        self.instrs.append({"valu": [
            ("+", v_idx_last2_a, v_idx_last2_a, v_tmp3_a), ("+", v_idx_last2_b, v_idx_last2_b, v_tmp3_b),
        ]})
        # Merge loop control with epilogue index computation
        self.instrs.append({"valu": [
            ("<", v_tmp1_a, v_idx_last2_a, v_n_nodes), ("<", v_tmp1_b, v_idx_last2_b, v_n_nodes),
        ], "alu": [("+", round_counter, round_counter, one_const)]})
        self.instrs.append({"valu": [
            ("*", v_idx_last2_a, v_idx_last2_a, v_tmp1_a), ("*", v_idx_last2_b, v_idx_last2_b, v_tmp1_b),
        ], "alu": [("<", loop_cond, round_counter, self.scratch["rounds"])]})
        self.instrs.append({"flow": [("cond_jump", loop_cond, outer_loop_start_2)]})

        # Store ALL indices and values back to memory (pipelined: overlap addr compute with stores)
        # Indices: compute first pair of addresses
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["inp_indices_p"], addr_consts[0]),
            ("+", addr_tmp2, self.scratch["inp_indices_p"], addr_consts[1]),
        ]})
        # Pipelined: compute next addresses while storing current
        for i in range(2, batch_size // VLEN, 2):
            self.instrs.append({"alu": [
                ("+", addr_tmp3, self.scratch["inp_indices_p"], addr_consts[i]),
                ("+", addr_tmp4, self.scratch["inp_indices_p"], addr_consts[i+1]),
            ], "store": [
                ("vstore", addr_tmp, all_idx[i-2]),
                ("vstore", addr_tmp2, all_idx[i-1]),
            ]})
            # Swap address registers
            addr_tmp, addr_tmp3 = addr_tmp3, addr_tmp
            addr_tmp2, addr_tmp4 = addr_tmp4, addr_tmp2
        # Final stores for last pair
        self.instrs.append({"store": [
            ("vstore", addr_tmp, all_idx[-2]),
            ("vstore", addr_tmp2, all_idx[-1]),
        ]})

        # Values: same pattern
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["inp_values_p"], addr_consts[0]),
            ("+", addr_tmp2, self.scratch["inp_values_p"], addr_consts[1]),
        ]})
        for i in range(2, batch_size // VLEN, 2):
            self.instrs.append({"alu": [
                ("+", addr_tmp3, self.scratch["inp_values_p"], addr_consts[i]),
                ("+", addr_tmp4, self.scratch["inp_values_p"], addr_consts[i+1]),
            ], "store": [
                ("vstore", addr_tmp, all_val[i-2]),
                ("vstore", addr_tmp2, all_val[i-1]),
            ]})
            addr_tmp, addr_tmp3 = addr_tmp3, addr_tmp
            addr_tmp2, addr_tmp4 = addr_tmp4, addr_tmp2
        self.instrs.append({"store": [
            ("vstore", addr_tmp, all_val[-2]),
            ("vstore", addr_tmp2, all_val[-1]),
        ]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
