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
        # Define constants early for use in pipelined loading
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Pipelined parameter loading: overlap addr increment with memory load (14 -> 5 cycles)
        # Cycle 1: const tmp1=0, tmp2=1
        # Cycle 2-4: load 2 values while incrementing addresses with alu
        # Cycle 5: load last value
        self.instrs.append({"load": [("const", tmp1, 0), ("const", tmp2, 1)]})
        self.instrs.append({"load": [
            ("load", self.scratch["rounds"], tmp1),
            ("load", self.scratch["n_nodes"], tmp2),
        ], "alu": [("+", tmp1, tmp1, two_const), ("+", tmp2, tmp2, two_const)]})
        self.instrs.append({"load": [
            ("load", self.scratch["batch_size"], tmp1),
            ("load", self.scratch["forest_height"], tmp2),
        ], "alu": [("+", tmp1, tmp1, two_const), ("+", tmp2, tmp2, two_const)]})
        self.instrs.append({"load": [
            ("load", self.scratch["forest_values_p"], tmp1),
            ("load", self.scratch["inp_indices_p"], tmp2),
        ], "alu": [("+", tmp1, tmp1, two_const)]})
        self.instrs.append({"load": [("load", self.scratch["inp_values_p"], tmp1)]})
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

        # Initialize broadcast vectors - pack 6 per cycle for efficiency (11 cycles -> 4)
        # Pre-allocate const addresses for hash stages
        hash_const_addrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_const_addrs.append((self.scratch_const(val1), self.scratch_const(val3)))
        mul_const_addrs = [self.scratch_const(mul_val) for _, _, mul_val in mul_consts]

        # Cycle 1: v_zero, v_one, v_two, v_n_nodes, v_forest_p, hash[0][0]
        self.instrs.append({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
            ("vbroadcast", v_hash_consts[0][0], hash_const_addrs[0][0]),
        ]})
        # Cycle 2: hash[0][1], [1][0], [1][1], [2][0], [2][1], [3][0]
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_consts[0][1], hash_const_addrs[0][1]),
            ("vbroadcast", v_hash_consts[1][0], hash_const_addrs[1][0]),
            ("vbroadcast", v_hash_consts[1][1], hash_const_addrs[1][1]),
            ("vbroadcast", v_hash_consts[2][0], hash_const_addrs[2][0]),
            ("vbroadcast", v_hash_consts[2][1], hash_const_addrs[2][1]),
            ("vbroadcast", v_hash_consts[3][0], hash_const_addrs[3][0]),
        ]})
        # Cycle 3: hash[3][1], [4][0], [4][1], [5][0], [5][1], mul_4097
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_consts[3][1], hash_const_addrs[3][1]),
            ("vbroadcast", v_hash_consts[4][0], hash_const_addrs[4][0]),
            ("vbroadcast", v_hash_consts[4][1], hash_const_addrs[4][1]),
            ("vbroadcast", v_hash_consts[5][0], hash_const_addrs[5][0]),
            ("vbroadcast", v_hash_consts[5][1], hash_const_addrs[5][1]),
            ("vbroadcast", v_mul_4097, mul_const_addrs[0]),
        ]})
        # Cycle 4: mul_33, mul_9
        self.instrs.append({"valu": [
            ("vbroadcast", v_mul_33, mul_const_addrs[1]),
            ("vbroadcast", v_mul_9, mul_const_addrs[2]),
        ]})

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
        # Final index loads overlapped with first value address computation (saves 1 cycle)
        self.instrs.append({"load": [
            ("vload", all_idx[-2], addr_tmp),
            ("vload", all_idx[-1], addr_tmp2),
        ], "alu": [
            ("+", addr_tmp3, self.scratch["inp_values_p"], addr_consts[0]),
            ("+", addr_tmp4, self.scratch["inp_values_p"], addr_consts[1]),
        ]})
        addr_tmp, addr_tmp3 = addr_tmp3, addr_tmp
        addr_tmp2, addr_tmp4 = addr_tmp4, addr_tmp2

        # Values: continue the pattern
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

        # Additional registers for 4-vector batching
        v_node_val_c = self.alloc_scratch("v_node_val_c", VLEN)
        v_node_val_c2 = self.alloc_scratch("v_node_val_c2", VLEN)
        v_node_val_d = self.alloc_scratch("v_node_val_d", VLEN)
        v_node_val_d2 = self.alloc_scratch("v_node_val_d2", VLEN)
        v_addr_c = self.alloc_scratch("v_addr_c", VLEN)
        v_addr_c2 = self.alloc_scratch("v_addr_c2", VLEN)
        v_addr_d = self.alloc_scratch("v_addr_d", VLEN)
        v_addr_d2 = self.alloc_scratch("v_addr_d2", VLEN)

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
        # Additional tmp3 registers for 2-bit selection optimization
        v_tmp3_c = self.alloc_scratch("v_tmp3_c", VLEN)
        v_tmp3_d = self.alloc_scratch("v_tmp3_d", VLEN)
        v_tmp3_e = self.alloc_scratch("v_tmp3_e", VLEN)
        v_tmp3_f = self.alloc_scratch("v_tmp3_f", VLEN)

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

        # Load tree[1] and tree[2] (parallel: 4 cycles -> 2 cycles)
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["forest_values_p"], one_const),
            ("+", addr_tmp2, self.scratch["forest_values_p"], two_const),
        ]})
        self.instrs.append({"load": [
            ("load", tree1_scalar, addr_tmp),
            ("load", tree2_scalar, addr_tmp2),
        ]})

        # Compute diff = tree2 - tree1 and c = tree1 - diff for multiply_add optimization
        # node_val = tree1 + (idx - 1) * diff = idx * diff + (tree1 - diff) = multiply_add(idx, diff, c)
        c_scalar = self.alloc_scratch("c_scalar")
        v_c = self.alloc_scratch("v_c", VLEN)
        self.instrs.append({"alu": [("-", diff_scalar, tree2_scalar, tree1_scalar)]})
        self.instrs.append({"alu": [("-", c_scalar, tree1_scalar, diff_scalar)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree1, tree1_scalar),
            ("vbroadcast", v_diff, diff_scalar),
            ("vbroadcast", v_c, c_scalar),
        ]})

        # Process all vectors for round 1
        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            # Compute node_val = idx * diff + c using multiply_add (3 cycles -> 1 cycle)
            node_val_ops = [("multiply_add", t1, v_idx, v_diff, v_c) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": node_val_ops})

            # XOR: val = val ^ node_val (node_val is in tmp1_list)
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

            # Index computation: new_idx = idx * 2 + 1 + (val & 1)
            # Use multiply_add to combine idx*2+1, then add (val&1) (6 cycles -> 5)
            mul_add_idx_ops = [("multiply_add", v_idx, v_idx, v_two, v_one) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_add_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})
            # No wrap check needed - max idx is 6 which is << n_nodes

        # ============================================
        # ROUND 2: All indices are 3, 4, 5, or 6 (use one-hot selection instead of gather)
        # ============================================
        three_const = self.scratch_const(3)
        four_const = self.scratch_const(4)
        five_const = self.scratch_const(5)
        six_const = self.scratch_const(6)

        tree3_scalar = self.alloc_scratch("tree3_scalar")
        tree4_scalar = self.alloc_scratch("tree4_scalar")
        tree5_scalar = self.alloc_scratch("tree5_scalar")
        tree6_scalar = self.alloc_scratch("tree6_scalar")

        v_tree3 = self.alloc_scratch("v_tree3", VLEN)
        v_tree4 = self.alloc_scratch("v_tree4", VLEN)
        v_tree5 = self.alloc_scratch("v_tree5", VLEN)
        v_tree6 = self.alloc_scratch("v_tree6", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN)
        v_four = self.alloc_scratch("v_four", VLEN)
        v_five = self.alloc_scratch("v_five", VLEN)
        v_six = self.alloc_scratch("v_six", VLEN)

        # Load tree[3..6] and broadcast (overlap addr compute with load: 4 cycles -> 3 cycles)
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["forest_values_p"], three_const),
            ("+", addr_tmp2, self.scratch["forest_values_p"], four_const),
        ]})
        self.instrs.append({"load": [
            ("load", tree3_scalar, addr_tmp),
            ("load", tree4_scalar, addr_tmp2),
        ], "alu": [
            ("+", addr_tmp, self.scratch["forest_values_p"], five_const),
            ("+", addr_tmp2, self.scratch["forest_values_p"], six_const),
        ]})
        # Pre-compute diff values for 2-bit selection (saves 2 cycles per batch)
        diff_low_scalar = self.alloc_scratch("diff_low_scalar")
        diff_high_scalar = self.alloc_scratch("diff_high_scalar")
        v_diff_low = self.alloc_scratch("v_diff_low", VLEN)
        v_diff_high = self.alloc_scratch("v_diff_high", VLEN)
        # Overlap tree5,6 load with diff_low computation (diff_low uses tree3,4 which are already loaded)
        self.instrs.append({"load": [
            ("load", tree5_scalar, addr_tmp),
            ("load", tree6_scalar, addr_tmp2),
        ], "alu": [
            ("-", diff_low_scalar, tree3_scalar, tree4_scalar),
        ]})
        # Compute diff_high after tree5,6 are loaded
        self.instrs.append({"alu": [
            ("-", diff_high_scalar, tree5_scalar, tree6_scalar),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree3, tree3_scalar),
            ("vbroadcast", v_tree4, tree4_scalar),
            ("vbroadcast", v_tree5, tree5_scalar),
            ("vbroadcast", v_tree6, tree6_scalar),
            ("vbroadcast", v_five, five_const),
            ("vbroadcast", v_diff_low, diff_low_scalar),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_diff_high, diff_high_scalar),
        ]})

        # Process all vectors for round 2 using 2-bit selection (8 cycles -> 6 cycles per batch)
        # node_val = tree[idx] where idx is 3, 4, 5, or 6
        # Use: cmp_lt5 = (idx < 5), cmp_odd = (idx & 1)
        # base_low = tree4 + cmp_odd * (tree3 - tree4)
        # base_high = tree6 + cmp_odd * (tree5 - tree6)
        # node_val = base_high + cmp_lt5 * (base_low - base_high)
        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]
            tmp3_list = [v_tmp3_a, v_tmp3_b, v_tmp3_c, v_tmp3_d, v_tmp3_e, v_tmp3_f][:num_vecs]

            # Step 1: cmp_lt5 = (idx < 5) -> tmp1_list
            cmp_lt5_ops = [("<", t1, v_idx, v_five) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": cmp_lt5_ops})

            # Step 2: cmp_odd = (idx & 1) -> tmp2_list
            cmp_odd_ops = [("&", t2, v_idx, v_one) for (v_idx, v_val), t2 in zip(vecs, tmp2_list)]
            self.instrs.append({"valu": cmp_odd_ops})

            # Step 3: base_low = tree4 + cmp_odd * diff_low = multiply_add(cmp_odd, diff_low, tree4) -> tmp3_list
            base_low_ops = [("multiply_add", t3, t2, v_diff_low, v_tree4) for t2, t3 in zip(tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": base_low_ops})

            # Step 4: base_high = tree6 + cmp_odd * diff_high = multiply_add(cmp_odd, diff_high, tree6) -> tmp2_list (overwrite)
            base_high_ops = [("multiply_add", t2, t2, v_diff_high, v_tree6) for t2 in tmp2_list[:num_vecs]]
            self.instrs.append({"valu": base_high_ops})

            # Step 5: diff_bases = base_low - base_high -> tmp3_list (overwrite)
            diff_bases_ops = [("-", t3, t3, t2) for t2, t3 in zip(tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": diff_bases_ops})

            # Step 6: node_val = base_high + cmp_lt5 * diff_bases = multiply_add(cmp_lt5, diff_bases, base_high) -> tmp1_list
            node_val_ops = [("multiply_add", t1, t1, t3, t2) for t1, t2, t3 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": node_val_ops})
            # Now tmp1_list[i] contains node_val for each vector

            # XOR: val = val ^ node_val
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

            # Index computation: new_idx = idx * 2 + 1 + (val & 1)
            # Use multiply_add to combine idx*2+1 (6 cycles -> 5)
            mul_add_idx_ops = [("multiply_add", v_idx, v_idx, v_two, v_one) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_add_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})
            # No wrap check needed - max idx is 14 which is << n_nodes

        # ============================================
        # ROUNDS 3 to 10: 4-vector pipelined loop
        # ============================================
        # 4-VECTOR PIPELINED INNER LOOP
        # Process 4 vectors per batch (8 batches total for 32 vectors)
        # 16 gather cycles per batch (4 vectors × 8 elements / 2 loads per cycle)
        NUM_BATCHES_4 = batch_size // (VLEN * 4)  # 8 batches

        # Hash constants for quick access
        v_c1_0 = v_hash_consts[0][0]
        v_c1_2 = v_hash_consts[2][0]
        v_c1_4 = v_hash_consts[4][0]
        h_stage1 = HASH_STAGES[1]
        v_c1_1, v_c3_1 = v_hash_consts[1]
        h_stage3 = HASH_STAGES[3]
        v_c1_3, v_c3_3 = v_hash_consts[3]
        h_stage5 = HASH_STAGES[5]
        v_c1_5, v_c3_5 = v_hash_consts[5]

        # Prologue: Gather batch 0 (vectors 0,1,2,3) - only runs on first iteration
        # Combine const load with addr compute to save 1 cycle
        self.instrs.append({"valu": [
            ("+", v_addr_a, all_idx[0], v_forest_p),
            ("+", v_addr_b, all_idx[1], v_forest_p),
            ("+", v_addr_c, all_idx[2], v_forest_p),
            ("+", v_addr_d, all_idx[3], v_forest_p),
        ], "load": [("const", round_counter, 3)]})
        # 16 gather cycles for 4 vectors
        for vec_i, (nv, addr) in enumerate([(v_node_val_a, v_addr_a), (v_node_val_b, v_addr_b),
                                             (v_node_val_c, v_addr_c), (v_node_val_d, v_addr_d)]):
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [
                    ("load_offset", nv, addr, i),
                    ("load_offset", nv, addr, i + 1),
                ]})

        # Loop start is AFTER prologue - subsequent iterations skip the prologue
        # because the epilogue pre-gathers for the next round
        outer_loop_start = len(self.instrs)

        # Steady state: batches 1 to NUM_BATCHES_4-1
        for batch in range(1, NUM_BATCHES_4):
            # Previous batch vector indices
            prev_base = (batch - 1) * 4
            v_idx_prev = [all_idx[prev_base + j] for j in range(4)]
            v_val_prev = [all_val[prev_base + j] for j in range(4)]

            # Current batch vector indices
            cur_base = batch * 4
            v_idx_cur = [all_idx[cur_base + j] for j in range(4)]

            # Double buffering: alternate between two sets of registers
            if batch % 2 == 1:
                nv_cur = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
                nv_prev = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
                addr_cur = [v_addr_a2, v_addr_b2, v_addr_c2, v_addr_d2]
            else:
                nv_cur = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
                nv_prev = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
                addr_cur = [v_addr_a, v_addr_b, v_addr_c, v_addr_d]

            # Cycle 1: XOR prev batch (4 ops) + compute addr for current (2 ops)
            self.instrs.append({"valu": [
                ("^", v_val_prev[0], v_val_prev[0], nv_prev[0]),
                ("^", v_val_prev[1], v_val_prev[1], nv_prev[1]),
                ("^", v_val_prev[2], v_val_prev[2], nv_prev[2]),
                ("^", v_val_prev[3], v_val_prev[3], nv_prev[3]),
                ("+", addr_cur[0], v_idx_cur[0], v_forest_p),
                ("+", addr_cur[1], v_idx_cur[1], v_forest_p),
            ]})
            # Cycle 2: remaining addr (2 ops) + stage 0 (4 ops = multiply_add for a,b,c,d)
            self.instrs.append({"valu": [
                ("+", addr_cur[2], v_idx_cur[2], v_forest_p),
                ("+", addr_cur[3], v_idx_cur[3], v_forest_p),
                ("multiply_add", v_val_prev[0], v_val_prev[0], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev[1], v_val_prev[1], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev[2], v_val_prev[2], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev[3], v_val_prev[3], v_mul_4097, v_c1_0),
            ]})

            # Gather current batch (16 cycles) with overlapped hash/index for prev batch
            # gi=0: gather A[0,1], stage 1a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[0], addr_cur[0], 0),
                ("load_offset", nv_cur[0], addr_cur[0], 1),
            ], "valu": [
                (h_stage1[0], v_tmp1_a, v_val_prev[0], v_c1_1),
                (h_stage1[3], v_tmp2_a, v_val_prev[0], v_c3_1),
                (h_stage1[0], v_tmp1_b, v_val_prev[1], v_c1_1),
                (h_stage1[3], v_tmp2_b, v_val_prev[1], v_c3_1),
            ]})
            # gi=1: gather A[2,3], stage 1a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[0], addr_cur[0], 2),
                ("load_offset", nv_cur[0], addr_cur[0], 3),
            ], "valu": [
                (h_stage1[0], v_tmp1_c, v_val_prev[2], v_c1_1),
                (h_stage1[3], v_tmp2_c, v_val_prev[2], v_c3_1),
                (h_stage1[0], v_tmp1_d, v_val_prev[3], v_c1_1),
                (h_stage1[3], v_tmp2_d, v_val_prev[3], v_c3_1),
            ]})
            # gi=2: gather A[4,5], stage 1b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[0], addr_cur[0], 4),
                ("load_offset", nv_cur[0], addr_cur[0], 5),
            ], "valu": [
                (h_stage1[2], v_val_prev[0], v_tmp1_a, v_tmp2_a),
                (h_stage1[2], v_val_prev[1], v_tmp1_b, v_tmp2_b),
                (h_stage1[2], v_val_prev[2], v_tmp1_c, v_tmp2_c),
                (h_stage1[2], v_val_prev[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=3: gather A[6,7], stage 2 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[0], addr_cur[0], 6),
                ("load_offset", nv_cur[0], addr_cur[0], 7),
            ], "valu": [
                ("multiply_add", v_val_prev[0], v_val_prev[0], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev[1], v_val_prev[1], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev[2], v_val_prev[2], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev[3], v_val_prev[3], v_mul_33, v_c1_2),
            ]})
            # gi=4: gather B[0,1], stage 3a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[1], addr_cur[1], 0),
                ("load_offset", nv_cur[1], addr_cur[1], 1),
            ], "valu": [
                (h_stage3[0], v_tmp1_a, v_val_prev[0], v_c1_3),
                (h_stage3[3], v_tmp2_a, v_val_prev[0], v_c3_3),
                (h_stage3[0], v_tmp1_b, v_val_prev[1], v_c1_3),
                (h_stage3[3], v_tmp2_b, v_val_prev[1], v_c3_3),
            ]})
            # gi=5: gather B[2,3], stage 3a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[1], addr_cur[1], 2),
                ("load_offset", nv_cur[1], addr_cur[1], 3),
            ], "valu": [
                (h_stage3[0], v_tmp1_c, v_val_prev[2], v_c1_3),
                (h_stage3[3], v_tmp2_c, v_val_prev[2], v_c3_3),
                (h_stage3[0], v_tmp1_d, v_val_prev[3], v_c1_3),
                (h_stage3[3], v_tmp2_d, v_val_prev[3], v_c3_3),
            ]})
            # gi=6: gather B[4,5], stage 3b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[1], addr_cur[1], 4),
                ("load_offset", nv_cur[1], addr_cur[1], 5),
            ], "valu": [
                (h_stage3[2], v_val_prev[0], v_tmp1_a, v_tmp2_a),
                (h_stage3[2], v_val_prev[1], v_tmp1_b, v_tmp2_b),
                (h_stage3[2], v_val_prev[2], v_tmp1_c, v_tmp2_c),
                (h_stage3[2], v_val_prev[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=7: gather B[6,7], stage 4 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[1], addr_cur[1], 6),
                ("load_offset", nv_cur[1], addr_cur[1], 7),
            ], "valu": [
                ("multiply_add", v_val_prev[0], v_val_prev[0], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev[1], v_val_prev[1], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev[2], v_val_prev[2], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev[3], v_val_prev[3], v_mul_9, v_c1_4),
            ]})
            # gi=8: gather C[0,1], stage 5a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[2], addr_cur[2], 0),
                ("load_offset", nv_cur[2], addr_cur[2], 1),
            ], "valu": [
                (h_stage5[0], v_tmp1_a, v_val_prev[0], v_c1_5),
                (h_stage5[3], v_tmp2_a, v_val_prev[0], v_c3_5),
                (h_stage5[0], v_tmp1_b, v_val_prev[1], v_c1_5),
                (h_stage5[3], v_tmp2_b, v_val_prev[1], v_c3_5),
            ]})
            # gi=9: gather C[2,3], stage 5a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[2], addr_cur[2], 2),
                ("load_offset", nv_cur[2], addr_cur[2], 3),
            ], "valu": [
                (h_stage5[0], v_tmp1_c, v_val_prev[2], v_c1_5),
                (h_stage5[3], v_tmp2_c, v_val_prev[2], v_c3_5),
                (h_stage5[0], v_tmp1_d, v_val_prev[3], v_c1_5),
                (h_stage5[3], v_tmp2_d, v_val_prev[3], v_c3_5),
            ]})
            # gi=10: gather C[4,5], stage 5b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[2], addr_cur[2], 4),
                ("load_offset", nv_cur[2], addr_cur[2], 5),
            ], "valu": [
                (h_stage5[2], v_val_prev[0], v_tmp1_a, v_tmp2_a),
                (h_stage5[2], v_val_prev[1], v_tmp1_b, v_tmp2_b),
                (h_stage5[2], v_val_prev[2], v_tmp1_c, v_tmp2_c),
                (h_stage5[2], v_val_prev[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=11: gather C[6,7], idx*2+1 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[2], addr_cur[2], 6),
                ("load_offset", nv_cur[2], addr_cur[2], 7),
            ], "valu": [
                ("multiply_add", v_idx_prev[0], v_idx_prev[0], v_two, v_one),
                ("multiply_add", v_idx_prev[1], v_idx_prev[1], v_two, v_one),
                ("multiply_add", v_idx_prev[2], v_idx_prev[2], v_two, v_one),
                ("multiply_add", v_idx_prev[3], v_idx_prev[3], v_two, v_one),
            ]})
            # gi=12: gather D[0,1], val&1 (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[3], addr_cur[3], 0),
                ("load_offset", nv_cur[3], addr_cur[3], 1),
            ], "valu": [
                ("&", v_tmp1_a, v_val_prev[0], v_one),
                ("&", v_tmp1_b, v_val_prev[1], v_one),
                ("&", v_tmp1_c, v_val_prev[2], v_one),
                ("&", v_tmp1_d, v_val_prev[3], v_one),
            ]})
            # gi=13: gather D[2,3], idx += mask (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[3], addr_cur[3], 2),
                ("load_offset", nv_cur[3], addr_cur[3], 3),
            ], "valu": [
                ("+", v_idx_prev[0], v_idx_prev[0], v_tmp1_a),
                ("+", v_idx_prev[1], v_idx_prev[1], v_tmp1_b),
                ("+", v_idx_prev[2], v_idx_prev[2], v_tmp1_c),
                ("+", v_idx_prev[3], v_idx_prev[3], v_tmp1_d),
            ]})
            # gi=14: gather D[4,5], idx < n_nodes (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[3], addr_cur[3], 4),
                ("load_offset", nv_cur[3], addr_cur[3], 5),
            ], "valu": [
                ("<", v_tmp1_a, v_idx_prev[0], v_n_nodes),
                ("<", v_tmp1_b, v_idx_prev[1], v_n_nodes),
                ("<", v_tmp1_c, v_idx_prev[2], v_n_nodes),
                ("<", v_tmp1_d, v_idx_prev[3], v_n_nodes),
            ]})
            # gi=15: gather D[6,7], idx *= cmp (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur[3], addr_cur[3], 6),
                ("load_offset", nv_cur[3], addr_cur[3], 7),
            ], "valu": [
                ("*", v_idx_prev[0], v_idx_prev[0], v_tmp1_a),
                ("*", v_idx_prev[1], v_idx_prev[1], v_tmp1_b),
                ("*", v_idx_prev[2], v_idx_prev[2], v_tmp1_c),
                ("*", v_idx_prev[3], v_idx_prev[3], v_tmp1_d),
            ]})

        # Epilogue: Process last batch (NUM_BATCHES_4-1) with overlapped next-round gather
        # This overlaps the epilogue hash/index computation with gathering for the next round
        last_base = (NUM_BATCHES_4 - 1) * 4
        v_idx_last = [all_idx[last_base + j] for j in range(4)]
        v_val_last = [all_val[last_base + j] for j in range(4)]
        if (NUM_BATCHES_4 - 1) % 2 == 1:
            nv_last = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
        else:
            nv_last = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]

        # For next round's prologue, we'll gather into the OTHER set of registers
        if (NUM_BATCHES_4 - 1) % 2 == 1:
            nv_next = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
            addr_next = [v_addr_a, v_addr_b, v_addr_c, v_addr_d]
        else:
            nv_next = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
            addr_next = [v_addr_a2, v_addr_b2, v_addr_c2, v_addr_d2]

        # XOR + compute addresses for next round's batch 0
        self.instrs.append({"valu": [
            ("^", v_val_last[0], v_val_last[0], nv_last[0]),
            ("^", v_val_last[1], v_val_last[1], nv_last[1]),
            ("^", v_val_last[2], v_val_last[2], nv_last[2]),
            ("^", v_val_last[3], v_val_last[3], nv_last[3]),
            ("+", addr_next[0], all_idx[0], v_forest_p),
            ("+", addr_next[1], all_idx[1], v_forest_p),
        ]})
        # Hash stage 0 + remaining addresses
        self.instrs.append({"valu": [
            ("multiply_add", v_val_last[0], v_val_last[0], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last[1], v_val_last[1], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last[2], v_val_last[2], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last[3], v_val_last[3], v_mul_4097, v_c1_0),
            ("+", addr_next[2], all_idx[2], v_forest_p),
            ("+", addr_next[3], all_idx[3], v_forest_p),
        ]})
        # Now overlap hash stages 1-5 and index with 16 gather cycles for next round
        # gi=0: gather A[0,1], stage 1a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next[0], addr_next[0], 0),
            ("load_offset", nv_next[0], addr_next[0], 1),
        ], "valu": [
            (h_stage1[0], v_tmp1_a, v_val_last[0], v_c1_1),
            (h_stage1[3], v_tmp2_a, v_val_last[0], v_c3_1),
            (h_stage1[0], v_tmp1_b, v_val_last[1], v_c1_1),
            (h_stage1[3], v_tmp2_b, v_val_last[1], v_c3_1),
        ]})
        # gi=1: gather A[2,3], stage 1a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next[0], addr_next[0], 2),
            ("load_offset", nv_next[0], addr_next[0], 3),
        ], "valu": [
            (h_stage1[0], v_tmp1_c, v_val_last[2], v_c1_1),
            (h_stage1[3], v_tmp2_c, v_val_last[2], v_c3_1),
            (h_stage1[0], v_tmp1_d, v_val_last[3], v_c1_1),
            (h_stage1[3], v_tmp2_d, v_val_last[3], v_c3_1),
        ]})
        # gi=2: gather A[4,5], stage 1b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next[0], addr_next[0], 4),
            ("load_offset", nv_next[0], addr_next[0], 5),
        ], "valu": [
            (h_stage1[2], v_val_last[0], v_tmp1_a, v_tmp2_a),
            (h_stage1[2], v_val_last[1], v_tmp1_b, v_tmp2_b),
            (h_stage1[2], v_val_last[2], v_tmp1_c, v_tmp2_c),
            (h_stage1[2], v_val_last[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=3: gather A[6,7], stage 2
        self.instrs.append({"load": [
            ("load_offset", nv_next[0], addr_next[0], 6),
            ("load_offset", nv_next[0], addr_next[0], 7),
        ], "valu": [
            ("multiply_add", v_val_last[0], v_val_last[0], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last[1], v_val_last[1], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last[2], v_val_last[2], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last[3], v_val_last[3], v_mul_33, v_c1_2),
        ]})
        # gi=4: gather B[0,1], stage 3a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next[1], addr_next[1], 0),
            ("load_offset", nv_next[1], addr_next[1], 1),
        ], "valu": [
            (h_stage3[0], v_tmp1_a, v_val_last[0], v_c1_3),
            (h_stage3[3], v_tmp2_a, v_val_last[0], v_c3_3),
            (h_stage3[0], v_tmp1_b, v_val_last[1], v_c1_3),
            (h_stage3[3], v_tmp2_b, v_val_last[1], v_c3_3),
        ]})
        # gi=5: gather B[2,3], stage 3a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next[1], addr_next[1], 2),
            ("load_offset", nv_next[1], addr_next[1], 3),
        ], "valu": [
            (h_stage3[0], v_tmp1_c, v_val_last[2], v_c1_3),
            (h_stage3[3], v_tmp2_c, v_val_last[2], v_c3_3),
            (h_stage3[0], v_tmp1_d, v_val_last[3], v_c1_3),
            (h_stage3[3], v_tmp2_d, v_val_last[3], v_c3_3),
        ]})
        # gi=6: gather B[4,5], stage 3b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next[1], addr_next[1], 4),
            ("load_offset", nv_next[1], addr_next[1], 5),
        ], "valu": [
            (h_stage3[2], v_val_last[0], v_tmp1_a, v_tmp2_a),
            (h_stage3[2], v_val_last[1], v_tmp1_b, v_tmp2_b),
            (h_stage3[2], v_val_last[2], v_tmp1_c, v_tmp2_c),
            (h_stage3[2], v_val_last[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=7: gather B[6,7], stage 4
        self.instrs.append({"load": [
            ("load_offset", nv_next[1], addr_next[1], 6),
            ("load_offset", nv_next[1], addr_next[1], 7),
        ], "valu": [
            ("multiply_add", v_val_last[0], v_val_last[0], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last[1], v_val_last[1], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last[2], v_val_last[2], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last[3], v_val_last[3], v_mul_9, v_c1_4),
        ]})
        # gi=8: gather C[0,1], stage 5a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next[2], addr_next[2], 0),
            ("load_offset", nv_next[2], addr_next[2], 1),
        ], "valu": [
            (h_stage5[0], v_tmp1_a, v_val_last[0], v_c1_5),
            (h_stage5[3], v_tmp2_a, v_val_last[0], v_c3_5),
            (h_stage5[0], v_tmp1_b, v_val_last[1], v_c1_5),
            (h_stage5[3], v_tmp2_b, v_val_last[1], v_c3_5),
        ]})
        # gi=9: gather C[2,3], stage 5a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next[2], addr_next[2], 2),
            ("load_offset", nv_next[2], addr_next[2], 3),
        ], "valu": [
            (h_stage5[0], v_tmp1_c, v_val_last[2], v_c1_5),
            (h_stage5[3], v_tmp2_c, v_val_last[2], v_c3_5),
            (h_stage5[0], v_tmp1_d, v_val_last[3], v_c1_5),
            (h_stage5[3], v_tmp2_d, v_val_last[3], v_c3_5),
        ]})
        # gi=10: gather C[4,5], stage 5b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next[2], addr_next[2], 4),
            ("load_offset", nv_next[2], addr_next[2], 5),
        ], "valu": [
            (h_stage5[2], v_val_last[0], v_tmp1_a, v_tmp2_a),
            (h_stage5[2], v_val_last[1], v_tmp1_b, v_tmp2_b),
            (h_stage5[2], v_val_last[2], v_tmp1_c, v_tmp2_c),
            (h_stage5[2], v_val_last[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=11: gather C[6,7], idx*2+1
        self.instrs.append({"load": [
            ("load_offset", nv_next[2], addr_next[2], 6),
            ("load_offset", nv_next[2], addr_next[2], 7),
        ], "valu": [
            ("multiply_add", v_idx_last[0], v_idx_last[0], v_two, v_one),
            ("multiply_add", v_idx_last[1], v_idx_last[1], v_two, v_one),
            ("multiply_add", v_idx_last[2], v_idx_last[2], v_two, v_one),
            ("multiply_add", v_idx_last[3], v_idx_last[3], v_two, v_one),
        ]})
        # gi=12: gather D[0,1], val&1
        self.instrs.append({"load": [
            ("load_offset", nv_next[3], addr_next[3], 0),
            ("load_offset", nv_next[3], addr_next[3], 1),
        ], "valu": [
            ("&", v_tmp1_a, v_val_last[0], v_one),
            ("&", v_tmp1_b, v_val_last[1], v_one),
            ("&", v_tmp1_c, v_val_last[2], v_one),
            ("&", v_tmp1_d, v_val_last[3], v_one),
        ]})
        # gi=13: gather D[2,3], idx += mask
        self.instrs.append({"load": [
            ("load_offset", nv_next[3], addr_next[3], 2),
            ("load_offset", nv_next[3], addr_next[3], 3),
        ], "valu": [
            ("+", v_idx_last[0], v_idx_last[0], v_tmp1_a),
            ("+", v_idx_last[1], v_idx_last[1], v_tmp1_b),
            ("+", v_idx_last[2], v_idx_last[2], v_tmp1_c),
            ("+", v_idx_last[3], v_idx_last[3], v_tmp1_d),
        ]})
        # gi=14: gather D[4,5], idx < n_nodes
        self.instrs.append({"load": [
            ("load_offset", nv_next[3], addr_next[3], 4),
            ("load_offset", nv_next[3], addr_next[3], 5),
        ], "valu": [
            ("<", v_tmp1_a, v_idx_last[0], v_n_nodes),
            ("<", v_tmp1_b, v_idx_last[1], v_n_nodes),
            ("<", v_tmp1_c, v_idx_last[2], v_n_nodes),
            ("<", v_tmp1_d, v_idx_last[3], v_n_nodes),
        ], "alu": [("+", round_counter, round_counter, one_const)]})
        # gi=15: gather D[6,7], idx *= cmp + loop control
        self.instrs.append({"load": [
            ("load_offset", nv_next[3], addr_next[3], 6),
            ("load_offset", nv_next[3], addr_next[3], 7),
        ], "valu": [
            ("*", v_idx_last[0], v_idx_last[0], v_tmp1_a),
            ("*", v_idx_last[1], v_idx_last[1], v_tmp1_b),
            ("*", v_idx_last[2], v_idx_last[2], v_tmp1_c),
            ("*", v_idx_last[3], v_idx_last[3], v_tmp1_d),
        ], "alu": [("<", loop_cond, round_counter, eleven_const)]})
        # Overlap tree[0] load with loop branch (speculative: saves 1 cycle on loop exit)
        self.instrs.append({"flow": [("cond_jump", loop_cond, outer_loop_start)],
                           "load": [("load", tree0_scalar, self.scratch["forest_values_p"])]})

        # ============================================
        # ROUND 11: All items have idx=0 (after round 10 wrapped all to 0)
        # Use broadcast instead of gather (tree[0] already loaded above)
        # ============================================
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
        # Load tree[1] and tree[2] (parallel: 4 cycles -> 2 cycles)
        self.instrs.append({"alu": [
            ("+", addr_tmp, self.scratch["forest_values_p"], one_const),
            ("+", addr_tmp2, self.scratch["forest_values_p"], two_const),
        ]})
        self.instrs.append({"load": [
            ("load", tree1_scalar, addr_tmp),
            ("load", tree2_scalar, addr_tmp2),
        ]})
        # Compute diff and c for multiply_add optimization
        self.instrs.append({"alu": [("-", diff_scalar, tree2_scalar, tree1_scalar)]})
        self.instrs.append({"alu": [("-", c_scalar, tree1_scalar, diff_scalar)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree1, tree1_scalar),
            ("vbroadcast", v_diff, diff_scalar),
            ("vbroadcast", v_c, c_scalar),
        ]})

        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]

            # Compute node_val = idx * diff + c using multiply_add (3 cycles -> 1 cycle)
            node_val_ops = [("multiply_add", t1, v_idx, v_diff, v_c) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": node_val_ops})

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

            # Index computation: idx*2+1 + (val&1) using multiply_add
            mul_add_idx_ops = [("multiply_add", v_idx, v_idx, v_two, v_one) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_add_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})
            # No wrap check needed - max idx is 6 which is << n_nodes

        # ============================================
        # ROUND 13: All indices are 3, 4, 5, or 6 (use one-hot selection instead of gather)
        # Same as round 2
        # ============================================
        # Tree values 3-6 are already loaded, just reuse v_tree3, v_tree4, v_tree5, v_tree6

        # Process all vectors for round 13 using 2-bit selection (same as round 2)
        for start_vec, num_vecs in vec_batches:
            vecs = [(all_idx[i], all_val[i]) for i in range(start_vec, start_vec + num_vecs)]
            tmp1_list = [v_tmp1_a, v_tmp1_b, v_tmp1_c, v_tmp1_d, v_tmp1_e, v_tmp1_f][:num_vecs]
            tmp2_list = [v_tmp2_a, v_tmp2_b, v_tmp2_c, v_tmp2_d, v_tmp2_e, v_tmp2_f][:num_vecs]
            tmp3_list = [v_tmp3_a, v_tmp3_b, v_tmp3_c, v_tmp3_d, v_tmp3_e, v_tmp3_f][:num_vecs]

            # 2-bit selection: cmp_lt5, cmp_odd -> base_low, base_high -> diff_bases -> node_val
            cmp_lt5_ops = [("<", t1, v_idx, v_five) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": cmp_lt5_ops})

            cmp_odd_ops = [("&", t2, v_idx, v_one) for (v_idx, v_val), t2 in zip(vecs, tmp2_list)]
            self.instrs.append({"valu": cmp_odd_ops})

            base_low_ops = [("multiply_add", t3, t2, v_diff_low, v_tree4) for t2, t3 in zip(tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": base_low_ops})

            base_high_ops = [("multiply_add", t2, t2, v_diff_high, v_tree6) for t2 in tmp2_list[:num_vecs]]
            self.instrs.append({"valu": base_high_ops})

            diff_bases_ops = [("-", t3, t3, t2) for t2, t3 in zip(tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": diff_bases_ops})

            node_val_ops = [("multiply_add", t1, t1, t3, t2) for t1, t2, t3 in zip(tmp1_list[:num_vecs], tmp2_list[:num_vecs], tmp3_list[:num_vecs])]
            self.instrs.append({"valu": node_val_ops})

            # XOR: val = val ^ node_val
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

            # Index computation: idx*2+1 + (val&1) using multiply_add
            mul_add_idx_ops = [("multiply_add", v_idx, v_idx, v_two, v_one) for (v_idx, v_val) in vecs]
            self.instrs.append({"valu": mul_add_idx_ops})

            and_ops = [("&", t1, v_val, v_one) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": and_ops})

            add_idx_ops = [("+", v_idx, v_idx, t1) for (v_idx, v_val), t1 in zip(vecs, tmp1_list)]
            self.instrs.append({"valu": add_idx_ops})
            # No wrap check needed - max idx is 14 which is << n_nodes

        # ============================================
        # ROUNDS 14-15: 4-vector pipelined loop
        # ============================================
        # Prologue: Gather batch 0 (vectors 0,1,2,3) - only runs on first iteration
        # Combine const load with addr compute to save 1 cycle
        self.instrs.append({"valu": [
            ("+", v_addr_a, all_idx[0], v_forest_p),
            ("+", v_addr_b, all_idx[1], v_forest_p),
            ("+", v_addr_c, all_idx[2], v_forest_p),
            ("+", v_addr_d, all_idx[3], v_forest_p),
        ], "load": [("const", round_counter, 14)]})
        # 16 gather cycles for 4 vectors
        for vec_i, (nv, addr) in enumerate([(v_node_val_a, v_addr_a), (v_node_val_b, v_addr_b),
                                             (v_node_val_c, v_addr_c), (v_node_val_d, v_addr_d)]):
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [
                    ("load_offset", nv, addr, i),
                    ("load_offset", nv, addr, i + 1),
                ]})

        # Loop start is AFTER prologue
        outer_loop_start_2 = len(self.instrs)

        # Steady state: batches 1 to NUM_BATCHES_4-1
        for batch in range(1, NUM_BATCHES_4):
            prev_base = (batch - 1) * 4
            v_idx_prev2 = [all_idx[prev_base + j] for j in range(4)]
            v_val_prev2 = [all_val[prev_base + j] for j in range(4)]

            cur_base = batch * 4
            v_idx_cur2 = [all_idx[cur_base + j] for j in range(4)]

            if batch % 2 == 1:
                nv_cur2 = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
                nv_prev2 = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
                addr_cur2 = [v_addr_a2, v_addr_b2, v_addr_c2, v_addr_d2]
            else:
                nv_cur2 = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
                nv_prev2 = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
                addr_cur2 = [v_addr_a, v_addr_b, v_addr_c, v_addr_d]

            # Cycle 1: XOR prev batch (4 ops) + compute addr for current (2 ops)
            self.instrs.append({"valu": [
                ("^", v_val_prev2[0], v_val_prev2[0], nv_prev2[0]),
                ("^", v_val_prev2[1], v_val_prev2[1], nv_prev2[1]),
                ("^", v_val_prev2[2], v_val_prev2[2], nv_prev2[2]),
                ("^", v_val_prev2[3], v_val_prev2[3], nv_prev2[3]),
                ("+", addr_cur2[0], v_idx_cur2[0], v_forest_p),
                ("+", addr_cur2[1], v_idx_cur2[1], v_forest_p),
            ]})
            # Cycle 2: remaining addr (2 ops) + stage 0 (4 ops)
            self.instrs.append({"valu": [
                ("+", addr_cur2[2], v_idx_cur2[2], v_forest_p),
                ("+", addr_cur2[3], v_idx_cur2[3], v_forest_p),
                ("multiply_add", v_val_prev2[0], v_val_prev2[0], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev2[1], v_val_prev2[1], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev2[2], v_val_prev2[2], v_mul_4097, v_c1_0),
                ("multiply_add", v_val_prev2[3], v_val_prev2[3], v_mul_4097, v_c1_0),
            ]})

            # Gather current batch (16 cycles) with overlapped hash/index
            # gi=0: gather A[0,1], stage 1a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[0], addr_cur2[0], 0),
                ("load_offset", nv_cur2[0], addr_cur2[0], 1),
            ], "valu": [
                (h_stage1[0], v_tmp1_a, v_val_prev2[0], v_c1_1),
                (h_stage1[3], v_tmp2_a, v_val_prev2[0], v_c3_1),
                (h_stage1[0], v_tmp1_b, v_val_prev2[1], v_c1_1),
                (h_stage1[3], v_tmp2_b, v_val_prev2[1], v_c3_1),
            ]})
            # gi=1: gather A[2,3], stage 1a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[0], addr_cur2[0], 2),
                ("load_offset", nv_cur2[0], addr_cur2[0], 3),
            ], "valu": [
                (h_stage1[0], v_tmp1_c, v_val_prev2[2], v_c1_1),
                (h_stage1[3], v_tmp2_c, v_val_prev2[2], v_c3_1),
                (h_stage1[0], v_tmp1_d, v_val_prev2[3], v_c1_1),
                (h_stage1[3], v_tmp2_d, v_val_prev2[3], v_c3_1),
            ]})
            # gi=2: gather A[4,5], stage 1b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[0], addr_cur2[0], 4),
                ("load_offset", nv_cur2[0], addr_cur2[0], 5),
            ], "valu": [
                (h_stage1[2], v_val_prev2[0], v_tmp1_a, v_tmp2_a),
                (h_stage1[2], v_val_prev2[1], v_tmp1_b, v_tmp2_b),
                (h_stage1[2], v_val_prev2[2], v_tmp1_c, v_tmp2_c),
                (h_stage1[2], v_val_prev2[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=3: gather A[6,7], stage 2 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[0], addr_cur2[0], 6),
                ("load_offset", nv_cur2[0], addr_cur2[0], 7),
            ], "valu": [
                ("multiply_add", v_val_prev2[0], v_val_prev2[0], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev2[1], v_val_prev2[1], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev2[2], v_val_prev2[2], v_mul_33, v_c1_2),
                ("multiply_add", v_val_prev2[3], v_val_prev2[3], v_mul_33, v_c1_2),
            ]})
            # gi=4: gather B[0,1], stage 3a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[1], addr_cur2[1], 0),
                ("load_offset", nv_cur2[1], addr_cur2[1], 1),
            ], "valu": [
                (h_stage3[0], v_tmp1_a, v_val_prev2[0], v_c1_3),
                (h_stage3[3], v_tmp2_a, v_val_prev2[0], v_c3_3),
                (h_stage3[0], v_tmp1_b, v_val_prev2[1], v_c1_3),
                (h_stage3[3], v_tmp2_b, v_val_prev2[1], v_c3_3),
            ]})
            # gi=5: gather B[2,3], stage 3a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[1], addr_cur2[1], 2),
                ("load_offset", nv_cur2[1], addr_cur2[1], 3),
            ], "valu": [
                (h_stage3[0], v_tmp1_c, v_val_prev2[2], v_c1_3),
                (h_stage3[3], v_tmp2_c, v_val_prev2[2], v_c3_3),
                (h_stage3[0], v_tmp1_d, v_val_prev2[3], v_c1_3),
                (h_stage3[3], v_tmp2_d, v_val_prev2[3], v_c3_3),
            ]})
            # gi=6: gather B[4,5], stage 3b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[1], addr_cur2[1], 4),
                ("load_offset", nv_cur2[1], addr_cur2[1], 5),
            ], "valu": [
                (h_stage3[2], v_val_prev2[0], v_tmp1_a, v_tmp2_a),
                (h_stage3[2], v_val_prev2[1], v_tmp1_b, v_tmp2_b),
                (h_stage3[2], v_val_prev2[2], v_tmp1_c, v_tmp2_c),
                (h_stage3[2], v_val_prev2[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=7: gather B[6,7], stage 4 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[1], addr_cur2[1], 6),
                ("load_offset", nv_cur2[1], addr_cur2[1], 7),
            ], "valu": [
                ("multiply_add", v_val_prev2[0], v_val_prev2[0], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev2[1], v_val_prev2[1], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev2[2], v_val_prev2[2], v_mul_9, v_c1_4),
                ("multiply_add", v_val_prev2[3], v_val_prev2[3], v_mul_9, v_c1_4),
            ]})
            # gi=8: gather C[0,1], stage 5a (tmp1,tmp2 for a,b)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[2], addr_cur2[2], 0),
                ("load_offset", nv_cur2[2], addr_cur2[2], 1),
            ], "valu": [
                (h_stage5[0], v_tmp1_a, v_val_prev2[0], v_c1_5),
                (h_stage5[3], v_tmp2_a, v_val_prev2[0], v_c3_5),
                (h_stage5[0], v_tmp1_b, v_val_prev2[1], v_c1_5),
                (h_stage5[3], v_tmp2_b, v_val_prev2[1], v_c3_5),
            ]})
            # gi=9: gather C[2,3], stage 5a (tmp1,tmp2 for c,d)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[2], addr_cur2[2], 2),
                ("load_offset", nv_cur2[2], addr_cur2[2], 3),
            ], "valu": [
                (h_stage5[0], v_tmp1_c, v_val_prev2[2], v_c1_5),
                (h_stage5[3], v_tmp2_c, v_val_prev2[2], v_c3_5),
                (h_stage5[0], v_tmp1_d, v_val_prev2[3], v_c1_5),
                (h_stage5[3], v_tmp2_d, v_val_prev2[3], v_c3_5),
            ]})
            # gi=10: gather C[4,5], stage 5b (combine all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[2], addr_cur2[2], 4),
                ("load_offset", nv_cur2[2], addr_cur2[2], 5),
            ], "valu": [
                (h_stage5[2], v_val_prev2[0], v_tmp1_a, v_tmp2_a),
                (h_stage5[2], v_val_prev2[1], v_tmp1_b, v_tmp2_b),
                (h_stage5[2], v_val_prev2[2], v_tmp1_c, v_tmp2_c),
                (h_stage5[2], v_val_prev2[3], v_tmp1_d, v_tmp2_d),
            ]})
            # gi=11: gather C[6,7], idx*2+1 (multiply_add for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[2], addr_cur2[2], 6),
                ("load_offset", nv_cur2[2], addr_cur2[2], 7),
            ], "valu": [
                ("multiply_add", v_idx_prev2[0], v_idx_prev2[0], v_two, v_one),
                ("multiply_add", v_idx_prev2[1], v_idx_prev2[1], v_two, v_one),
                ("multiply_add", v_idx_prev2[2], v_idx_prev2[2], v_two, v_one),
                ("multiply_add", v_idx_prev2[3], v_idx_prev2[3], v_two, v_one),
            ]})
            # gi=12: gather D[0,1], val&1 (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[3], addr_cur2[3], 0),
                ("load_offset", nv_cur2[3], addr_cur2[3], 1),
            ], "valu": [
                ("&", v_tmp1_a, v_val_prev2[0], v_one),
                ("&", v_tmp1_b, v_val_prev2[1], v_one),
                ("&", v_tmp1_c, v_val_prev2[2], v_one),
                ("&", v_tmp1_d, v_val_prev2[3], v_one),
            ]})
            # gi=13: gather D[2,3], idx += mask (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[3], addr_cur2[3], 2),
                ("load_offset", nv_cur2[3], addr_cur2[3], 3),
            ], "valu": [
                ("+", v_idx_prev2[0], v_idx_prev2[0], v_tmp1_a),
                ("+", v_idx_prev2[1], v_idx_prev2[1], v_tmp1_b),
                ("+", v_idx_prev2[2], v_idx_prev2[2], v_tmp1_c),
                ("+", v_idx_prev2[3], v_idx_prev2[3], v_tmp1_d),
            ]})
            # gi=14: gather D[4,5], idx < n_nodes (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[3], addr_cur2[3], 4),
                ("load_offset", nv_cur2[3], addr_cur2[3], 5),
            ], "valu": [
                ("<", v_tmp1_a, v_idx_prev2[0], v_n_nodes),
                ("<", v_tmp1_b, v_idx_prev2[1], v_n_nodes),
                ("<", v_tmp1_c, v_idx_prev2[2], v_n_nodes),
                ("<", v_tmp1_d, v_idx_prev2[3], v_n_nodes),
            ]})
            # gi=15: gather D[6,7], idx *= cmp (for all 4)
            self.instrs.append({"load": [
                ("load_offset", nv_cur2[3], addr_cur2[3], 6),
                ("load_offset", nv_cur2[3], addr_cur2[3], 7),
            ], "valu": [
                ("*", v_idx_prev2[0], v_idx_prev2[0], v_tmp1_a),
                ("*", v_idx_prev2[1], v_idx_prev2[1], v_tmp1_b),
                ("*", v_idx_prev2[2], v_idx_prev2[2], v_tmp1_c),
                ("*", v_idx_prev2[3], v_idx_prev2[3], v_tmp1_d),
            ]})

        # Epilogue for third loop - with overlapped next-round gather
        last_base2 = (NUM_BATCHES_4 - 1) * 4
        v_idx_last2 = [all_idx[last_base2 + j] for j in range(4)]
        v_val_last2 = [all_val[last_base2 + j] for j in range(4)]
        if (NUM_BATCHES_4 - 1) % 2 == 1:
            nv_last2 = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
        else:
            nv_last2 = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]

        # For next round's prologue
        if (NUM_BATCHES_4 - 1) % 2 == 1:
            nv_next2 = [v_node_val_a, v_node_val_b, v_node_val_c, v_node_val_d]
            addr_next2 = [v_addr_a, v_addr_b, v_addr_c, v_addr_d]
        else:
            nv_next2 = [v_node_val_a2, v_node_val_b2, v_node_val_c2, v_node_val_d2]
            addr_next2 = [v_addr_a2, v_addr_b2, v_addr_c2, v_addr_d2]

        # XOR + compute addresses for next round's batch 0
        self.instrs.append({"valu": [
            ("^", v_val_last2[0], v_val_last2[0], nv_last2[0]),
            ("^", v_val_last2[1], v_val_last2[1], nv_last2[1]),
            ("^", v_val_last2[2], v_val_last2[2], nv_last2[2]),
            ("^", v_val_last2[3], v_val_last2[3], nv_last2[3]),
            ("+", addr_next2[0], all_idx[0], v_forest_p),
            ("+", addr_next2[1], all_idx[1], v_forest_p),
        ]})
        # Hash stage 0 + remaining addresses
        self.instrs.append({"valu": [
            ("multiply_add", v_val_last2[0], v_val_last2[0], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last2[1], v_val_last2[1], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last2[2], v_val_last2[2], v_mul_4097, v_c1_0),
            ("multiply_add", v_val_last2[3], v_val_last2[3], v_mul_4097, v_c1_0),
            ("+", addr_next2[2], all_idx[2], v_forest_p),
            ("+", addr_next2[3], all_idx[3], v_forest_p),
        ]})
        # Overlap hash stages 1-5 and index with 16 gather cycles for next round
        # gi=0: gather A[0,1], stage 1a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[0], addr_next2[0], 0),
            ("load_offset", nv_next2[0], addr_next2[0], 1),
        ], "valu": [
            (h_stage1[0], v_tmp1_a, v_val_last2[0], v_c1_1),
            (h_stage1[3], v_tmp2_a, v_val_last2[0], v_c3_1),
            (h_stage1[0], v_tmp1_b, v_val_last2[1], v_c1_1),
            (h_stage1[3], v_tmp2_b, v_val_last2[1], v_c3_1),
        ]})
        # gi=1: gather A[2,3], stage 1a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[0], addr_next2[0], 2),
            ("load_offset", nv_next2[0], addr_next2[0], 3),
        ], "valu": [
            (h_stage1[0], v_tmp1_c, v_val_last2[2], v_c1_1),
            (h_stage1[3], v_tmp2_c, v_val_last2[2], v_c3_1),
            (h_stage1[0], v_tmp1_d, v_val_last2[3], v_c1_1),
            (h_stage1[3], v_tmp2_d, v_val_last2[3], v_c3_1),
        ]})
        # gi=2: gather A[4,5], stage 1b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[0], addr_next2[0], 4),
            ("load_offset", nv_next2[0], addr_next2[0], 5),
        ], "valu": [
            (h_stage1[2], v_val_last2[0], v_tmp1_a, v_tmp2_a),
            (h_stage1[2], v_val_last2[1], v_tmp1_b, v_tmp2_b),
            (h_stage1[2], v_val_last2[2], v_tmp1_c, v_tmp2_c),
            (h_stage1[2], v_val_last2[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=3: gather A[6,7], stage 2
        self.instrs.append({"load": [
            ("load_offset", nv_next2[0], addr_next2[0], 6),
            ("load_offset", nv_next2[0], addr_next2[0], 7),
        ], "valu": [
            ("multiply_add", v_val_last2[0], v_val_last2[0], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last2[1], v_val_last2[1], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last2[2], v_val_last2[2], v_mul_33, v_c1_2),
            ("multiply_add", v_val_last2[3], v_val_last2[3], v_mul_33, v_c1_2),
        ]})
        # gi=4: gather B[0,1], stage 3a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[1], addr_next2[1], 0),
            ("load_offset", nv_next2[1], addr_next2[1], 1),
        ], "valu": [
            (h_stage3[0], v_tmp1_a, v_val_last2[0], v_c1_3),
            (h_stage3[3], v_tmp2_a, v_val_last2[0], v_c3_3),
            (h_stage3[0], v_tmp1_b, v_val_last2[1], v_c1_3),
            (h_stage3[3], v_tmp2_b, v_val_last2[1], v_c3_3),
        ]})
        # gi=5: gather B[2,3], stage 3a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[1], addr_next2[1], 2),
            ("load_offset", nv_next2[1], addr_next2[1], 3),
        ], "valu": [
            (h_stage3[0], v_tmp1_c, v_val_last2[2], v_c1_3),
            (h_stage3[3], v_tmp2_c, v_val_last2[2], v_c3_3),
            (h_stage3[0], v_tmp1_d, v_val_last2[3], v_c1_3),
            (h_stage3[3], v_tmp2_d, v_val_last2[3], v_c3_3),
        ]})
        # gi=6: gather B[4,5], stage 3b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[1], addr_next2[1], 4),
            ("load_offset", nv_next2[1], addr_next2[1], 5),
        ], "valu": [
            (h_stage3[2], v_val_last2[0], v_tmp1_a, v_tmp2_a),
            (h_stage3[2], v_val_last2[1], v_tmp1_b, v_tmp2_b),
            (h_stage3[2], v_val_last2[2], v_tmp1_c, v_tmp2_c),
            (h_stage3[2], v_val_last2[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=7: gather B[6,7], stage 4
        self.instrs.append({"load": [
            ("load_offset", nv_next2[1], addr_next2[1], 6),
            ("load_offset", nv_next2[1], addr_next2[1], 7),
        ], "valu": [
            ("multiply_add", v_val_last2[0], v_val_last2[0], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last2[1], v_val_last2[1], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last2[2], v_val_last2[2], v_mul_9, v_c1_4),
            ("multiply_add", v_val_last2[3], v_val_last2[3], v_mul_9, v_c1_4),
        ]})
        # gi=8: gather C[0,1], stage 5a (part 1)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[2], addr_next2[2], 0),
            ("load_offset", nv_next2[2], addr_next2[2], 1),
        ], "valu": [
            (h_stage5[0], v_tmp1_a, v_val_last2[0], v_c1_5),
            (h_stage5[3], v_tmp2_a, v_val_last2[0], v_c3_5),
            (h_stage5[0], v_tmp1_b, v_val_last2[1], v_c1_5),
            (h_stage5[3], v_tmp2_b, v_val_last2[1], v_c3_5),
        ]})
        # gi=9: gather C[2,3], stage 5a (part 2)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[2], addr_next2[2], 2),
            ("load_offset", nv_next2[2], addr_next2[2], 3),
        ], "valu": [
            (h_stage5[0], v_tmp1_c, v_val_last2[2], v_c1_5),
            (h_stage5[3], v_tmp2_c, v_val_last2[2], v_c3_5),
            (h_stage5[0], v_tmp1_d, v_val_last2[3], v_c1_5),
            (h_stage5[3], v_tmp2_d, v_val_last2[3], v_c3_5),
        ]})
        # gi=10: gather C[4,5], stage 5b (combine)
        self.instrs.append({"load": [
            ("load_offset", nv_next2[2], addr_next2[2], 4),
            ("load_offset", nv_next2[2], addr_next2[2], 5),
        ], "valu": [
            (h_stage5[2], v_val_last2[0], v_tmp1_a, v_tmp2_a),
            (h_stage5[2], v_val_last2[1], v_tmp1_b, v_tmp2_b),
            (h_stage5[2], v_val_last2[2], v_tmp1_c, v_tmp2_c),
            (h_stage5[2], v_val_last2[3], v_tmp1_d, v_tmp2_d),
        ]})
        # gi=11: gather C[6,7], idx*2+1
        self.instrs.append({"load": [
            ("load_offset", nv_next2[2], addr_next2[2], 6),
            ("load_offset", nv_next2[2], addr_next2[2], 7),
        ], "valu": [
            ("multiply_add", v_idx_last2[0], v_idx_last2[0], v_two, v_one),
            ("multiply_add", v_idx_last2[1], v_idx_last2[1], v_two, v_one),
            ("multiply_add", v_idx_last2[2], v_idx_last2[2], v_two, v_one),
            ("multiply_add", v_idx_last2[3], v_idx_last2[3], v_two, v_one),
        ]})
        # gi=12: gather D[0,1], val&1
        self.instrs.append({"load": [
            ("load_offset", nv_next2[3], addr_next2[3], 0),
            ("load_offset", nv_next2[3], addr_next2[3], 1),
        ], "valu": [
            ("&", v_tmp1_a, v_val_last2[0], v_one),
            ("&", v_tmp1_b, v_val_last2[1], v_one),
            ("&", v_tmp1_c, v_val_last2[2], v_one),
            ("&", v_tmp1_d, v_val_last2[3], v_one),
        ]})
        # gi=13: gather D[2,3], idx += mask
        self.instrs.append({"load": [
            ("load_offset", nv_next2[3], addr_next2[3], 2),
            ("load_offset", nv_next2[3], addr_next2[3], 3),
        ], "valu": [
            ("+", v_idx_last2[0], v_idx_last2[0], v_tmp1_a),
            ("+", v_idx_last2[1], v_idx_last2[1], v_tmp1_b),
            ("+", v_idx_last2[2], v_idx_last2[2], v_tmp1_c),
            ("+", v_idx_last2[3], v_idx_last2[3], v_tmp1_d),
        ]})
        # gi=14: gather D[4,5], idx < n_nodes
        self.instrs.append({"load": [
            ("load_offset", nv_next2[3], addr_next2[3], 4),
            ("load_offset", nv_next2[3], addr_next2[3], 5),
        ], "valu": [
            ("<", v_tmp1_a, v_idx_last2[0], v_n_nodes),
            ("<", v_tmp1_b, v_idx_last2[1], v_n_nodes),
            ("<", v_tmp1_c, v_idx_last2[2], v_n_nodes),
            ("<", v_tmp1_d, v_idx_last2[3], v_n_nodes),
        ], "alu": [("+", round_counter, round_counter, one_const)]})
        # gi=15: gather D[6,7], idx *= cmp + loop control
        self.instrs.append({"load": [
            ("load_offset", nv_next2[3], addr_next2[3], 6),
            ("load_offset", nv_next2[3], addr_next2[3], 7),
        ], "valu": [
            ("*", v_idx_last2[0], v_idx_last2[0], v_tmp1_a),
            ("*", v_idx_last2[1], v_idx_last2[1], v_tmp1_b),
            ("*", v_idx_last2[2], v_idx_last2[2], v_tmp1_c),
            ("*", v_idx_last2[3], v_idx_last2[3], v_tmp1_d),
        ], "alu": [("<", loop_cond, round_counter, self.scratch["rounds"])]})
        # Overlap initial store addr compute with loop branch (speculative: saves 1 cycle on loop exit)
        self.instrs.append({"flow": [("cond_jump", loop_cond, outer_loop_start_2)], "alu": [
            ("+", addr_tmp, self.scratch["inp_indices_p"], addr_consts[0]),
            ("+", addr_tmp2, self.scratch["inp_indices_p"], addr_consts[1]),
        ]})

        # Store ALL indices and values back to memory (pipelined: overlap addr compute with stores)
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
