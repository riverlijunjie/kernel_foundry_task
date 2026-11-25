# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OV-faithful JIT codegen for the extracted MoE 3-GEMM SwiGLU kernel
(moe_3gemm_swiglu_mlp.cl), GGUF-native-decode path used by Qwen3.6-35B-A3B decode.

Reproduces (1:1) two things OV does at runtime:

1. The build preamble (graph/impls/ocl_v2/utils/kernel_generator.cpp) — the FUNC/FUNC_CALL/
   KERNEL decoration macros + one `#define <name> <value>` per JitConstant. Identical to the
   gguf_kernels harness (kernel-agnostic), copied here so moe_kernel is self-contained.

2. The per-stage JitConstants from moe/moe_3gemm_swiglu_opt.cpp:
     - add_common_consts()               (MAX_TOPK, EXPERT_NUM, HIDDEN/INTERMEDIATE_SIZE, N_BLOCK,
                                           SUBGROUP_SIZE, SUBGROUP_NUM, *_GROUP_SIZE, MOE_DTYPE,
                                           HAS_ZP, WEIGHT_IS_SIGNED, and the GGUF block-decode set:
                                           WEIGHT_COMPRESSEION_DT=3, MOE_WEI_DT=uchar,
                                           GATEUP_DECODE/DOWN_DECODE/SHARED_DECODE, MOE_GGUF_DP4A,
                                           MOE_GGUF_LISA)
     - MoE3GemmSwigluMLPGateUp::get_jit_constants  (GATE_UP_ENABLE=1, GATE_ACT_*, SHARED_EXPERT_ENABLE)
     - MoE3GemmSwigluMLPDown::get_jit_constants     (DOWN_ENABLE=1, SHARED_EXPERT_ENABLE)

Defaults match the model run on Arc B390 (Xe3, arch>=xe2): Q4_K experts, MOE_GGUF_DP4A=1 (default),
MOE_GGUF_LISA=0 (default), SUBGROUP_SIZE=32, N_BLOCK=4, SUBGROUP_NUM=8, SWIGLU activation.
We build with SHARED_EXPERT_ENABLE=0 (routed experts only) — this is the dominant GEMV cost and the
per-expert decode math is identical to the shared expert; it keeps the standalone arg set minimal.
"""
import os
from collections import OrderedDict

# ----- performance-tuning constants (moe_3gemm_swiglu_opt.cpp) -----
N_BLOCK = 4
SUBGROUP_NUM = 8
SUBGROUP_SIZE = 32          # info.arch >= xe2 ? 32 : 16  (B390 is Xe3)

# GGUF MoE decode-type code (moe_gguf_decode_code): 1=Q4_K 2=Q5_K 3=Q6_K 4=Q8_0
GGUF_DECODE_CODE = {"gguf_q4_k": 1, "gguf_q5_k": 2, "gguf_q6_k": 3, "gguf_q8_0": 4}
# GGUF block geometry (block_elem_count, block_byte_size) — mirrors OV core element_type.hpp
GGUF_BLK = {"gguf_q4_k": (256, 144), "gguf_q5_k": (256, 176), "gguf_q6_k": (256, 210), "gguf_q8_0": (32, 34)}


def _decoration_macro(name, prefix, postfix):
    sep = "##_" if postfix else ""
    return "#define %s(name) %s _##name%s%s" % (name, prefix, sep, postfix)


def build_preamble(entry_point, consts):
    """Full JIT macro preamble for one kernel build (mirrors CodeBuilder decoration+value macros)."""
    s = ["//==================================================== JIT preamble (OV-faithful)",
         "// Kernel entry point: %s" % entry_point,
         "#define FUNC(name) _##name##_%s" % entry_point,
         "#define FUNC_CALL(name) _##name##_%s" % entry_point,
         "#define CONST_ARRAY_DECL(name) __constant size_t _##name##_%s []" % entry_point,
         "#define CONST_ARRAY_REF(name) _##name##_%s" % entry_point,
         "#define KERNEL(name) __kernel void %s" % entry_point,
         "#define KERNEL_ID %s" % entry_point]
    for k, v in consts.items():
        s.append("#define %s %s" % (k, v))
    s.append("//==================================================== end JIT preamble")
    return "\n".join(s) + "\n"


def _common_consts(fmt, hidden, inter, experts, topk):
    """add_common_consts() for the GGUF native-decode path (WEIGHT_COMPRESSEION_DT=3)."""
    code = GGUF_DECODE_CODE[fmt]
    c = OrderedDict()
    c["MAX_TOPK"] = str(topk)
    c["EXPERT_NUM"] = str(experts)
    c["HIDDEN_SIZE"] = str(hidden)
    c["INTERMEDIATE_SIZE"] = str(inter)
    c["N_BLOCK"] = str(N_BLOCK)
    c["SUBGROUP_SIZE"] = str(SUBGROUP_SIZE)
    c["SUBGROUP_NUM"] = str(SUBGROUP_NUM)
    # group_size == max -> gate_up uses hidden, down uses inter (unused by the GGUF super-block path,
    # but FAKE_GROUP_SIZE is still #defined from these, so give the same valid values OV would).
    c["GATE_UP_GROUP_SIZE"] = str(hidden)
    c["DOWN_GROUP_SIZE"] = str(inter)
    c["MOE_DTYPE"] = "half"
    c["MOE_DTYPE_SIZE"] = "2"
    c["HAS_ZP"] = "0"
    c["WEIGHT_IS_SIGNED"] = "0"
    # GGUF native block decode
    c["WEIGHT_COMPRESSEION_DT"] = "3"
    c["MOE_WEI_DT"] = "uchar"
    c["MOE_SCALE_DT"] = "half"
    c["MOE_ZP_DT"] = "uchar"
    c["GATEUP_DECODE"] = str(code)
    c["DOWN_DECODE"] = str(code)
    c["SHARED_DECODE"] = str(code)
    c["MOE_GGUF_DP4A"] = "0" if os.environ.get("MOE_NO_DP4A") else "1"   # default ON, Xe2+
    c["MOE_GGUF_LISA"] = "0"   # default OFF
    return c


def gate_up_jit(fmt, hidden, inter, experts, topk):
    """MoE3GemmSwigluMLPGateUp::get_jit_constants (SWIGLU, shared disabled)."""
    ep = "moe_3gemm_swiglu_mlp_gate_up_%s_H%d_I%d_E%d" % (fmt, hidden, inter, experts)
    c = _common_consts(fmt, hidden, inter, experts, topk)
    c["GATE_UP_ENABLE"] = "1"
    c["SHARED_EXPERT_ENABLE"] = "0"
    return ep, c


def down_jit(fmt, hidden, inter, experts, topk):
    """MoE3GemmSwigluMLPDown::get_jit_constants (shared disabled)."""
    ep = "moe_3gemm_swiglu_mlp_down_%s_H%d_I%d_E%d" % (fmt, hidden, inter, experts)
    c = _common_consts(fmt, hidden, inter, experts, topk)
    c["DOWN_ENABLE"] = "1"
    c["SHARED_EXPERT_ENABLE"] = "0"
    return ep, c


# ----- GGUF MoE prefill transcode kernel (moe_gguf_transcode.cl) -----
# Constants from moe/moe_3gemm_swiglu_opt.cpp:
#   MOE_GGUF_REQUANT_GROUP = 32   (elements sharing one f16 scale)
#   MOE_GGUF_TRANSCODE_LWS = 16   (SG_SIZE / local x)
MOE_GGUF_REQUANT_GROUP = 32
MOE_GGUF_TRANSCODE_LWS = 16

# moe_transcode_target(): Q4_0/Q4_K -> i4 (qmax 7); Q5_K/Q6_K/Q8_0 -> i8 (qmax 127).
#   fmt -> (TRANSCODE_TO_I4, QMAX)
TRANSCODE_TARGET = {
    "gguf_q4_0": (True, 7),
    "gguf_q4_k": (True, 7),
    "gguf_q5_k": (False, 127),
    "gguf_q6_k": (False, 127),
    "gguf_q8_0": (False, 127),
}
# moe_gguf_type_jit_flag(): the -D<flag>=1 selecting the block decoder in the .cl.
GGUF_TYPE_FLAG = {
    "gguf_q4_0": "GGUF_IS_Q4_0",
    "gguf_q8_0": "GGUF_IS_Q8_0",
    "gguf_q4_k": "GGUF_IS_Q4_K",
    "gguf_q5_k": "GGUF_IS_Q5_K",
    "gguf_q6_k": "GGUF_IS_Q6_K",
}


def transcode_jit(fmt, experts, n, k):
    """MoE3GemmSwigluGgufTranscode::get_jit_constants / get_entry_point (moe_gguf_transcode.cl).

    Mirrors the impl 1:1: NUM_EXPERTS/N_SIZE/K_SIZE + GGUF block geometry + REQUANT_GROUP +
    TRANSCODE_TO_I4/QMAX (from moe_transcode_target) + SG_SIZE + the GGUF_IS_* decoder flag.
    Entry point == get_kernel_name()+"_"+type_name+"_E<E>_N<N>_K<K>".
    """
    be, bb = GGUF_BLK[fmt]
    to_i4, qmax = TRANSCODE_TARGET[fmt]
    ep = "moe_gguf_transcode_%s_E%d_N%d_K%d" % (fmt, experts, n, k)
    c = OrderedDict()
    c["NUM_EXPERTS"] = str(experts)
    c["N_SIZE"] = str(n)
    c["K_SIZE"] = str(k)
    c["GGUF_BLOCK_ELEM"] = str(be)
    c["GGUF_BLOCK_BYTES"] = str(bb)
    c["REQUANT_GROUP"] = str(MOE_GGUF_REQUANT_GROUP)
    c["TRANSCODE_TO_I4"] = "1" if to_i4 else "0"
    c["QMAX"] = str(qmax)
    c["SG_SIZE"] = str(MOE_GGUF_TRANSCODE_LWS)
    c[GGUF_TYPE_FLAG[fmt]] = "1"
    return ep, c


# OV build options for Intel devices (KernelGenerator::get_build_options): -cl-mad-enable + CL3.0
# (CL3.0 exposes cl_khr_integer_dot_product dp4a builtins used by the MOE_GGUF_DP4A path).
OV_BUILD_OPTIONS = ["-cl-mad-enable", "-cl-std=CL3.0"]
