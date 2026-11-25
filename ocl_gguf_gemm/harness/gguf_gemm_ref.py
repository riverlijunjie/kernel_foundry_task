# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Host-side model of the OpenVINO intel_gpu GGUF prefill GEMM (fc_gguf_gemm_opt.cl).

Mirrors, vectorised in NumPy:
  * random valid GGUF weights for Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0
  * the SG-shuffle weight repack (RepackGGUFWeightsShuffle in repack_gguf_weights.cpp)
  * the NumPy dequantisers used as the correctness oracle
  * the OV JIT preamble / tile config / dispatch (gguf/fc_gguf_opt.cpp FCGGUFGemmGenerator)
"""

from collections import OrderedDict, namedtuple

import numpy as np

SG = 16  # GGUF_GEMV_SG_SIZE

FORMATS = ["q4_0", "q4_1", "q4_k", "q5_k", "q6_k", "q8_0"]
FOUR_BIT = ("q4_0", "q4_1", "q4_k")
GEOM = {  # fmt -> (block_elem_count, block_byte_size)
    "q4_0": (32, 18), "q4_1": (32, 20), "q4_k": (256, 144),
    "q5_k": (256, 176), "q6_k": (256, 210), "q8_0": (32, 34),
}
FORMAT_DEFINE = {f: "GG_FMT_" + f.upper() for f in FORMATS}
ENTRY_POINT = {"q4_0": "fc_gguf_gemm_q4_0", "q4_1": "fc_gguf_gemm_q4_1", "q4_k": "fc_gguf_gemm_q4k",
               "q5_k": "fc_gguf_gemm_q5k", "q6_k": "fc_gguf_gemm_q6k", "q8_0": "fc_gguf_gemm_q8_0"}
_D_OFFSETS = {"q4_0": [0], "q4_1": [0, 2], "q4_k": [0, 2], "q5_k": [0, 2], "q6_k": [208], "q8_0": [0]}

OV_BUILD_OPTIONS = ["-cl-mad-enable", "-cl-std=CL3.0"]  # KernelGenerator::get_build_options (Intel)


# --------------------------------------------------------------------------------------
# Random native GGUF weights  [N, K/be * bb] uint8
# --------------------------------------------------------------------------------------
def gen_weight_bytes(fmt, N, K, seed=1234, scale_amp=0.02):
    be, bb = GEOM[fmt]
    assert K % 256 == 0
    nb = K // be
    rng = np.random.default_rng(seed)
    W = rng.integers(0, 256, size=(N, nb, bb), dtype=np.uint8)
    for off in _D_OFFSETS[fmt]:
        v = rng.uniform(-scale_amp, scale_amp, size=(N, nb)).astype(np.float16)
        v[v == 0] = np.float16(scale_amp)
        W[:, :, off:off + 2] = v.view(np.uint8).reshape(N, nb, 2)
    return W.reshape(N, nb * bb)


def _blocks(W, fmt):
    return W.reshape(W.shape[0], -1, GEOM[fmt][1])


def _h(B, off):
    """f16 field at byte offset `off` of every block -> float32 [N, nb]."""
    return np.ascontiguousarray(B[..., off:off + 2]).view(np.float16)[..., 0].astype(np.float32)


# --------------------------------------------------------------------------------------
# NumPy dequantisers (ggml reference semantics) -> float32 [N, K]
# --------------------------------------------------------------------------------------
def _scale_min_k4(sc):
    q = sc.astype(np.int32)
    s = np.empty(q.shape[:-1] + (8,), np.int32)
    m = np.empty_like(s)
    s[..., :4] = q[..., 0:4] & 63
    m[..., :4] = q[..., 4:8] & 63
    s[..., 4:] = (q[..., 8:12] & 0x0F) | ((q[..., 0:4] >> 6) << 4)
    m[..., 4:] = (q[..., 8:12] >> 4) | ((q[..., 4:8] >> 6) << 4)
    return s, m


def deq_q4_0(W):
    B = _blocks(W, "q4_0"); N, nb = B.shape[:2]
    d = _h(B, 0)
    qs = B[..., 2:18].astype(np.int32)
    q = np.concatenate([(qs & 0xF) - 8, (qs >> 4) - 8], -1)
    return (q * d[..., None]).reshape(N, -1).astype(np.float32)


def deq_q4_1(W):
    B = _blocks(W, "q4_1"); N, nb = B.shape[:2]
    d, m = _h(B, 0), _h(B, 2)
    qs = B[..., 4:20].astype(np.int32)
    q = np.concatenate([qs & 0xF, qs >> 4], -1)
    return (q * d[..., None] + m[..., None]).reshape(N, -1).astype(np.float32)


def deq_q8_0(W):
    B = _blocks(W, "q8_0"); N, nb = B.shape[:2]
    d = _h(B, 0)
    q = np.ascontiguousarray(B[..., 2:34]).view(np.int8).astype(np.float32)
    return (q * d[..., None]).reshape(N, -1)


def _k45_rw(B, fmt):
    """5-bit (Q5_K) / 4-bit (Q4_K) quants [N, nb, 8, 32] in dequant order + (sc, mn) [N, nb, 8]."""
    N, nb = B.shape[:2]
    s, m = _scale_min_k4(B[..., 4:16])
    if fmt == "q4_k":
        ql = B[..., 16:144].astype(np.int32).reshape(N, nb, 4, 32)
        hb = 0
    else:
        qh = B[..., 16:48].astype(np.int32)
        ql = B[..., 48:176].astype(np.int32).reshape(N, nb, 4, 32)
        bits = (qh[..., None, :] >> np.arange(8)[:, None]) & 1  # [N, nb, 8, 32]
        hb = 16 * bits.reshape(N, nb, 4, 2, 32)
    q = np.stack([ql & 0xF, ql >> 4], axis=-2) + hb  # [N, nb, 4, 2, 32]
    return q.reshape(N, nb, 8, 32), s, m


def _deq_k45(W, fmt):
    B = _blocks(W, fmt); N, nb = B.shape[:2]
    q, s, m = _k45_rw(B, fmt)
    d, dmin = _h(B, 0), _h(B, 2)
    out = (d[..., None, None] * s[..., None]) * q - (dmin[..., None, None] * m[..., None])
    return out.reshape(N, -1).astype(np.float32)


def deq_q4_k(W):
    return _deq_k45(W, "q4_k")


def deq_q5_k(W):
    return _deq_k45(W, "q5_k")


def _q6_rw(B):
    """Merged 6-bit quants [N, nb, 256] in the order used by the reference dequant (no -32)."""
    N, nb = B.shape[:2]
    ql = B[..., 0:128].astype(np.int32).reshape(N, nb, 2, 64)
    qh = B[..., 128:192].astype(np.int32).reshape(N, nb, 2, 32)
    qa, qb = ql[..., :32], ql[..., 32:]
    q = np.stack([(qa & 0xF) | (((qh >> 0) & 3) << 4), (qb & 0xF) | (((qh >> 2) & 3) << 4),
                  (qa >> 4) | (((qh >> 4) & 3) << 4), (qb >> 4) | (((qh >> 6) & 3) << 4)], axis=-2)
    return q.reshape(N, nb, 256)  # [half, quarter(4), l(32)]


def deq_q6_k(W):
    B = _blocks(W, "q6_k"); N, nb = B.shape[:2]
    q = _q6_rw(B).reshape(N, nb, 2, 4, 2, 16) - 32
    sc = np.ascontiguousarray(B[..., 192:208]).view(np.int8).astype(np.int32).reshape(N, nb, 2, 4, 2, 1)
    d = _h(B, 208)[..., None, None, None, None]
    return (d * sc * q).reshape(N, -1).astype(np.float32)


DEQUANT = {"q4_0": deq_q4_0, "q4_1": deq_q4_1, "q4_k": deq_q4_k, "q5_k": deq_q5_k,
           "q6_k": deq_q6_k, "q8_0": deq_q8_0}


def dequantize(fmt, W):
    return DEQUANT[fmt](W)


# --------------------------------------------------------------------------------------
# SG shuffle repack (same total size as the native layout)
# --------------------------------------------------------------------------------------
def _chunks(P):
    """[N, nb, pbytes] -> SG-interleaved 4-byte chunks per (row group, block): [c][lane][4]."""
    N, nb, pb = P.shape
    return P.reshape(N // SG, SG, nb, pb // 4, 4).transpose(0, 2, 3, 1, 4).reshape(-1)


def _lane_field(F):
    """[N, nb, k] -> per (row group, block) entry with lane-major k-byte fields: [lane][k]."""
    N, nb, k = F.shape
    return F.reshape(N // SG, SG, nb, k).transpose(0, 2, 1, 3).reshape(N // SG, nb, SG * k)


def _field_major(F):
    """[N, nb, f] (one byte per field per lane) -> [field][lane]."""
    N, nb, f = F.shape
    return F.reshape(N // SG, SG, nb, f).transpose(0, 2, 3, 1).reshape(N // SG, nb, f * SG)


def _u8(a):
    return np.ascontiguousarray(a.astype(np.uint8))


def _psl(B, rs, rm):
    N, nb = B.shape[:2]
    sl = (rs[..., 0::2] & 0xF) | ((rs[..., 1::2] & 0xF) << 4)
    ml = (rm[..., 0::2] & 0xF) | ((rm[..., 1::2] & 0xF) << 4)

    def hi(r):
        r = r.reshape(N, nb, 2, 4)
        return ((r[..., 0] & 0x30) >> 4) | ((r[..., 1] & 0x30) >> 2) | (r[..., 2] & 0x30) | ((r[..., 3] & 0x30) << 2)

    ent = np.concatenate([_lane_field(_u8(sl)), _lane_field(_u8(ml)), _lane_field(_u8(hi(rs))),
                          _lane_field(_u8(hi(rm))), _lane_field(_u8(B[..., 0:2])),
                          _lane_field(_u8(B[..., 2:4]))], axis=-1)
    assert ent.shape[-1] == SG * 16
    return ent.reshape(-1)


def _pqs_from_rw(rw):
    N, nb = rw.shape[:2]
    return _u8((rw[..., :16] & 0xF) | ((rw[..., 16:] & 0xF) << 4)).reshape(N, nb, 128)


def shuffle_weights(fmt, W):
    """Native GGUF bytes [N, nbn*bb] -> SG-shuffled flat bytes (RepackGGUFWeightsShuffle)."""
    be, bb = GEOM[fmt]
    N = W.shape[0]
    assert N % SG == 0
    B = _blocks(W, fmt)
    nbn = B.shape[1]
    if fmt in ("q4_k", "q5_k"):
        rw, rs, rm = _k45_rw(B, fmt)
        nb = nbn
        planes = [_chunks(_pqs_from_rw(rw))]
        if fmt == "q5_k":
            hb = (rw >> 4) & 1  # [N, nb, 8, 32]
            h = (hb.reshape(N, nb, 8, 8, 4) << np.arange(8)[:, None]).sum(-2)  # [N, nb, 8, 4]
            planes.append(_chunks(_u8(h).reshape(N, nb, 32)))
        planes.append(_psl(B, rs, rm))
    elif fmt == "q6_k":
        nb = nbn
        rw = _q6_rw(B)
        pql = _u8((rw.reshape(N, nb, 8, 32)[..., :16] & 0xF) | ((rw.reshape(N, nb, 8, 32)[..., 16:] & 0xF) << 4))
        r = rw.reshape(N, nb, 16, 4, 4)  # [j, g, k]
        pqh = _u8(((r[..., 0, :] & 0x30) >> 4) | ((r[..., 1, :] & 0x30) >> 2) | (r[..., 2, :] & 0x30) | ((r[..., 3, :] & 0x30) << 2))
        ps = np.ascontiguousarray(B[..., 192:208])
        planes = [_chunks(pql.reshape(N, nb, 128)), _chunks(pqh.reshape(N, nb, 64)),
                  _field_major(ps).reshape(-1), _lane_field(_u8(B[..., 208:210])).reshape(-1)]
    else:  # small-block formats: 8 native 32-elem blocks -> one 256-elem super-block
        nb = nbn // 8
        B8 = B.reshape(N, nb, 8, bb)
        if fmt == "q8_0":
            pqs = B8[..., 2:34].reshape(N, nb, 256)
        else:
            off = 4 if fmt == "q4_1" else 2
            pqs = B8[..., off:off + 16].reshape(N, nb, 128)

        def soa(off):
            f = np.ascontiguousarray(B8[..., off:off + 2])  # [N, nb, 8, 2]
            return f.reshape(N // SG, SG, nb, 8, 2).transpose(0, 2, 3, 1, 4).reshape(-1)

        planes = [_chunks(np.ascontiguousarray(pqs)), soa(0)]
        if fmt == "q4_1":
            planes.append(soa(2))
    out = np.concatenate([p.reshape(-1) for p in planes])
    assert out.size == W.size, (fmt, out.size, W.size)
    return out


# --------------------------------------------------------------------------------------
# OV JIT preamble / tile config / dispatch (FCGGUFGemmGenerator)
# --------------------------------------------------------------------------------------
_TYPES = {  # dtype -> (ocl, convert, as, val_max, val_min, size)
    "f16": ("half", "convert_half", "as_half", "HALF_MAX", "-HALF_MAX", 2),
    "f32": ("float", "convert_float", "as_float", "FLT_MAX", "-FLT_MAX", 4),
}


def _type_consts(name, dt):
    ocl, conv, asf, vmax, vmin, sz = _TYPES[dt]
    return OrderedDict([
        ("%s_TYPE" % name, ocl), ("%s_VAL_MAX" % name, vmax), ("%s_VAL_MIN" % name, vmin),
        ("%s_VAL_ONE" % name, "(%s) 1" % ocl), ("%s_VAL_ZERO" % name, "(%s) 0" % ocl),
        ("TO_%s_TYPE(v)" % name, "%s(v)" % conv), ("TO_%s_TYPE_SAT(v)" % name, "%s_sat(v)" % conv),
        ("AS_%s_TYPE(v)" % name, "%s(v)" % asf),
        ("%s_MAX_FUNC" % name, "fmax"), ("%s_MIN_FUNC" % name, "fmin"), ("%s_ABS_FUNC" % name, "fabs"),
        ("%s_TYPE_SIZE" % name, str(sz)), ("%s_IS_FP" % name, "1"),
    ])


OclGemmConfig = namedtuple("OclGemmConfig", "row_groups token_groups token_local row_blocks slm_jblk large_grf n_band")


def pick_config(fmt, N, K, xe3):
    """pick_ocl_gemm_config()."""
    if not xe3:
        return OclGemmConfig(2, 4, 8, 1, 4, False, 0)
    four_bit = fmt in FOUR_BIT
    cfg = OclGemmConfig(4, 4, 16, 2, 4, True, 0) if four_bit and N % 128 == 0 else OclGemmConfig(4, 4, 8, 1, 4, True, 0)
    be, bb = GEOM[fmt]
    w_bytes = N * K // be * bb
    band = 12 << 20
    if not four_bit and (16 << 20) < w_bytes <= (32 << 20):
        ngroups = N // (SG * cfg.row_groups * cfg.row_blocks)
        cfg = cfg._replace(n_band=(ngroups * band + w_bytes - 1) // w_bytes)
    return cfg


def n_align(cfg):
    return SG * cfg.row_groups * cfg.row_blocks


def build_preamble(entry, consts):
    s = ["// JIT preamble (OV-faithful): " + entry,
         "#define FUNC(name) _##name##_%s" % entry,
         "#define FUNC_CALL(name) _##name##_%s" % entry,
         "#define CONST_ARRAY_DECL(name) __constant size_t _##name##_%s []" % entry,
         "#define CONST_ARRAY_REF(name) _##name##_%s" % entry,
         "#define KERNEL(name) __kernel void %s" % entry,
         "#define KERNEL_ID %s" % entry]
    s += ["#define %s %s" % (k, v) for k, v in consts.items()]
    return "\n".join(s) + "\n"


def jit_constants(fmt, cfg, in_dt="f16", out_dt="f16"):
    c = OrderedDict()
    c["OPTIONAL_SHAPE_INFO_ARG"] = ""
    c["OPTIONAL_SHAPE_INFO_TENSOR"] = ""
    c.update(_type_consts("INPUT0", in_dt))
    c.update(_type_consts("OUTPUT", out_dt))
    c["ROW_GROUPS"] = str(cfg.row_groups)
    c["TOKEN_GROUPS"] = str(cfg.token_groups)
    c["TOKEN_LOCAL"] = str(cfg.token_local)
    c["ROW_BLOCKS"] = str(cfg.row_blocks)
    c["SLM_JBLK"] = str(cfg.slm_jblk)
    c["PF_A"] = "1"
    c["GG_N_BAND"] = str(cfg.n_band)
    c["GG_A_F32"] = "1" if in_dt == "f32" else "0"
    c["GG_OUT_F32"] = "1" if out_dt == "f32" else "0"
    c[FORMAT_DEFINE[fmt]] = "1"
    return c


def build_options(cfg):
    opts = list(OV_BUILD_OPTIONS)
    if cfg.large_grf:
        opts.append("-cl-intel-256-GRF-per-thread")
    return opts


def dispatch(cfg, M, N):
    """get_dispatch_data_func(): (global, local)."""
    tokens_per_thread = 8 * cfg.token_groups
    token_slots = cfg.token_local // cfg.row_blocks
    ntiles = (M + tokens_per_thread - 1) // tokens_per_thread
    ngroups = N // (SG * cfg.row_groups * cfg.row_blocks)
    gy = ((ntiles + token_slots - 1) // token_slots) * cfg.token_local
    return (ngroups * SG, gy, 1), (SG, cfg.token_local, 1)


CACHE_FLUSH_CL = """
__kernel void cache_flush(__global float* scratch, const uint n_elems, __global float* sink) {
    const uint gid = get_global_id(0);
    const uint nthreads = get_global_size(0);
    float acc = 0.0f;
    for (uint i = gid; i < n_elems; i += nthreads) {
        const float v = scratch[i] + 1.0f;
        scratch[i] = v;
        acc += v;
    }
    if (acc == -1.0f)
        sink[gid] = acc;
}
"""

if __name__ == "__main__":
    for f in FORMATS:
        W = gen_weight_bytes(f, 32, 512)
        D = dequantize(f, W)
        S = shuffle_weights(f, W)
        print("%-5s deq %s [% .3f, % .3f]  shuffled %d B" % (f, D.shape, D.min(), D.max(), S.size))
