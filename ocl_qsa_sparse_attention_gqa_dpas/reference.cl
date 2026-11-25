// Standalone prelude: replaces include/batch_headers/common.cl and the JIT plumbing of the OpenVINO GPU plugin.
#if defined(cl_khr_fp16)
#    pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#    pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif
#ifdef cl_intel_subgroups_short
#    pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif
#ifdef cl_intel_subgroups
#    pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif
#ifdef cl_intel_subgroup_local_block_io
#    pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif
#ifdef cl_intel_subgroups_char
#    pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#ifdef cl_intel_required_subgroup_size
#    define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#    define REQD_SUB_GROUP_SIZE(sg_size)
#endif
#define FUNC(name) name
#define FUNC_CALL(name) name
#define KERNEL(name) __kernel void name

// [REFERENCE_START]
// Q3 GQA-aware systolic sparse attention of the paged SparsePA primitive (OpenVINO intel_gpu, sparse_attention_gqa_dpas.cl;
// paged addressing, prefill variant: the dense SparseSDPA addressing and the decode split-K variant are not included).
//
// One work-group = 4 subgroups = ONE token x ONE KV head. A KV step is 16 rows (16 / r selected blocks); each subgroup
// owns a Dh / 4 slice of the QK reduction and a Dv / 4 slice of the output.
//   QK: A = K rows of the step (2D block reads), B = Q with lane = query head of the KV group (the remaining lanes
//       carry zero Q); the score tile has lane = head, so the softmax is in-lane. Worker partials meet in SLM.
//   PV: A = P with lane = KV row (transposed through a per-worker SLM tile), B = V in VNNI form, so the accumulator
//       has element = head, lane = output column.
// Valid rows of a step are always the prefix [0, valid_kv), made of page-local runs of R rows. A step that is one
// contiguous 16-row run is read with one 2D message per 32 columns (VNNI transform for V), otherwise with one message
// per block and 32 columns, all issued before any is consumed (rows past a short block read as zero; V is packed to
// VNNI in registers). The 2D path needs 64-byte aligned K/V buffers; without 2D block IO rows are
// read one by one.
// p stays below 2^8 in half thanks to a lazy rescale of the running max (scores are in log2 units).
//
// Quantized K/V (paged u8 pages, layout below) are read row by row and dequantized to half in registers, so the dpas
// operands are identical to the f16 ones; the 2D block path is f16 only.
//
// Per token the kernel attends to the `selected_counts[token]` selected blocks of R tokens (block ids in
// `selected_blocks[token, 0..K_TOPK)`) plus the not yet complete tail block [(pos + 1) / R * R, pos] of the token.
//
// Quantized page layout (per (block, kv head) page, u8 tensor [blocks, Hkv, page_size + parameter rows, row bytes]):
// `page_size` token-major data rows of QSA_X_ROW_BYTES bytes (4-bit codes packed two per byte, low nibble = even
// channel), followed at QSA_X_PARAM_OFFSET by the parameters:
//   BY_TOKEN   : per token [scale f16][zp f16]               (the zp half is unused when symmetric)
//   BY_CHANNEL : per group of QSA_K_GROUP tokens [scale f16 x head_size][zp f16 x head_size]  (no zp part when symmetric)
// Dequantization: x = (q - zp) * scale. A symmetric cache stores biased codes (zp = 128 / 8).
// f16 caches (MODE 0) are plain [blocks, Hkv, page_size, head_size].
//
// JIT: QSA_HEADS, QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_COMPRESS_RATIO, QSA_BLOCK_TOPK, QSA_SCALE,
//      QSA_IS_CAUSAL, QSA_PAGE_SIZE, SUBGROUP_SIZE,
//      QSA_{K,V}_{MODE,BITS,SYM,GROUP,ROW_BYTES,PAGE_BYTES,PARAM_OFFSET}   (MODE: 0 = f16, 1 = by token, 2 = by channel)
// Default dispatch (see the launch block below): gws = [tokens * 4 * SG, QSA_KV_HEADS, 1], lws = [4 * SG, 1, 1]

// [LAUNCH_START]
// How task.py launches this program (read from this comment block; the values below are the defaults of the plugin kernel).
//   kernel         entry point name
//   args           kernel arguments in order, comma separated.
//                  Buffers: query, key_cache, value_cache, past_lens, subsequence_begins, block_indices,
//                  block_indices_begins, selected_blocks, selected_counts, output.
//                  Scalars: num_tokens, num_seqs, q_base, q_pitch (uint) and scale (float).
//   gws, lws       global / local work size, three integer expressions each. Variables: tokens, num_seqs, sg, heads,
//                  kv_heads, head_dim, topk, block_size, page_size. Operators: + - * // %, min(), max(), cdiv().
//   subgroup_size  value of the SUBGROUP_SIZE macro (8, 16 or 32).
//   grf_256        1 compiles with -cl-intel-256-GRF-per-thread.
//   build_options  extra -cl-* options and -DNAME=VALUE macros, separated by spaces.
// The program is always compiled with the JIT macros listed above (the kernel may ignore or #undef them). Every output
// element must be written, so gws has to cover all tokens and heads.
//
// kernel = sparse_attention_paged_gqa_dpas
// args = query, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, selected_blocks, selected_counts, output, num_tokens, num_seqs, q_base, q_pitch
// gws = tokens * 4 * sg, kv_heads, 1
// lws = 4 * sg, 1, 1
// subgroup_size = 16
// grf_256 = 1
// build_options =
// [LAUNCH_END]

// Subgroups per work-group; keep in sync with lws.
#define QSA_XMX_WORKERS 4

#ifndef QSA_MAX_NE
#    define QSA_MAX_NE 32
#endif
#define QSA_SG SUBGROUP_SIZE
#define QSA_NEG_INF (-3.0e38f)
#define QSA_LOG2E 1.4426950408889634f

// ---------------------------------------------------------------------------
// Subgroup block IO: out[j] is element (lane + SG * j) of the row (the layout produced by intel_sub_group_block_read_us*)
// ---------------------------------------------------------------------------
inline void FUNC(qsa_rd)(const __global ushort* p, __private ushort* out, const uint n) {
    uint j = 0;
    for (; j + 8 <= n; j += 8) {
        ushort8 r = intel_sub_group_block_read_us8(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        out[j + 4] = r.s4; out[j + 5] = r.s5; out[j + 6] = r.s6; out[j + 7] = r.s7;
    }
    if (n - j >= 4) {
        ushort4 r = intel_sub_group_block_read_us4(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        j += 4;
    }
    if (n - j >= 2) {
        ushort2 r = intel_sub_group_block_read_us2(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1;
        j += 2;
    }
    if (n - j >= 1) {
        out[j] = intel_sub_group_block_read_us(p + j * QSA_SG);
    }
}

// Subgroup block read of bytes: out[j] is byte (lane + SG * j) of p.
inline void FUNC(qsa_rd_uc)(const __global uchar* p, __private uchar* out, const uint n) {
    uint j = 0;
    for (; j + 8 <= n; j += 8) {
        uchar8 r = intel_sub_group_block_read_uc8(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        out[j + 4] = r.s4; out[j + 5] = r.s5; out[j + 6] = r.s6; out[j + 7] = r.s7;
    }
    if (n - j >= 4) {
        uchar4 r = intel_sub_group_block_read_uc4(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        j += 4;
    }
    if (n - j >= 2) {
        uchar2 r = intel_sub_group_block_read_uc2(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1;
        j += 2;
    }
    if (n - j >= 1) {
        out[j] = intel_sub_group_block_read_uc(p + j * QSA_SG);
    }
}

// ---------------------------------------------------------------------------
// Address arithmetic
// ---------------------------------------------------------------------------
// Element offset of token `token_in_page` of an f16 cache page.
inline uint FUNC(qsa_kv_cache_offset)(uint page, uint kv_head, uint token_in_page, uint kv_heads, uint pa_block_size, uint head_size) {
    return ((page * kv_heads + kv_head) * pa_block_size + token_in_page) * head_size;
}

// Resolves a flattened paged token to its sequence and absolute position; false for rows past the last real token.
inline bool FUNC(qsa_resolve_token_paged)(const __global int* subsequence_begins,
                                          const __global int* past_lens,
                                          uint num_seqs,
                                          uint token_idx,
                                          __private uint* seq,
                                          __private uint* abs_pos) {
    *seq = 0;
    *abs_pos = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            *seq = i;
            *abs_pos = (uint)past_lens[i] + token_idx - (uint)subsequence_begins[i];
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Quantized main-KV cache helpers
// ---------------------------------------------------------------------------
#define QSA_K_QUANT (QSA_K_MODE != 0)
#define QSA_V_QUANT (QSA_V_MODE != 0)
#if QSA_K_QUANT
#    define QSA_KC_T uchar
#else
#    define QSA_KC_T half
#endif
#if QSA_V_QUANT
#    define QSA_VC_T uchar
#else
#    define QSA_VC_T half
#endif

inline float FUNC(qsa_q_zero_sym)(const uint bits) {
    return bits == 8 ? 128.0f : 8.0f;
}

// (code - zp) * scale in f16 arithmetic: every step is rounded to half, so all kernels see the same dequantized values.
inline float FUNC(qsa_q_dq)(const float code, const float zp, const float scale) {
    return convert_float(convert_half_rte(convert_float(convert_half_rte(code - zp)) * scale));
}

// Byte offsets of token `tok` of `page`: its data row and its parameters (a token dword for by-token caches, the
// group record for by-channel ones).
inline void FUNC(qsa_q_loc)(const uint page,
                            const uint kv_head,
                            const uint tok,
                            const uint mode,
                            const uint group,
                            const uint page_bytes,
                            const uint row_bytes,
                            const uint param_off,
                            const uint head,
                            const uint sym,
                            const uint kv_heads,
                            __private uint* row_off,
                            __private uint* par_off) {
    const uint pb = (page * kv_heads + kv_head) * page_bytes;
    *row_off = pb + tok * row_bytes;
    *par_off = mode == 1 ? pb + param_off + tok * 4 : pb + param_off + (tok / group) * (head * (sym ? 2 : 4));
}

// Byte offsets of token `tok` of `page` of the K / V cache of kv head `kv_head` (a variable of the caller).
#define QSA_K_LOC(page, tok, ro, po) \
    FUNC_CALL(qsa_q_loc)(page, kv_head, tok, QSA_K_MODE, QSA_K_GROUP, QSA_K_PAGE_BYTES, QSA_K_ROW_BYTES, QSA_K_PARAM_OFFSET, QSA_K_HEAD_SIZE, QSA_K_SYM, QSA_KV_HEADS, ro, po)
#define QSA_V_LOC(page, tok, ro, po) \
    FUNC_CALL(qsa_q_loc)(page, kv_head, tok, QSA_V_MODE, 1, QSA_V_PAGE_BYTES, QSA_V_ROW_BYTES, QSA_V_PARAM_OFFSET, QSA_V_HEAD_SIZE, QSA_V_SYM, QSA_KV_HEADS, ro, po)

// Codes of channels x0 + SG * j + lane (j < ne) of the packed row `row`. 8-bit rows are read with block reads; 4-bit
// rows with one byte load per lane and channel (a block read plus shuffles is slower).
inline void FUNC(qsa_q_load_codes)(const __global uchar* row, const uint x0, __private float* code, const uint ne, const uint bits, const bool is_signed) {
    if (bits == 8) {
        uchar b[QSA_MAX_NE];
        FUNC_CALL(qsa_rd_uc)(row + x0, b, ne);
        unroll_for (uint j = 0; j < ne; j++) {
            code[j] = is_signed ? convert_float((char)b[j]) : convert_float(b[j]);
        }
    } else {
        const uint lane = get_sub_group_local_id();
        unroll_for (uint j = 0; j < ne; j++) {
            const uchar v = row[(x0 + QSA_SG * j + lane) >> 1];
            code[j] = convert_float((lane & 1) ? (v >> 4) : (v & 0xF));
        }
    }
}

// Dequantized channels x0 + SG * j + lane (j < ne) of a quantized paged row, block-read.
inline void FUNC(qsa_q_load_cols)(const __global uchar* cache,
                                  const uint row_off,
                                  const uint par_off,
                                  const uint x0,
                                  __private float* out,
                                  const uint ne,
                                  const uint mode,
                                  const uint bits,
                                  const uint sym,
                                  const uint head) {
    FUNC_CALL(qsa_q_load_codes)(cache + row_off, x0, out, ne, bits, false);
    if (mode == 1) {
        const uint pd = *(const __global uint*)(cache + par_off);
        const float scale = convert_float(as_half((ushort)(pd & 0xFFFFu)));
        const float zp = sym ? FUNC_CALL(qsa_q_zero_sym)(bits) : convert_float(as_half((ushort)(pd >> 16)));
        unroll_for (uint j = 0; j < ne; j++) {
            out[j] = FUNC_CALL(qsa_q_dq)(out[j], zp, scale);
        }
    } else {
        ushort sc[QSA_MAX_NE], zq[QSA_MAX_NE];
        FUNC_CALL(qsa_rd)((const __global ushort*)(cache + par_off) + x0, sc, ne);
        if (!sym) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(cache + par_off + head * 2) + x0, zq, ne);
        }
        unroll_for (uint j = 0; j < ne; j++) {
            const float zp = sym ? FUNC_CALL(qsa_q_zero_sym)(bits) : convert_float(as_half(zq[j]));
            out[j] = FUNC_CALL(qsa_q_dq)(out[j], zp, convert_float(as_half(sc[j])));
        }
    }
}

// Dequantized channels of the K / V row at byte offset `ro` with parameter locator `po`: out[j] is channel x0 + SG * j + lane.
#define QSA_LOAD_K_COLS(ro, po, x0, out, ne) FUNC_CALL(qsa_q_load_cols)((const __global uchar*)key_cache, ro, po, x0, out, ne, QSA_K_MODE, QSA_K_BITS, QSA_K_SYM, QSA_K_HEAD_SIZE)
#define QSA_LOAD_V_COLS(ro, po, x0, out, ne) FUNC_CALL(qsa_q_load_cols)((const __global uchar*)value_cache, ro, po, x0, out, ne, QSA_V_MODE, QSA_V_BITS, QSA_V_SYM, QSA_V_HEAD_SIZE)

// ---------------------------------------------------------------------------
// XMX building blocks (SIMD16). One KV step = 16 rows, row i of the step is held by lane i; the lane axis of the
// score tile is the query column.
// ---------------------------------------------------------------------------
#define QSA_XW QSA_XMX_WORKERS
#define QSA_XDH (QSA_K_HEAD_SIZE / QSA_XW)
#define QSA_XDV (QSA_V_HEAD_SIZE / QSA_XW)
#define QSA_XKT (QSA_XDH / 16)
#define QSA_XST_FLOATS (2 * 8 * QSA_SG)

#ifdef cl_intel_subgroup_local_block_io
#    define QSA_SLM_WRITE8(p, v) intel_sub_group_block_write8((__local uint*)(p), as_uint8(v))
#    define QSA_SLM_READ8(p) as_float8(intel_sub_group_block_read8((__local uint*)(p)))
#else
#    define QSA_SLM_WRITE8(p, v) FUNC_CALL(qsa_slm_write8)((__local float*)(p), v)
#    define QSA_SLM_READ8(p) FUNC_CALL(qsa_slm_read8)((__local float*)(p))
inline void FUNC(qsa_slm_write8)(__local float* p, float8 v) {
    const uint l = get_sub_group_local_id();
    p[0 * QSA_SG + l] = v.s0; p[1 * QSA_SG + l] = v.s1; p[2 * QSA_SG + l] = v.s2; p[3 * QSA_SG + l] = v.s3;
    p[4 * QSA_SG + l] = v.s4; p[5 * QSA_SG + l] = v.s5; p[6 * QSA_SG + l] = v.s6; p[7 * QSA_SG + l] = v.s7;
}
inline float8 FUNC(qsa_slm_read8)(__local float* p) {
    const uint l = get_sub_group_local_id();
    return (float8)(p[0 * QSA_SG + l], p[1 * QSA_SG + l], p[2 * QSA_SG + l], p[3 * QSA_SG + l],
                    p[4 * QSA_SG + l], p[5 * QSA_SG + l], p[6 * QSA_SG + l], p[7 * QSA_SG + l]);
}
#endif

#if SUBGROUP_SIZE != 16
#    error "the systolic attention needs a 16-wide subgroup"
#endif
#if !QSA_IS_CAUSAL
#    error "only causal QSA is supported"
#endif
#if QSA_COMPRESS_RATIO != 2 && QSA_COMPRESS_RATIO != 4 && QSA_COMPRESS_RATIO != 8 && QSA_COMPRESS_RATIO != 16
#    error "R must be 2, 4, 8 or 16"
#endif
#if (QSA_K_HEAD_SIZE % (QSA_XW * 32)) != 0 || (QSA_V_HEAD_SIZE % (QSA_XW * 32)) != 0
#    error "worker slices must be 32-half aligned"
#endif

#define GQA_R QSA_COMPRESS_RATIO
#define GQA_BLOCKS (16 / QSA_COMPRESS_RATIO)
#define GQA_RESCALE_LOG2 8.0f
#define GQA_VT (QSA_XDV / 16)  // 16-column PV tiles of a worker
#define GQA_HEADS (QSA_HEADS / QSA_KV_HEADS)
#define GQA_HH (GQA_HEADS > 8 ? 2 : 1)  // 8-head halves of the PV accumulator

#undef GQA_BLOCK_READ
#ifdef cl_intel_subgroup_2d_block_io
#    define GQA_2D 1
#    if QSA_COMPRESS_RATIO == 2
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_2r16x2c
#    elif QSA_COMPRESS_RATIO == 4
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_4r16x2c
#    elif QSA_COMPRESS_RATIO == 8
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_8r16x2c
#    else
#        define GQA_BLOCK_READ intel_sub_group_2d_block_read_16b_16r16x2c
#    endif
#else
#    define GQA_2D 0
#endif

#define GQA_S8(t, o) as_short8((ushort8)((t)[(o) + 0], (t)[(o) + 1], (t)[(o) + 2], (t)[(o) + 3], (t)[(o) + 4], (t)[(o) + 5], (t)[(o) + 6], (t)[(o) + 7]))
#define GQA_I8(t, o) as_int8((uint8)((t)[(o) + 0], (t)[(o) + 1], (t)[(o) + 2], (t)[(o) + 3], (t)[(o) + 4], (t)[(o) + 5], (t)[(o) + 6], (t)[(o) + 7]))

// K of one step as A operands: ka[k][h] holds rows 8h..8h+7 (element) x columns worker * XDH + 16 k + lane.
// Quantized K: `koff` is the byte offset of this lane's row and `kpar` its parameter locator (see QSA_LOAD_K_COLS).
inline void FUNC(gqa_load_k)(const __global QSA_KC_T* key_cache,
                             const uint koff,
                             const uint kpar,
                             const uint valid_kv,
                             const bool contiguous,
                             const uint worker,
                             __private short8 (*ka)[2]) {
    const int row_bytes = QSA_K_HEAD_SIZE * 2;
    const uint x0 = worker * QSA_XDH;
#if QSA_K_QUANT
    {
        unroll_for (uint i = 0; i < 16; i++) {
            const uint ro = sub_group_broadcast(koff, i);
            const uint po = sub_group_broadcast(kpar, i);
            float x[QSA_XKT];
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                x[k] = 0.0f;
            }
            if (i < valid_kv) {
                QSA_LOAD_K_COLS(ro, po, x0, x, QSA_XKT);
            }
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                ka[k][i / 8][i % 8] = as_short(convert_half_rte(x[k]));
            }
        }
    }
#elif GQA_2D
    {
        if (contiguous) {
            const __global half* base = key_cache + sub_group_broadcast(koff, 0);
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                ushort t[32];
                intel_sub_group_2d_block_read_16b_16r16x2c(base, row_bytes, 16, row_bytes, (int2)(x0 + m * 32, 0), t);
                ka[2 * m][0] = GQA_S8(t, 0);
                ka[2 * m][1] = GQA_S8(t, 8);
                ka[2 * m + 1][0] = GQA_S8(t, 16);
                ka[2 * m + 1][1] = GQA_S8(t, 24);
            }
            return;
        }
        // Every read is issued before any is consumed (an empty block reads one row of a valid address and is
        // masked), so the latencies overlap instead of serializing on per-block branches.
        ushort t[GQA_BLOCKS][QSA_XKT / 2][2 * GQA_R];
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const uint rows = valid_kv > j * GQA_R ? min(valid_kv - j * GQA_R, (uint)GQA_R) : 0u;
            const __global half* base = key_cache + sub_group_broadcast(koff, j * GQA_R);
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                GQA_BLOCK_READ(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(x0 + m * 32, 0), t[j][m]);
            }
        }
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const bool empty = valid_kv <= j * GQA_R;
            unroll_for (uint m = 0; m < QSA_XKT / 2; m++) {
                unroll_for (uint r = 0; r < GQA_R; r++) {
                    ka[2 * m][(j * GQA_R + r) / 8][(j * GQA_R + r) % 8] = empty ? 0 : (short)t[j][m][r];
                    ka[2 * m + 1][(j * GQA_R + r) / 8][(j * GQA_R + r) % 8] = empty ? 0 : (short)t[j][m][GQA_R + r];
                }
            }
        }
    }
#else
    unroll_for (uint i = 0; i < 16; i++) {
        ushort r[QSA_XKT];
        const uint ko = sub_group_broadcast(koff, i);
        if (i < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(key_cache + ko + x0), r, QSA_XKT);
        } else {
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                r[k] = 0;
            }
        }
        unroll_for (uint k = 0; k < QSA_XKT; k++) {
            ka[k][i / 8][i % 8] = (short)r[k];
        }
    }
#endif
}

// V of one step as VNNI B operands: vb[t] element j packs rows 2j, 2j+1 of column worker * XDV + 16 t + lane.
inline void FUNC(gqa_load_v)(const __global QSA_VC_T* value_cache,
                             const uint voff,
                             const uint vpar,
                             const uint valid_kv,
                             const bool contiguous,
                             const uint worker,
                             __private int8* vb) {
    const int row_bytes = QSA_V_HEAD_SIZE * 2;
    const uint x0 = worker * QSA_XDV;
#if QSA_V_QUANT
    {
        unroll_for (uint i = 0; i < 16; i += 2) {
            const uint ro0 = sub_group_broadcast(voff, i);
            const uint ro1 = sub_group_broadcast(voff, i + 1);
            const uint po0 = sub_group_broadcast(vpar, i);
            const uint po1 = sub_group_broadcast(vpar, i + 1);
            float v0[GQA_VT], v1[GQA_VT];
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                v0[t] = 0.0f;
                v1[t] = 0.0f;
            }
            if (i < valid_kv) {
                QSA_LOAD_V_COLS(ro0, po0, x0, v0, GQA_VT);
            }
            if (i + 1 < valid_kv) {
                QSA_LOAD_V_COLS(ro1, po1, x0, v1, GQA_VT);
            }
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                vb[t][i / 2] = (int)((uint)as_ushort(convert_half_rte(v0[t])) | ((uint)as_ushort(convert_half_rte(v1[t])) << 16));
            }
        }
    }
#elif GQA_2D
    {
        if (contiguous) {
            const __global half* base = value_cache + sub_group_broadcast(voff, 0);
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                uint t[16];
                intel_sub_group_2d_block_read_transform_16b_16r16x2c(base, row_bytes, 16, row_bytes, (int2)(x0 + m * 32, 0), t);
                vb[2 * m] = GQA_I8(t, 0);
                vb[2 * m + 1] = GQA_I8(t, 8);
            }
            return;
        }
        // A VNNI transform read is at least 16 rows high, so short blocks are read plainly (all reads first, as for K)
        // and packed in registers.
        ushort t[GQA_BLOCKS][GQA_VT / 2][2 * GQA_R];
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const uint rows = valid_kv > j * GQA_R ? min(valid_kv - j * GQA_R, (uint)GQA_R) : 0u;
            const __global half* base = value_cache + sub_group_broadcast(voff, j * GQA_R);
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                GQA_BLOCK_READ(base, row_bytes, (int)max(rows, 1u), row_bytes, (int2)(x0 + m * 32, 0), t[j][m]);
            }
        }
        unroll_for (uint j = 0; j < GQA_BLOCKS; j++) {
            const bool empty = valid_kv <= j * GQA_R;
            unroll_for (uint m = 0; m < GQA_VT / 2; m++) {
                unroll_for (uint q = 0; q < GQA_R / 2; q++) {
                    const uint lo = (uint)t[j][m][2 * q] | ((uint)t[j][m][2 * q + 1] << 16);
                    const uint hi = (uint)t[j][m][GQA_R + 2 * q] | ((uint)t[j][m][GQA_R + 2 * q + 1] << 16);
                    vb[2 * m][j * GQA_R / 2 + q] = empty ? 0 : (int)lo;
                    vb[2 * m + 1][j * GQA_R / 2 + q] = empty ? 0 : (int)hi;
                }
            }
        }
    }
#else
    unroll_for (uint i = 0; i < 16; i += 2) {
        ushort r0[GQA_VT], r1[GQA_VT];
        const uint vo0 = sub_group_broadcast(voff, i);
        const uint vo1 = sub_group_broadcast(voff, i + 1);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            r0[t] = 0;
            r1[t] = 0;
        }
        if (i < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(value_cache + vo0 + x0), r0, GQA_VT);
        }
        if (i + 1 < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(value_cache + vo1 + x0), r1, GQA_VT);
        }
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            vb[t][i / 2] = (int)((uint)r0[t] | ((uint)r1[t] << 16));
        }
    }
#endif
}

// One KV step of the online softmax for this worker: QK partials summed over the workers through `slm_st`, softmax
// with a lazy rescale, P^T through the worker's `slm_p` tile, then PV into ro.
inline void FUNC(gqa_step)(const __global QSA_KC_T* key_cache,
                           const __global QSA_VC_T* value_cache,
                           const uint koff,
                           const uint voff,
                           const uint kpar,
                           const uint vpar,
                           const uint valid_kv,
                           const bool kcont,
                           const bool vcont,
                           const uint worker,
                           __private const int8* rq,
                           __local float* slm_st,  // [W][2][8][SG] of the current slot
                           __local uint8* slm_p,   // [16] of this worker and slot
                           __private float* running_max,
                           __private float* running_sum,
                           __private float8 (*ro)[GQA_VT]) {
    const uint lane = get_sub_group_local_id();
    short8 ka[QSA_XKT][2];
    FUNC_CALL(gqa_load_k)(key_cache, koff, kpar, valid_kv, kcont, worker, ka);
    // V is in flight during QK and the softmax.
    int8 vb[GQA_VT];
    FUNC_CALL(gqa_load_v)(value_cache, voff, vpar, valid_kv, vcont, worker, vb);

    // QK partial of this worker, summed over the workers in the fixed order 0..W-1 (bit-identical everywhere).
    float8 st0 = (float8)(0.0f);
    float8 st1 = (float8)(0.0f);
    unroll_for (uint k = 0; k < QSA_XKT; k++) {
        st0 = intel_sub_group_f16_f16_matrix_mad_k16(ka[k][0], rq[k], st0);
        st1 = intel_sub_group_f16_f16_matrix_mad_k16(ka[k][1], rq[k], st1);
    }
    QSA_SLM_WRITE8(slm_st + worker * QSA_XST_FLOATS, st0);
    QSA_SLM_WRITE8(slm_st + worker * QSA_XST_FLOATS + 8 * QSA_SG, st1);
    barrier(CLK_LOCAL_MEM_FENCE);
    st0 = QSA_SLM_READ8(slm_st);
    st1 = QSA_SLM_READ8(slm_st + 8 * QSA_SG);
    unroll_for (uint w = 1; w < QSA_XW; w++) {
        st0 += QSA_SLM_READ8(slm_st + w * QSA_XST_FLOATS);
        st1 += QSA_SLM_READ8(slm_st + w * QSA_XST_FLOATS + 8 * QSA_SG);
    }
    float st[16];
    vstore8(st0, 0, st);
    vstore8(st1, 0, st + 8);

    // Padded head lanes see zero Q (finite scores, never stored): only the rows past valid_kv need masking.
    if (valid_kv != 16) {
        unroll_for (uint i = 0; i < 16; i++) {
            st[i] = i >= valid_kv ? QSA_NEG_INF : st[i];
        }
    }
    float tile_max = st[0];
    unroll_for (uint i = 1; i < 16; i++) {
        tile_max = fmax(tile_max, st[i]);
    }
    // Lazy rescale: the running max only moves when some lane grows by more than 2^GQA_RESCALE_LOG2.
    float m = *running_max;
    float s = *running_sum;
    if (sub_group_any(tile_max > m + GQA_RESCALE_LOG2)) {
        const float new_max = fmax(m, tile_max);
        const float corr = native_exp2(m - new_max);
        s *= corr;
        m = new_max;
        unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
            float8 c;
            unroll_for (uint i = 0; i < 8; i++) {
                c[i] = sub_group_broadcast(corr, hh * 8 + i);
            }
            unroll_for (uint t = 0; t < GQA_VT; t++) {
                ro[hh][t] *= c;
            }
        }
    }
    float p[16];
    unroll_for (uint i = 0; i < 16; i++) {
        p[i] = native_exp2(st[i] - m);
        s += p[i];
    }
    *running_max = m;
    *running_sum = s;

    // P^T through SLM: lane h stores its row P[h][0..15]; a block read then gives lane = KV row, element = head.
    slm_p[lane] = (uint8)(as_uint(convert_half2_rte((float2)(p[0], p[1]))), as_uint(convert_half2_rte((float2)(p[2], p[3]))),
                          as_uint(convert_half2_rte((float2)(p[4], p[5]))), as_uint(convert_half2_rte((float2)(p[6], p[7]))),
                          as_uint(convert_half2_rte((float2)(p[8], p[9]))), as_uint(convert_half2_rte((float2)(p[10], p[11]))),
                          as_uint(convert_half2_rte((float2)(p[12], p[13]))), as_uint(convert_half2_rte((float2)(p[14], p[15]))));
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
        const __local ushort* pt = (const __local ushort*)slm_p + hh * 128;
#ifdef cl_intel_subgroup_local_block_io
        const short8 pa = as_short8(intel_sub_group_block_read_us8(pt));
#else
        const short8 pa = (short8)((short)pt[lane], (short)pt[16 + lane], (short)pt[32 + lane], (short)pt[48 + lane],
                                   (short)pt[64 + lane], (short)pt[80 + lane], (short)pt[96 + lane], (short)pt[112 + lane]);
#endif
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            ro[hh][t] = intel_sub_group_f16_f16_matrix_mad_k16(pa, vb[t], ro[hh][t]);
        }
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_paged_gqa_dpas)(const __global half* query,                 // [tokens, Hq*Dh]
                                        const __global QSA_KC_T* key_cache,         // paged K cache (f16 or u8 pages)
                                        const __global QSA_VC_T* value_cache,       // paged V cache (f16 or u8 pages)
                                        const __global int* past_lens,              // [num_seqs]
                                        const __global int* subsequence_begins,     // [num_seqs+1]
                                        const __global int* block_indices,          // [used_blocks]
                                        const __global int* block_indices_begins,   // [num_seqs+1]
                                        const __global int* selected_blocks,        // [tokens, QSA_BLOCK_TOPK]
                                        const __global int* selected_counts,        // [tokens]
                                        __global half* output,                      // [tokens, Hq*Dv]
                                        const uint num_entries,                     // tokens
                                        const uint num_seqs,
                                        const uint q_base,
                                        const uint q_pitch) {
    const uint HQ = QSA_HEADS;
    const uint HKV = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint DV = QSA_V_HEAD_SIZE;
    const uint R = QSA_COMPRESS_RATIO;
    const uint K_TOPK = QSA_BLOCK_TOPK;
    const uint HEADS = GQA_HEADS;

    __local float slm_st[2][QSA_XW][2][8][QSA_SG];
    __local uint8 slm_p[2][QSA_XW][16];  // per worker: row h = P[head h][kv rows 0..15] as halves

    const uint worker = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint entry = get_group_id(0);
    const uint kv_head = get_group_id(1);
    if (kv_head >= HKV || entry >= num_entries) {
        return;
    }
    const uint head_base = kv_head * HEADS;
    const uint lane_blk = lane / R;
    const uint lane_rr = lane % R;

    const uint token = entry;
    uint seq = 0, abs_pos = 0;
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token, &seq, &abs_pos)) {
        return;
    }
    const uint page_begin = (uint)block_indices_begins[seq];
    const uint count = (uint)selected_counts[token];
    const uint sel_base = token * K_TOPK;
    const uint tail_begin = (abs_pos + 1) / R * R;
    const uint tail_rows = abs_pos + 1 - tail_begin;
    const uint q_head0 = q_base + token * q_pitch + head_base * DH;
#define GQA_Q_HEAD(h) (q_head0 + (h) * DH)
#define GQA_OUT_ROW(h) (token * HQ + head_base + (h))

    // Q as the B operand: this lane's own head row slice, pre-scaled by scale * log2(e) in the half domain.
    int8 rq[QSA_XKT];
    unroll_for (uint k = 0; k < QSA_XKT; k++) {
        half16 qh = (half16)((half)0.0f);
        if (lane < HEADS) {
            qh = vload16(0, query + GQA_Q_HEAD(lane) + worker * QSA_XDH + k * 16);
        }
        qh = qh * (half)((float)QSA_SCALE * QSA_LOG2E);
        rq[k] = as_int8(qh);
    }

    float running_max = QSA_NEG_INF;  // of head `lane`
    float running_sum = 0.0f;
    float8 ro[GQA_HH][GQA_VT];  // element = head 8 hh + i, lane = output column 16 t + lane of the worker slice
    unroll_for (uint hh = 0; hh < GQA_HH; hh++) {
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            ro[hh][t] = (float8)(0.0f);
        }
    }

    const uint steps = (count + GQA_BLOCKS - 1) / GQA_BLOCKS;
    const uint step_begin = 0;
    const uint step_end = steps + (tail_rows != 0 ? 1 : 0);

#define GQA_LOAD_ID(step) (((step) < steps && (step) * GQA_BLOCKS + lane_blk < count) ? selected_blocks[sel_base + (step) * GQA_BLOCKS + lane_blk] : -1)
// Lane = KV row of the step: its validity and absolute position (step == steps is the tail).
#define GQA_META(step, id, lv, pos)          \
    if ((step) == steps) {                   \
        lv = lane < tail_rows;               \
        pos = tail_begin + lane;             \
    } else {                                 \
        lv = (id) >= 0;                      \
        pos = (uint)(id) * R + lane_rr;      \
    }
#define GQA_PAGE(lv, pos) ((lv) ? (uint)block_indices[page_begin + (pos) / QSA_PAGE_SIZE] : 0u)

    // Software-pipelined metadata: the next step's pages and the following step's ids are in flight while a step
    // computes, so the KV loads never wait on the id -> page chain.
    bool cur_lv;
    uint cur_pos;
    const int first_id = GQA_LOAD_ID(step_begin);
    GQA_META(step_begin, first_id, cur_lv, cur_pos);
    uint cur_page = GQA_PAGE(cur_lv, cur_pos);
    int next_id = GQA_LOAD_ID(step_begin + 1);

    for (uint step = step_begin; step < step_end; step++) {
        bool nlv;
        uint npos;
        GQA_META(step + 1, next_id, nlv, npos);
        const uint npage = GQA_PAGE(nlv, npos);
        next_id = GQA_LOAD_ID(step + 2);

        const bool lv = cur_lv;
        const uint pos = cur_pos;
        uint koff = 0, voff = 0, kpar = 0, vpar = 0;
        if (lv) {
#if QSA_K_QUANT
            QSA_K_LOC(cur_page, pos % QSA_PAGE_SIZE, &koff, &kpar);
#else
            koff = FUNC_CALL(qsa_kv_cache_offset)(cur_page, kv_head, pos % QSA_PAGE_SIZE, HKV, QSA_PAGE_SIZE, DH);
#endif
#if QSA_V_QUANT
            QSA_V_LOC(cur_page, pos % QSA_PAGE_SIZE, &voff, &vpar);
#else
            voff = FUNC_CALL(qsa_kv_cache_offset)(cur_page, kv_head, pos % QSA_PAGE_SIZE, HKV, QSA_PAGE_SIZE, DV);
#endif
        }
        const uint valid_kv = sub_group_reduce_add(lv ? 1u : 0u);
        // A step read with one 2D message per operand needs ascending contiguous f16 rows.
        bool kcont = false, vcont = false;
#if !QSA_K_QUANT
        kcont = valid_kv == 16 && sub_group_all(koff == sub_group_broadcast(koff, 0) + lane * DH);
#endif
#if !QSA_V_QUANT
        vcont = valid_kv == 16 && sub_group_all(voff == sub_group_broadcast(voff, 0) + lane * DV);
#endif

        FUNC_CALL(gqa_step)(key_cache, value_cache, koff, voff, kpar, vpar, valid_kv, kcont, vcont, worker, rq, &slm_st[step & 1][0][0][0][0],
                            &slm_p[step & 1][worker][0], &running_max, &running_sum, ro);
        cur_lv = nlv;
        cur_pos = npos;
        cur_page = npage;
    }

    // Never store the padded heads: that would overwrite the next KV group or token.
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    unroll_for (uint h = 0; h < GQA_HEADS; h++) {
        const float ih = sub_group_broadcast(inv, h);
        __global ushort* dst = (__global ushort*)(output + GQA_OUT_ROW(h) * DV + worker * QSA_XDV);
        unroll_for (uint t = 0; t < GQA_VT; t++) {
            intel_sub_group_block_write_us(dst + t * 16, as_ushort(convert_half_rte(ro[h / 8][t][h % 8] * ih)));
        }
    }
}
// [REFERENCE_END]
