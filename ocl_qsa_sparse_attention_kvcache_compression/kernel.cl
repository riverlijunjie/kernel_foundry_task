// [USER_INSTRUCTIONS_START]
// Optimize the main KV cache write ("KV cache compression") of the OpenVINO intel_gpu SparsePA primitive: the new
// K / V rows are quantized (u8 / u4) and scattered into the paged cache. Target: Intel Xe2 (SIMD16), OpenCL.
//
// Workload (Qwen4-exp config.json): 2 kv heads, head size 256 (K and V), page size 16 tokens. Benchmarked config: key cache
// BY_CHANNEL u8 asymmetric (group of 16 tokens, kernel sparse_attention_kv_write_kchannel), value cache BY_TOKEN u8
// asymmetric (kernel sparse_attention_kv_write). Benchmarks: single-sequence prefill of 1K..64K tokens in one call, and
// decode (1 new token, context 1K..64K) repeated for the 12 full-attention layers. The decode token is the last of its
// 16-token group, so the BY_CHANNEL kernel re-quantizes a full group. Correctness is also checked for u4, symmetric and
// BY_TOKEN-key variants.
//
// Fixed by task.py (do not change):
//  - the two entry points sparse_attention_kv_write and sparse_attention_kv_write_kchannel, their argument lists and
//    the JIT macros; both programs are compiled from this file, the second one with -DQSA_KV_WRITE_KCHANNEL=1 and
//    -cl-intel-256-GRF-per-thread, with SUBGROUP_SIZE=16;
//  - the dispatch geometry given in the header comment of the kernels (work-group = one subgroup);
//  - the cache byte format and the arithmetic contract: scale / zero point are f16, codes = clamp(x / scale + zp) rounded
//    half up, dequantization x = (q - zp) * scale in f16 steps. Results may differ from the reference only by rare
//    one-code rounding differences; bytes outside the written tokens / groups must stay untouched.
//
// Where the time goes / ideas: qsa_q_store_codes writes one byte per lane and channel (strided byte stores, shuffles for
// 4-bit); the BY_CHANNEL kernel loads every key row twice (two passes, 2-byte loads per lane); kv_write scans all
// sequences per token. Candidates: vectorized / block loads and packed stores, keeping the group rows in registers
// between the passes (256 GRF mode), cheaper sequence lookup, fewer instructions in the quantization math.
// [USER_INSTRUCTIONS_END]

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

// [EVOLVE_START]
// Main KV cache write of the paged SparsePA primitive (OpenVINO intel_gpu, sparse_attention_kv_write.cl). Two kernels
// share this source, selected by QSA_KV_WRITE_KCHANNEL.
//
// Default (sparse_attention_kv_write): one subgroup copies (f16 caches) or quantizes (BY_TOKEN quantized caches) the K and
// V row of one (token, kv head) into the paged cache. The input rows may be padded and offset (key_pitch / key_offset).
// BY_CHANNEL keys are skipped here.
//   Dispatch: gws = [SG, QSA_KV_HEADS, tokens_padded], lws = [SG, 1, 1]
//
// QSA_KV_WRITE_KCHANNEL (sparse_attention_kv_write_kchannel): BY_CHANNEL key cache write. One subgroup owns one
// (sequence, token group, kv head); a group is QSA_K_GROUP consecutive tokens of one page that share one scale / zero
// point per channel, so appending a token re-quantizes the whole group: pass 1 gets the per-channel range (cached tokens
// dequantized with their old parameters, new tokens from the key input), pass 2 re-quantizes and rewrites every row,
// then the parameters. Groups without a new token are never touched.
//   Dispatch: gws = [(tokens / QSA_K_GROUP + 2 * num_seqs) * SG, QSA_KV_HEADS, 1], lws = [SG, 1, 1]
//
// Quantized page layout (per (block, kv head) page, u8 tensor [blocks, Hkv, page_size + parameter rows, row bytes]):
// `page_size` token-major data rows of QSA_X_ROW_BYTES bytes (4-bit codes packed two per byte, low nibble = even
// channel), followed at QSA_X_PARAM_OFFSET by the parameters:
//   BY_TOKEN   : per token [scale f16][zp f16]               (the zp half is unused when symmetric)
//   BY_CHANNEL : per group of QSA_K_GROUP tokens [scale f16 x head_size][zp f16 x head_size]  (no zp part when symmetric)
// Dequantization: x = (q - zp) * scale. A symmetric cache stores biased codes (zp = 128 / 8).
//
// JIT: QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_PAGE_SIZE, SUBGROUP_SIZE,
//      QSA_{K,V}_{MODE,BITS,SYM,GROUP,ROW_BYTES,PAGE_BYTES,PARAM_OFFSET}   (MODE: 0 = f16, 1 = by token, 2 = by channel)

#ifndef QSA_MAX_NE
#    define QSA_MAX_NE 32
#endif
#define QSA_SG SUBGROUP_SIZE

// Subgroup block write: element (lane + SG * j) of the row is v[j].
inline void FUNC(qsa_wr)(__global ushort* p, const __private ushort* v, const uint n) {
    uint j = 0;
    for (; j + 8 <= n; j += 8) {
        intel_sub_group_block_write_us8(p + j * QSA_SG, (ushort8)(v[j + 0], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7]));
    }
    if (n - j >= 4) {
        intel_sub_group_block_write_us4(p + j * QSA_SG, (ushort4)(v[j + 0], v[j + 1], v[j + 2], v[j + 3]));
        j += 4;
    }
    if (n - j >= 2) {
        intel_sub_group_block_write_us2(p + j * QSA_SG, (ushort2)(v[j + 0], v[j + 1]));
        j += 2;
    }
    if (n - j >= 1) {
        intel_sub_group_block_write_us(p + j * QSA_SG, v[j]);
    }
}

// Element offset of token `token_in_page` of an f16 cache page.
inline uint FUNC(qsa_kv_cache_offset)(uint page, uint kv_head, uint token_in_page, uint kv_heads, uint pa_block_size, uint head_size) {
    return ((page * kv_heads + kv_head) * pa_block_size + token_in_page) * head_size;
}

// Smallest scale kept, so that 1 / scale stays finite in f16 / f32.
#define QSA_KVQ_MIN_SCALE (1.0f / 16384.0f)

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

// Quantizes a lane-mapped row (x[j] = channel lane + SG * j, j < ne) to per-lane codes. scale / zp are the f16-rounded
// parameters, uniform over the subgroup.
inline void FUNC(qsa_q_quant_row)(const __private float* x,
                                  __private uint* code,
                                  const uint ne,
                                  const uint bits,
                                  const uint sym,
                                  __private half* scale_h,
                                  __private half* zp_h) {
    float mn = x[0];
    float mx = x[0];
    unroll_for (uint j = 1; j < ne; j++) {
        mn = fmin(mn, x[j]);
        mx = fmax(mx, x[j]);
    }
    mn = sub_group_reduce_min(mn);
    mx = sub_group_reduce_max(mx);
    const float qmax = bits == 8 ? 255.0f : 15.0f;
    float scale, zp;
    if (sym) {
        scale = fmax(mx, -mn) / (bits == 8 ? 127.0f : 7.0f);
        scale = fmax(scale, QSA_KVQ_MIN_SCALE);
        zp = FUNC_CALL(qsa_q_zero_sym)(bits);
    } else {
        mn = fmin(mn, 0.0f);
        mx = fmax(mx, 0.0f);
        scale = fmax((mx - mn) / qmax, QSA_KVQ_MIN_SCALE);
        zp = -mn / scale;
    }
    *scale_h = convert_half_rte(scale);
    *zp_h = convert_half_rte(zp);
    const float inv = 1.0f / convert_float(*scale_h);
    const float zpf = convert_float(*zp_h);
    unroll_for (uint j = 0; j < ne; j++) {
        const float y = clamp(x[j] * inv + zpf, 0.0f, qmax) + 0.5f;
        code[j] = (uint)y;
    }
}

// Stores lane-mapped codes as one packed row (4-bit: adjacent channels share a byte, low nibble = even channel).
inline void FUNC(qsa_q_store_codes)(__global uchar* cache, const uint row_off, const __private uint* code, const uint ne, const uint bits) {
    const uint lane = get_sub_group_local_id();
    unroll_for (uint j = 0; j < ne; j++) {
        const uint c = lane + QSA_SG * j;
        if (bits == 8) {
            cache[row_off + c] = (uchar)code[j];
        } else {
            const uint partner = intel_sub_group_shuffle(code[j], lane ^ 1);
            if ((lane & 1) == 0) {
                cache[row_off + (c >> 1)] = (uchar)(code[j] | (partner << 4));
            }
        }
    }
}

#if QSA_KV_WRITE_KCHANNEL

#define KCH_NE (QSA_K_HEAD_SIZE / QSA_SG)

#if (QSA_K_HEAD_SIZE % SUBGROUP_SIZE) != 0 || QSA_K_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE
#    error "head size must be a whole number of subgroup rows of at most QSA_MAX_NE elements per lane"
#endif
#if QSA_K_MODE != 2
#    error "this kernel only writes BY_CHANNEL keys"
#endif
#if (QSA_PAGE_SIZE % QSA_K_GROUP) != 0
#    error "a group must fit a page"
#endif

// Maps work-group `wg` of a grid sized `tokens / group + 2 * num_seqs` onto the (sequence, token group) it re-quantizes:
// sequence i owns the groups touched by its new tokens.
inline bool FUNC(qsa_resolve_group)(const __global int* subsequence_begins,
                                    const __global int* past_lens,
                                    const uint num_seqs,
                                    const uint wg,
                                    const uint group_size,
                                    __private uint* seq,
                                    __private uint* group) {
    uint acc = 0;
    for (uint i = 0; i < num_seqs; i++) {
        const uint q_len = (uint)subsequence_begins[i + 1] - (uint)subsequence_begins[i];
        if (q_len == 0) {
            continue;
        }
        const uint past = (uint)past_lens[i];
        const uint first = past / group_size;
        const uint n = (past + q_len - 1) / group_size - first + 1;
        if (wg < acc + n) {
            *seq = i;
            *group = first + (wg - acc);
            return true;
        }
        acc += n;
    }
    return false;
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_kv_write_kchannel)(const __global half* key,                  // [num_tokens, Hkv*Dh] token-major
                                           const __global int* past_lens,             // [num_seqs]
                                           const __global int* block_indices,         // [used_blocks]
                                           const __global int* block_indices_begins,  // [num_seqs+1]
                                           const __global int* subsequence_begins,    // [num_seqs+1]
                                           __global uchar* key_cache,                 // [num_blocks, Hkv, page_size + parameter rows, row bytes]
                                           const uint key_pitch,
                                           const uint key_offset,
                                           const uint num_seqs) {
    const uint KV_HEADS = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint PAGE = QSA_PAGE_SIZE;
    const uint G = QSA_K_GROUP;
    const uint BITS = QSA_K_BITS;
    const uint ROWB = QSA_K_ROW_BYTES;
    const uint SYM = QSA_K_SYM;
    const uint REC_BYTES = DH * (SYM ? 2 : 4);
    const float QMAX = BITS == 8 ? 255.0f : 15.0f;

    const uint head = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    uint seq = 0, group = 0;
    if (head >= KV_HEADS || !FUNC_CALL(qsa_resolve_group)(subsequence_begins, past_lens, num_seqs, get_group_id(0), G, &seq, &group)) {
        return;
    }

    const uint seq_begin = (uint)subsequence_begins[seq];
    const uint past = (uint)past_lens[seq];
    const uint total = past + (uint)subsequence_begins[seq + 1] - seq_begin;
    const uint g0 = group * G;
    const uint n_valid = total - g0 < G ? total - g0 : G;
    const uint n_old = past > g0 ? (past - g0 < n_valid ? past - g0 : n_valid) : 0;

    const uint tok0 = g0 % PAGE;
    const uint physical_page = (uint)block_indices[(uint)block_indices_begins[seq] + g0 / PAGE];
    const uint pb = (physical_page * KV_HEADS + head) * QSA_K_PAGE_BYTES;
    const uint rec = pb + QSA_K_PARAM_OFFSET + (tok0 / G) * REC_BYTES;

    float old_scale[QSA_K_HEAD_SIZE / QSA_SG];
    float old_zp[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        const uint c = lane + QSA_SG * j;
        old_scale[j] = 1.0f;
        old_zp[j] = 0.0f;
        if (n_old > 0) {
            old_scale[j] = convert_float(*(const __global half*)(key_cache + rec + c * 2));
            old_zp[j] = SYM ? FUNC_CALL(qsa_q_zero_sym)(BITS) : convert_float(*(const __global half*)(key_cache + rec + DH * 2 + c * 2));
        }
    }

    // Row t of the group as f32 (cached rows dequantized with their old parameters in f16 arithmetic).
#define KCH_LOAD_ROW(t, h)                                                                                                                      \
    {                                                                                                                                           \
        if ((t) < n_old) {                                                                                                                      \
            const uint row_off = pb + (tok0 + (t)) * ROWB;                                                                                      \
            unroll_for (uint j = 0; j < KCH_NE; j++) {                                                                                          \
                const uint c = lane + QSA_SG * j;                                                                                               \
                float code;                                                                                                                     \
                if (BITS == 8) {                                                                                                                \
                    code = convert_float(key_cache[row_off + c]);                                                                               \
                } else {                                                                                                                        \
                    const uchar b = key_cache[row_off + (c >> 1)];                                                                              \
                    code = convert_float((c & 1) ? (b >> 4) : (b & 0xF));                                                                       \
                }                                                                                                                               \
                h[j] = FUNC_CALL(qsa_q_dq)(code, old_zp[j], old_scale[j]);                                                                      \
            }                                                                                                                                   \
        } else {                                                                                                                                \
            const __global ushort* src = (const __global ushort*)key + key_offset + (seq_begin + (g0 + (t) - past)) * key_pitch + head * DH + lane; \
            unroll_for (uint j = 0; j < KCH_NE; j++) {                                                                                          \
                h[j] = convert_float(as_half(src[j * QSA_SG]));                                                                                 \
            }                                                                                                                                   \
        }                                                                                                                                       \
    }

    // Pass 1: per-channel range, always including 0 so that the zero point stays inside the f16 range.
    float mn[QSA_K_HEAD_SIZE / QSA_SG];
    float mx[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        mn[j] = 0.0f;
        mx[j] = 0.0f;
    }
    for (uint t = 0; t < n_valid; t++) {
        float h[QSA_K_HEAD_SIZE / QSA_SG];
        KCH_LOAD_ROW(t, h);
        unroll_for (uint j = 0; j < KCH_NE; j++) {
            mn[j] = fmin(mn[j], h[j]);
            mx[j] = fmax(mx[j], h[j]);
        }
    }

    half scale_h[QSA_K_HEAD_SIZE / QSA_SG];
    half zero_h[QSA_K_HEAD_SIZE / QSA_SG];
    float inv[QSA_K_HEAD_SIZE / QSA_SG];
    float zero_f[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        float scale, zero;
        if (SYM) {
            scale = fmax(fmax(mx[j], -mn[j]) / (BITS == 8 ? 127.0f : 7.0f), QSA_KVQ_MIN_SCALE);
            zero = FUNC_CALL(qsa_q_zero_sym)(BITS);
        } else {
            scale = fmax((mx[j] - mn[j]) / QMAX, QSA_KVQ_MIN_SCALE);
            zero = -mn[j] / scale;
        }
        scale_h[j] = convert_half_rte(scale);
        zero_h[j] = convert_half_rte(zero);
        inv[j] = 1.0f / convert_float(scale_h[j]);
        zero_f[j] = convert_float(zero_h[j]);
    }

    // Pass 2: re-quantize every row of the group with the new parameters.
    for (uint t = 0; t < n_valid; t++) {
        float h[QSA_K_HEAD_SIZE / QSA_SG];
        KCH_LOAD_ROW(t, h);
        uint code[QSA_K_HEAD_SIZE / QSA_SG];
        unroll_for (uint j = 0; j < KCH_NE; j++) {
            code[j] = (uint)(clamp(h[j] * inv[j] + zero_f[j], 0.0f, QMAX) + 0.5f);
        }
        FUNC_CALL(qsa_q_store_codes)(key_cache, pb + (tok0 + t) * ROWB, code, KCH_NE, BITS);
    }

    unroll_for (uint j = 0; j < KCH_NE; j++) {
        const uint c = lane + QSA_SG * j;
        *(__global half*)(key_cache + rec + c * 2) = scale_h[j];
        if (!SYM) {
            *(__global half*)(key_cache + rec + DH * 2 + c * 2) = zero_h[j];
        }
    }
}

#else  // QSA_KV_WRITE_KCHANNEL

#if (QSA_K_HEAD_SIZE % SUBGROUP_SIZE) != 0 || (QSA_V_HEAD_SIZE % SUBGROUP_SIZE) != 0 || QSA_K_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE || QSA_V_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE
#    error "head sizes must be whole subgroup rows of at most QSA_MAX_NE elements per lane"
#endif

#if QSA_K_MODE != 0
#    define QSA_KV_K_T uchar
#else
#    define QSA_KV_K_T half
#endif
#if QSA_V_MODE != 0
#    define QSA_KV_V_T uchar
#else
#    define QSA_KV_V_T half
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_kv_write)(const __global half* key,                   // [num_tokens, Hkv*Dh] token-major
                                  const __global half* value,                 // [num_tokens, Hkv*Dv] token-major
                                  const __global int* past_lens,              // [num_seqs]
                                  const __global int* block_indices,          // [used_blocks]
                                  const __global int* block_indices_begins,   // [num_seqs+1]
                                  const __global int* subsequence_begins,     // [num_seqs+1]
                                  __global QSA_KV_K_T* key_cache,             // [num_blocks, Hkv, page_size (+ parameter rows), row]
                                  __global QSA_KV_V_T* value_cache,           // [num_blocks, Hkv, page_size (+ parameter rows), row]
                                  const uint key_pitch,
                                  const uint key_offset,
                                  const uint value_pitch,
                                  const uint value_offset,
                                  const uint num_seqs) {
    const uint KV_HEADS = QSA_KV_HEADS;
    const uint K_HEAD = QSA_K_HEAD_SIZE;
    const uint V_HEAD = QSA_V_HEAD_SIZE;
    const uint PAGE = QSA_PAGE_SIZE;

    const uint head_idx = get_group_id(1);
    const uint token_idx = get_group_id(2);
    const uint lane = get_sub_group_local_id();

    if (token_idx >= (uint)subsequence_begins[num_seqs]) {
        return;
    }

    uint seq = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            seq = i;
            break;
        }
    }

    const uint abs_pos = (uint)past_lens[seq] + token_idx - (uint)subsequence_begins[seq];
    const uint physical_page = (uint)block_indices[(uint)block_indices_begins[seq] + abs_pos / PAGE];
    const uint token_in_page = abs_pos % PAGE;

    const uint k_src = key_offset + token_idx * key_pitch + head_idx * K_HEAD;
    const uint v_src = value_offset + token_idx * value_pitch + head_idx * V_HEAD;
    const uint kv_head = head_idx;
    const __global ushort* ks = (const __global ushort*)key + k_src + lane;
    const __global ushort* vs = (const __global ushort*)value + v_src + lane;

    // Every load is issued before the first store: the source and cache pointers may alias as far as the compiler
    // knows, so interleaving them would serialize one memory round trip per element.
#if QSA_K_MODE == 0
    ushort kr[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < K_HEAD / QSA_SG; j++) {
        kr[j] = ks[j * QSA_SG];
    }
#elif QSA_K_MODE == 1
    float kf[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < K_HEAD / QSA_SG; j++) {
        kf[j] = convert_float(as_half(ks[j * QSA_SG]));
    }
#endif
#if QSA_V_MODE == 0
    ushort vr[QSA_V_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < V_HEAD / QSA_SG; j++) {
        vr[j] = vs[j * QSA_SG];
    }
#else
    float vf[QSA_V_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < V_HEAD / QSA_SG; j++) {
        vf[j] = convert_float(as_half(vs[j * QSA_SG]));
    }
#endif

#if QSA_K_MODE == 0
    FUNC_CALL(qsa_wr)((__global ushort*)(key_cache + FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, K_HEAD)), kr, K_HEAD / QSA_SG);
#elif QSA_K_MODE == 1
    {
        uint ro, po;
        QSA_K_LOC(physical_page, token_in_page, &ro, &po);
        uint code[QSA_K_HEAD_SIZE / QSA_SG];
        half sh, zh;
        FUNC_CALL(qsa_q_quant_row)(kf, code, K_HEAD / QSA_SG, QSA_K_BITS, QSA_K_SYM, &sh, &zh);
        FUNC_CALL(qsa_q_store_codes)((__global uchar*)key_cache, ro, code, K_HEAD / QSA_SG, QSA_K_BITS);
        if (lane == 0) {
            *(__global uint*)((__global uchar*)key_cache + po) = (uint)as_ushort(sh) | ((uint)as_ushort(zh) << 16);
        }
    }
#endif

#if QSA_V_MODE == 0
    FUNC_CALL(qsa_wr)((__global ushort*)(value_cache + FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, V_HEAD)), vr, V_HEAD / QSA_SG);
#else
    {
        uint ro, po;
        QSA_V_LOC(physical_page, token_in_page, &ro, &po);
        uint code[QSA_V_HEAD_SIZE / QSA_SG];
        half sh, zh;
        FUNC_CALL(qsa_q_quant_row)(vf, code, V_HEAD / QSA_SG, QSA_V_BITS, QSA_V_SYM, &sh, &zh);
        FUNC_CALL(qsa_q_store_codes)((__global uchar*)value_cache, ro, code, V_HEAD / QSA_SG, QSA_V_BITS);
        if (lane == 0) {
            *(__global uint*)((__global uchar*)value_cache + po) = (uint)as_ushort(sh) | ((uint)as_ushort(zh) << 16);
        }
    }
#endif
}

#endif  // QSA_KV_WRITE_KCHANNEL
// [EVOLVE_END]
