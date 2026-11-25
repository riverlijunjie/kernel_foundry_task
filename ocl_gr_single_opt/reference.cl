// GatedResidual READ / MIX reference: the unmodified OpenVINO intel_gpu kernel ocl_v2/gr_read_dpas.cl (f16 weights, no compressed-weight
// branches), with its includes inlined so that the file is standalone.

// Standalone prelude: replaces include/batch_headers/{common,sub_group_block_read,sub_group_block_write}.cl of the OpenVINO GPU plugin
// (copied unchanged; the JIT plumbing is done by task.py).
#if defined(cl_khr_fp16)
#    pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
#ifdef cl_intel_subgroup_matrix_multiply_accumulate
#    pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#endif
#ifdef cl_intel_subgroup_2d_block_io
#    pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#endif

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE_PREFIX_uchar as_
#define AS_TYPE_PREFIX_char as_
#define AS_TYPE_PREFIX_fp4e2m1_t _as_
#define AS_TYPE_PREFIX_fp8e5m2_t _as_
#define AS_TYPE_PREFIX_fp8e4m3_t _as_
#define AS_TYPE_PREFIX_fp8e8m0_t _as_
#define AS_TYPE_PREFIX_ushort as_
#define AS_TYPE_PREFIX_short as_
#define AS_TYPE_PREFIX_half as_
#define AS_TYPE_PREFIX_int as_
#define AS_TYPE_PREFIX_uint as_
#define AS_TYPE_PREFIX_float as_
#define AS_TYPE_PREFIX_ulong as_
#define AS_TYPE_PREFIX_long as_

#define AS_TYPE_EXT(type, val, src_type) CAT(CAT(AS_TYPE_PREFIX_, src_type), type)(val)
#define AS_TYPE(type, val) CAT(as_, type)(val)

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
#define TYPE_SIZE_uchar  1
#define TYPE_SIZE_char   1
#define TYPE_SIZE_fp8e5m2_t 1
#define TYPE_SIZE_fp8e4m3_t 1
#define TYPE_SIZE_fp8e8m0_t 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short  2
#define TYPE_SIZE_half   2
#define TYPE_SIZE_int    4
#define TYPE_SIZE_uint   4
#define TYPE_SIZE_float  4
#define TYPE_SIZE_ulong  8
#define TYPE_SIZE_long   8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

#define FUNC(name) name
#define FUNC_CALL(name) name
#define KERNEL(name) __kernel void name

// ====================================================================================================================
// BLOCK_READN(type, vector_size, ptr, offset)
//    - evaluates to intel_sub_group_block_read operation for specified "type" and "vector size", reading
//      "vector_size" elements from memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_READN_SLM(type, vector_size, ptr, offset)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half, char, etc..
// vector_size [PP] - Number of elements to read/write, ex 2 for intel_sub_group_block_read2.
// ptr              - Pointer to global memory where to read from/write to.
// offset           - Additional offset added to ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type" elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_READ<n>(ptr, offset)
// Where:
//    <tensor> is one of: INPUT - referencing type jitted as INPUT0,
//                        BIAS,
//                        FILTER
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_READ_TYPE_size1 uchar
#define BLOCK_READ_TYPE_size2 ushort
#define BLOCK_READ_TYPE_size4 uint
#define BLOCK_READ_TYPE_size8 ulong
#define BLOCK_READ_TYPE(type_size) CAT(BLOCK_READ_TYPE_size, type_size)

#define BLOCK_READ_FUNC_size1       _sub_group_block_read_uc
#define BLOCK_READ_FUNC_size2       _sub_group_block_read_us
#define BLOCK_READ_FUNC_size4       _sub_group_block_read
#define BLOCK_READ_FUNC_size8       _sub_group_block_read_ul
#define BLOCK_READ_FUNC(type_size)  CAT(BLOCK_READ_FUNC_size, type_size)

// __local-pointer variants. cl_intel_subgroups* only defines __global overloads of
// intel_sub_group_block_read*; the __local overloads come from the separate
// cl_intel_subgroup_local_block_io extension, which NEO 23.x+ no longer declares
// implicitly. The _slm-suffixed family below routes to the intrinsic when the
// extension is advertised and falls back to an inline-emulation per-lane gather
// otherwise — see DECLARE_BLOCK_READ_SLM_EMULATION and the macro blocks at the
// bottom of this file.
#define BLOCK_READ_FUNC_SLM_size1       _sub_group_block_read_slm_uc
#define BLOCK_READ_FUNC_SLM_size2       _sub_group_block_read_slm_us
#define BLOCK_READ_FUNC_SLM_size4       _sub_group_block_read_slm
#define BLOCK_READ_FUNC_SLM_size8       _sub_group_block_read_slm_ul
#define BLOCK_READ_FUNC_SLM(type_size)  CAT(BLOCK_READ_FUNC_SLM_size, type_size)

#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size1(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_size2(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_size8(vector_size)                 BLOCK_READN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size)            CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)

#define BLOCK_READN_FUNC_SLM_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(BLOCK_READ_FUNC_SLM(type_size), vector_size)
#define BLOCK_READN_FUNC_SLM_size1(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_SLM_size2(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_SLM_size4(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_SLM_size8(vector_size)                 BLOCK_READN_FUNC_SLM_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC_SLM(type_size, vector_size)            CAT(BLOCK_READN_FUNC_SLM_size, type_size)(vector_size)

#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset)                                        \
    BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))

#define BLOCK_READN_RAW_SLM(type_size, vector_size, ptr, offset)                                                \
    BLOCK_READN_FUNC_SLM(type_size, vector_size)((const __local BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))

#define BLOCK_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))

#define BLOCK_READN_SLM(type, vector_size, ptr, offset)                                                         \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW_SLM(TYPE_SIZE(type), vector_size, ptr, offset))

#define DT_INPUT_BLOCK_READ(ptr, offset)            BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)           BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)          BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

#define DT_BIAS_BLOCK_READ(ptr, offset)             BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset)            BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset)            BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset)            BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset)           BLOCK_READN(BIAS_TYPE, 16, ptr, offset)

#define DT_FILTER_BLOCK_READ(ptr, offset)           BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)          BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)          BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)          BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)         BLOCK_READN(FILTER_TYPE, 16, ptr, offset)


#define BLOCK_READ_IMPL_1 ret = ptr[idx];

#define BLOCK_READ_IMPL_2                                   \
        ret.s0 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_4                                   \
        BLOCK_READ_IMPL_2                                   \
        ret.s2 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_8                                   \
        BLOCK_READ_IMPL_4                                   \
        ret.s4 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s5 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s6 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_16                                  \
        BLOCK_READ_IMPL_8                                   \
        ret.s8 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.s9 = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sa = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sb = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sc = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sd = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.se = ptr[idx]; idx += get_max_sub_group_size(); \
        ret.sf = ptr[idx]; idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL(vec_size) CAT(BLOCK_READ_IMPL_, vec_size)
#define BLOCK_READ_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_READ_EMULATION(type_size, vec_size) \
    inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME(type_size, vec_size)(const __global BLOCK_READ_TYPE(type_size)* ptr) { \
    uint idx = get_sub_group_local_id(); \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
    BLOCK_READ_IMPL(vec_size) \
    return ret; \
}

#define BLOCK_READ_FUNC_NAME_SLM(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC_SLM(type_size), vec_size)
#define DECLARE_BLOCK_READ_SLM_EMULATION(type_size, vec_size) \
    inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME_SLM(type_size, vec_size)(const __local BLOCK_READ_TYPE(type_size)* ptr) { \
    uint idx = get_sub_group_local_id(); \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
    BLOCK_READ_IMPL(vec_size) \
    return ret; \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_read(ptr) intel_sub_group_block_read(ptr)
    #define _sub_group_block_read2(ptr) intel_sub_group_block_read2(ptr)
    #define _sub_group_block_read4(ptr) intel_sub_group_block_read4(ptr)
    #define _sub_group_block_read8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(4, 1)
    DECLARE_BLOCK_READ_EMULATION(4, 2)
    DECLARE_BLOCK_READ_EMULATION(4, 4)
    DECLARE_BLOCK_READ_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm(ptr)  intel_sub_group_block_read(ptr)
    #define _sub_group_block_read_slm2(ptr) intel_sub_group_block_read2(ptr)
    #define _sub_group_block_read_slm4(ptr) intel_sub_group_block_read4(ptr)
    #define _sub_group_block_read_slm8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_read_us(ptr) intel_sub_group_block_read_us(ptr)
    #define _sub_group_block_read_us2(ptr) intel_sub_group_block_read_us2(ptr)
    #define _sub_group_block_read_us4(ptr) intel_sub_group_block_read_us4(ptr)
    #define _sub_group_block_read_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(2, 1)
    DECLARE_BLOCK_READ_EMULATION(2, 2)
    DECLARE_BLOCK_READ_EMULATION(2, 4)
    DECLARE_BLOCK_READ_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_us(ptr)  intel_sub_group_block_read_us(ptr)
    #define _sub_group_block_read_slm_us2(ptr) intel_sub_group_block_read_us2(ptr)
    #define _sub_group_block_read_slm_us4(ptr) intel_sub_group_block_read_us4(ptr)
    #define _sub_group_block_read_slm_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_read_uc(ptr) intel_sub_group_block_read_uc(ptr)
    #define _sub_group_block_read_uc2(ptr) intel_sub_group_block_read_uc2(ptr)
    #define _sub_group_block_read_uc4(ptr) intel_sub_group_block_read_uc4(ptr)
    #define _sub_group_block_read_uc8(ptr) intel_sub_group_block_read_uc8(ptr)
    #define _sub_group_block_read_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(1, 1)
    DECLARE_BLOCK_READ_EMULATION(1, 2)
    DECLARE_BLOCK_READ_EMULATION(1, 4)
    DECLARE_BLOCK_READ_EMULATION(1, 8)
    DECLARE_BLOCK_READ_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_uc(ptr)   intel_sub_group_block_read_uc(ptr)
    #define _sub_group_block_read_slm_uc2(ptr)  intel_sub_group_block_read_uc2(ptr)
    #define _sub_group_block_read_slm_uc4(ptr)  intel_sub_group_block_read_uc4(ptr)
    #define _sub_group_block_read_slm_uc8(ptr)  intel_sub_group_block_read_uc8(ptr)
    #define _sub_group_block_read_slm_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 8)
    DECLARE_BLOCK_READ_SLM_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_read_ul(ptr)  intel_sub_group_block_read_ul(ptr)
    #define _sub_group_block_read_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
    #define _sub_group_block_read_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
    #define _sub_group_block_read_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_EMULATION(8, 1)
    DECLARE_BLOCK_READ_EMULATION(8, 2)
    DECLARE_BLOCK_READ_EMULATION(8, 4)
    DECLARE_BLOCK_READ_EMULATION(8, 8)
#endif

#if defined(cl_intel_subgroup_local_block_io)
    #define _sub_group_block_read_slm_ul(ptr)  intel_sub_group_block_read_ul(ptr)
    #define _sub_group_block_read_slm_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
    #define _sub_group_block_read_slm_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
    #define _sub_group_block_read_slm_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 1)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 2)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 4)
    DECLARE_BLOCK_READ_SLM_EMULATION(8, 8)
#endif

// ====================================================================================================================
// BLOCK_WRITEN(type, vector_size, ptr, offset, val)
//    - evaluates to intel_sub_group_block_write operation for specified "type" and "vector size", writing
//      "vector_size"-element vector "val" to memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read/write functions please,
//  refer to cl_intel_subgroups extension documentation.
//
// BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half, char, etc..
// vector_size [PP] - Number of elements to read/write, ex 2 for intel_sub_group_block_read2.
// ptr              - Pointer to global memory where to read from/write to.
// offset           - Additional offset added to ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type" elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_WRITE<n>(ptr, offset, offset)
// Where:
//    <tensor> is usually OUTPUT,
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will be a scalar
//
// ====================================================================================================================

#define BLOCK_WRITE_TYPE_size1 uchar
#define BLOCK_WRITE_TYPE_size2 ushort
#define BLOCK_WRITE_TYPE_size4 uint
#define BLOCK_WRITE_TYPE_size8 ulong
#define BLOCK_WRITE_TYPE(type_size) CAT(BLOCK_WRITE_TYPE_size, type_size)

#define BLOCK_WRITE_FUNC_size1       _sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_size2       _sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4       _sub_group_block_write
#define BLOCK_WRITE_FUNC_size8       _sub_group_block_write_ul
#define BLOCK_WRITE_FUNC(type_size)  CAT(BLOCK_WRITE_FUNC_size, type_size)

#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size)  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC_size8(vector_size)                BLOCK_WRITEN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size)           CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)

#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val, src_type)                        \
    BLOCK_WRITEN_FUNC(type_size, vector_size)(                                                                  \
        (addr_space BLOCK_WRITE_TYPE(type_size)*)(ptr) + (offset),                                              \
        AS_TYPE_EXT(MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vector_size), val, src_type))

#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                                                       \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val, type)

#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)                                                   \
    BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val, type)

#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)     BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val)   BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)

#define BLOCK_WRITE_IMPL_1 out_ptr[idx] = v;
#define BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s0; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s1; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_4                                    \
        BLOCK_WRITE_IMPL_2                                    \
        out_ptr[idx] = v.s2; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s3; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_8                                    \
        BLOCK_WRITE_IMPL_4                                    \
        out_ptr[idx] = v.s4; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s5; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s6; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s7; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_16                                   \
        BLOCK_WRITE_IMPL_8                                    \
        out_ptr[idx] = v.s8; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.s9; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sa; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sb; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sc; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sd; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.se; idx += get_max_sub_group_size(); \
        out_ptr[idx] = v.sf; idx += get_max_sub_group_size();

#define BLOCK_WRITE_IMPL(vec_size) CAT(BLOCK_WRITE_IMPL_, vec_size)
#define BLOCK_WRITE_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_WRITE_EMULATION(type_size, vec_size) \
    inline void BLOCK_WRITE_FUNC_NAME(type_size, vec_size)(__global BLOCK_WRITE_TYPE(type_size)* out_ptr, \
                                                           MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vec_size) v) { \
    uint idx = get_sub_group_local_id(); \
    BLOCK_WRITE_IMPL(vec_size) \
}

#if defined(cl_intel_subgroups)
    #define _sub_group_block_write(ptr, v) intel_sub_group_block_write(ptr, v)
    #define _sub_group_block_write2(ptr, v) intel_sub_group_block_write2(ptr, v)
    #define _sub_group_block_write4(ptr, v) intel_sub_group_block_write4(ptr, v)
    #define _sub_group_block_write8(ptr, v) intel_sub_group_block_write8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(4, 1)
    DECLARE_BLOCK_WRITE_EMULATION(4, 2)
    DECLARE_BLOCK_WRITE_EMULATION(4, 4)
    DECLARE_BLOCK_WRITE_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
    #define _sub_group_block_write_us(ptr, v) intel_sub_group_block_write_us(ptr, v)
    #define _sub_group_block_write_us2(ptr, v) intel_sub_group_block_write_us2(ptr, v)
    #define _sub_group_block_write_us4(ptr, v) intel_sub_group_block_write_us4(ptr, v)
    #define _sub_group_block_write_us8(ptr, v) intel_sub_group_block_write_us8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(2, 1)
    DECLARE_BLOCK_WRITE_EMULATION(2, 2)
    DECLARE_BLOCK_WRITE_EMULATION(2, 4)
    DECLARE_BLOCK_WRITE_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
    #define _sub_group_block_write_uc(ptr, v) intel_sub_group_block_write_uc(ptr, v)
    #define _sub_group_block_write_uc2(ptr, v) intel_sub_group_block_write_uc2(ptr, v)
    #define _sub_group_block_write_uc4(ptr, v) intel_sub_group_block_write_uc4(ptr, v)
    #define _sub_group_block_write_uc8(ptr, v) intel_sub_group_block_write_uc8(ptr, v)
    #define _sub_group_block_write_uc16(ptr, v) intel_sub_group_block_write_uc16(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(1, 1)
    DECLARE_BLOCK_WRITE_EMULATION(1, 2)
    DECLARE_BLOCK_WRITE_EMULATION(1, 4)
    DECLARE_BLOCK_WRITE_EMULATION(1, 8)
    DECLARE_BLOCK_WRITE_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
    #define _sub_group_block_write_ul(ptr, v)  intel_sub_group_block_write_ul(ptr, v)
    #define _sub_group_block_write_ul2(ptr, v) intel_sub_group_block_write_ul2(ptr, v)
    #define _sub_group_block_write_ul4(ptr, v) intel_sub_group_block_write_ul4(ptr, v)
    #define _sub_group_block_write_ul8(ptr, v) intel_sub_group_block_write_ul8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
    DECLARE_BLOCK_WRITE_EMULATION(8, 1)
    DECLARE_BLOCK_WRITE_EMULATION(8, 2)
    DECLARE_BLOCK_WRITE_EMULATION(8, 4)
    DECLARE_BLOCK_WRITE_EMULATION(8, 8)
#endif

// [REFERENCE_START]
// [LAUNCH_START]
// How task.py compiles and launches this program (read from this comment block; the values below are the ones of the OpenVINO
// plugin for the f16 READ kernel). task.py always adds -DGR_BRANCHES, GR_HIDDEN, GR_LOWRANK, GR_EPS, GR_PRODUCE_WRITE_GATE and
// SUBGROUP_SIZE=16; the kernel may ignore or #undef them.
//   defines        extra -DNAME=VALUE macros, separated by spaces. GR_WG_SIZE (subgroups per work-group) and GR_RD_TM (token rows per
//                  work-group) also define the dispatch: gws = [GR_WG_SIZE * 16, ceil(rows / GR_RD_TM), 1], lws = [GR_WG_SIZE * 16, 1, 1].
//   grf_256        1 compiles with -cl-intel-256-GRF-per-thread.
//   build_options  extra -cl-* options, separated by spaces.
// The kernel entry point is gr_read_dpas; its arguments are fixed (see below).
//
// defines = GR_WG_SIZE=16 GR_RD_TM=32 GR_RD_NT=2 GR_NV_KD=8 GR_RD_SLM_GAIN=2
// grf_256 = 1
// build_options =
// [LAUNCH_END]

// ---- shared helpers of the GatedResidual kernels (gr_common.cl, f16-weight part) ----
// Shared helpers of the GatedResidual OpenCL kernels. One subgroup
// (16 lanes) is one Xe2 hardware thread; a row of 16*NV halves is held lane-interleaved: element e lives in component (e / 16)
// of lane (e % 16), which is exactly what intel_sub_group_block_read_us{1,2,4,8} returns. The header is inlined into every
// kernel separately (no include guard, helper functions go through FUNC).


#define GR_LOG2E 1.4426950408889634f

#define GR_BR_1 intel_sub_group_block_read_us
#define GR_BR_2 intel_sub_group_block_read_us2
#define GR_BR_4 intel_sub_group_block_read_us4
#define GR_BR_8 intel_sub_group_block_read_us8
#define GR_BW_1 intel_sub_group_block_write_us
#define GR_BW_2 intel_sub_group_block_write_us2
#define GR_BW_4 intel_sub_group_block_write_us4
#define GR_BW_8 intel_sub_group_block_write_us8
#define GR_CVF_1 convert_float
#define GR_CVF_2 convert_float2
#define GR_CVF_4 convert_float4
#define GR_CVF_8 convert_float8
#define GR_ASH_1 as_half
#define GR_ASH_2 as_half2
#define GR_ASH_4 as_half4
#define GR_ASH_8 as_half8
#define GR_CVH_1 convert_half_rte
#define GR_CVH_2 convert_half2_rte
#define GR_CVH_4 convert_half4_rte
#define GR_CVH_8 convert_half8_rte
#define GR_ASU_1 as_ushort
#define GR_ASU_2 as_ushort2
#define GR_ASU_4 as_ushort4
#define GR_ASU_8 as_ushort8

#define GR_UT(nv) MAKE_VECTOR_TYPE(ushort, nv)
#define GR_FT(nv) MAKE_VECTOR_TYPE(float, nv)

// ushort block of 16*nv halves -> float vector, and back (round to nearest even)
#define GR_LDF(nv, p) CAT(GR_CVF_, nv)(CAT(GR_ASH_, nv)(CAT(GR_BR_, nv)(p)))
#define GR_BRU(nv, p) CAT(GR_BR_, nv)(p)
#define GR_STF(nv, p, f) CAT(GR_BW_, nv)(p, CAT(GR_ASU_, nv)(CAT(GR_CVH_, nv)(f)))
#define GR_U2F(nv, u) CAT(GR_CVF_, nv)(CAT(GR_ASH_, nv)(u))
#define GR_F2U(nv, f) CAT(GR_ASU_, nv)(CAT(GR_CVH_, nv)(f))

// sequential per-lane accumulation over the components
#define GR_ACC_1(s, v) { s += (v); }
#define GR_ACC_2(s, v) { s += (v).s0; s += (v).s1; }
#define GR_ACC_4(s, v) { s += (v).s0; s += (v).s1; s += (v).s2; s += (v).s3; }
#define GR_ACC_8(s, v) { s += (v).s0; s += (v).s1; s += (v).s2; s += (v).s3; s += (v).s4; s += (v).s5; s += (v).s6; s += (v).s7; }
#define GR_ACC(nv, s, v) CAT(GR_ACC_, nv)(s, v)

// s += a . b over the components
#define GR_DOT_1(s, a, b) { s += (a) * (b); }
#define GR_DOT_2(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; }
#define GR_DOT_4(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; s += (a).s2 * (b).s2; s += (a).s3 * (b).s3; }
#define GR_DOT_8(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; s += (a).s2 * (b).s2; s += (a).s3 * (b).s3; s += (a).s4 * (b).s4; s += (a).s5 * (b).s5; s += (a).s6 * (b).s6; s += (a).s7 * (b).s7; }
#define GR_DOT(nv, s, a, b) CAT(GR_DOT_, nv)(s, a, b)

// lane-interleaved SLM access of 16*nv ushorts
#define GR_SLMST_1(p, v, l) { (p)[l] = (v); }
#define GR_SLMST_2(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; }
#define GR_SLMST_4(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; (p)[32 + l] = (v).s2; (p)[48 + l] = (v).s3; }
#define GR_SLMST_8(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; (p)[32 + l] = (v).s2; (p)[48 + l] = (v).s3; (p)[64 + l] = (v).s4; (p)[80 + l] = (v).s5; (p)[96 + l] = (v).s6; (p)[112 + l] = (v).s7; }
#define GR_SLMST(nv, p, v, l) CAT(GR_SLMST_, nv)(p, v, l)
#define GR_SLMLD_1(p, l) ((p)[l])
#define GR_SLMLD_2(p, l) ((ushort2)((p)[l], (p)[16 + l]))
#define GR_SLMLD_4(p, l) ((ushort4)((p)[l], (p)[16 + l], (p)[32 + l], (p)[48 + l]))
#define GR_SLMLD_8(p, l) ((ushort8)((p)[l], (p)[16 + l], (p)[32 + l], (p)[48 + l], (p)[64 + l], (p)[80 + l], (p)[96 + l], (p)[112 + l]))
#define GR_SLMLD(nv, p, l) CAT(GR_SLMLD_, nv)(p, l)

#define GR_SIGMOID(x) native_recip(1.0f + native_exp2(-(x) * GR_LOG2E))

#define GR_MAD intel_sub_group_f16_f16_matrix_mad_k16

// 2D block reads of a row-major f16 matrix (surface: width/pitch in bytes, height in rows; rows past the height read 0).
// GR_RD2D(rows, ...): rows x 32 halves (two K16 slices) -> dst[s * rows + r] = M[y + r][x + 16 * s + lane]
// GR_RD2D1(rows, ...): rows x 16 halves                  -> dst[r]            = M[y + r][x + lane]
#define GR_RD2D_8 intel_sub_group_2d_block_read_16b_8r16x2c
#define GR_RD2D_16 intel_sub_group_2d_block_read_16b_16r16x2c
#define GR_RD2D_32 intel_sub_group_2d_block_read_16b_32r16x2c
#define GR_RD2D(rows, base, width, height, pitch, x, y, dst) CAT(GR_RD2D_, rows)((__global void*)(base), (width), (height), (pitch), (int2)((x), (y)), (dst))
#define GR_RD2D1_8 intel_sub_group_2d_block_read_16b_8r16x1c
#define GR_RD2D1(rows, base, width, height, pitch, x, y, dst) CAT(GR_RD2D1_, rows)((__global void*)(base), (width), (height), (pitch), (int2)((x), (y)), (dst))

// SLM block write; the __local overload of intel_sub_group_block_write needs cl_intel_subgroup_local_block_io, otherwise each lane stores
// its own column
#if defined(cl_intel_subgroup_local_block_io)
    #define GR_BLOCK_WRITEN_SLM(type, nv, ptr, offset, val) BLOCK_WRITEN_SLM(type, nv, ptr, offset, val)
#else
    #define GR_BLOCK_WRITEN_SLM(type, nv, ptr, offset, val) GR_SLMST(nv, (__local type*)(ptr) + (offset), val, get_sub_group_local_id())
#endif

// ---- gr_read_dpas.cl (f16 weights; the compressed-weight u4 / u8 variant is not part of this task) ----
// GatedResidual READ / MIX as ONE systolic (dpas) kernel. One work-group owns GR_RD_TM token rows:
//   phase 1  inv_rms[row, c] of the branch-wise RMSNorm                       -> SLM
//   phase 2  t = SiLU(Rbar @ W_down^T / C) (+ write gate S as an extra tile)  -> SLM (dpas A-operand slots)
//   phase 3  mixed = (1/C) sum_c sigmoid(t @ W_up^T)[c] * Rbar[c]             -> global
// Rbar is never materialized: phase 2 normalizes each residual A tile in f32 (inv_rms from SLM, then the gain) and rounds it
// once to half, as gr_rbar does; phase 3 recomputes Rbar in f32. The accumulator layout of a dpas (token i in
// component i, column n in lane n) is already the A-operand layout of the next one, so t only needs an SLM slot
// of 8 tokens x 16 low-rank values per (token tile, K16 block).
// The residual tiles come through 2D block reads (a whole TM x K32 A tile per message, rows past the end read 0); W rows
// are K-contiguous, so every lane reads its own row with one 64 B vload16.
//
// JIT: GR_BRANCHES, GR_HIDDEN (multiple of 32), GR_LOWRANK (multiple of 32), GR_EPS,
//      GR_PRODUCE_WRITE_GATE, GR_WG_SIZE (subgroups), GR_RD_TM (8/16/32), GR_RD_NT, GR_NV_KD (phase 1 chunk halves / 16),
//      GR_RD_SLM_GAIN (0 = gain from global, 1 = padded SLM copy, 2 = duplicated SLM copy), GR_P3_GB (optional: branches per
//      phase-3 K loop).
// Dispatch: gws = [WG * 16, ceil(rows / TM), 1], lws = [WG * 16, 1, 1].

#define GR_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_MT (GR_RD_TM / 8)
#define GR_KB (GR_LOWRANK / 16)
#define GR_TASKS (GR_KB / GR_RD_NT + (GR_PRODUCE_WRITE_GATE ? 1 : 0))
#define GR_KD (GR_NV_KD * 16)
#define GR_INV_C (1.0f / (float)GR_BRANCHES)

#if GR_HIDDEN % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KB % GR_RD_NT != 0
    #error "gr_read_dpas.cl: hidden size and low rank must be multiples of 32 and GR_RD_NT must divide L / 16"
#endif
#if GR_BRANCHES % 2 != 0 || GR_BRANCHES > 16
    #error "gr_read_dpas.cl: the write gate is one N16 tile stored in half pairs"
#endif

// The gain of a K16 slice is one register half: A rows alternate between the low and high half of a GRF, and IGC realigns
// (mov + mul + mov) whenever the gain sits at the other offset. The gain is therefore staged in SLM so that every gain
// half is also read at the other offset: duplicated (GS = (g0, g0, g1, g1)) or behind a 16-half pad (GE = (g0, g1),
// GS = (g-1, g0, g1, g2)).
#if GR_RD_SLM_GAIN == 2
    #define GR_SLM_G_SIZE (2 * GR_CD)
    #define GR_P2_GAIN(GE, GS, KK) { GS = BLOCK_READN_SLM(ushort, 4, slm_g + 2 * (KK), 0); }
    #define GR_G0E(GE, GS) as_half((GS).s0)
    #define GR_G0O(GE, GS) as_half((GS).s1)
    #define GR_G1E(GE, GS) as_half((GS).s2)
    #define GR_G1O(GE, GS) as_half((GS).s3)
#elif GR_RD_SLM_GAIN
    #define GR_SLM_G_SIZE (GR_CD + 48)
    #define GR_P2_GAIN(GE, GS, KK) { GE = BLOCK_READN_SLM(ushort, 2, slm_g + 16 + (KK), 0); GS = BLOCK_READN_SLM(ushort, 4, slm_g + (KK), 0); }
    #define GR_G0E(GE, GS) as_half((GE).s0)
    #define GR_G0O(GE, GS) as_half((GS).s1)
    #define GR_G1E(GE, GS) as_half((GS).s2)
    #define GR_G1O(GE, GS) as_half((GE).s1)
#else
    #define GR_P2_GAIN(GE, GS, KK) { GE = intel_sub_group_block_read_us2(norm_weight + (KK)); }
    #define GR_G0E(GE, GS) as_half((GE).s0)
    #define GR_G0O(GE, GS) as_half((GE).s0)
    #define GR_G1E(GE, GS) as_half((GE).s1)
    #define GR_G1O(GE, GS) as_half((GE).s1)
#endif

// One K32 step of phase 2: loads of the A tile (raw residual), the gain and the W rows of every column tile ...
#define GR_P2_LDB(dst, j, KK) dst = vload16(0, (const __global uint*)(brow[j] + (KK)))

#define GR_P2_LOAD(AT, GE, GS, B, KK) { GR_RD2D(GR_RD_TM, residual, GR_CD * 2, num_rows, GR_CD * 2, (KK), t0, AT); GR_P2_GAIN(GE, GS, KK); unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) { GR_P2_LDB(B[j_], j_, KK); } }

// Rbar of 8 token rows (component i = token i, lane = k) in f32, rounded once: the gr_rbar formula, so the A operand stays
// finite whenever Rbar does (|R * inv_rms| <= sqrt(D), whereas R * gain alone can overflow half).
#define GR_P2_SCALE(v, inv, g) as_short8(convert_half8_rte((convert_float8(as_half8(v)) * (inv)) * (g)))

// ... and its dpas: A = Rbar of branch c (inv_rms of its rows in invv), two K16 slices, slice 0 first
#define GR_P2_MMA(AT, GE, GS, B)                                                                                                   \
    {                                                                                                                              \
        const float g0_ = convert_float(GR_G0E(GE, GS));                                                                           \
        const float g1_ = convert_float(GR_G1E(GE, GS));                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                                               \
            const short8 av_ = GR_P2_SCALE(vload8(m_, (AT)), invv[m_], g0_);                                                       \
            unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {                                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].lo), acc[j_ * GR_MT + m_]);                                     \
            }                                                                                                                      \
        }                                                                                                                          \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                                               \
            const short8 av_ = GR_P2_SCALE(vload8(GR_MT + m_, (AT)), invv[m_], g1_);                                               \
            unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {                                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].hi), acc[j_ * GR_MT + m_]);                                     \
            }                                                                                                                      \
        }                                                                                                                          \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_read_dpas)(const __global ushort* residual,
                     const __global ushort* norm_weight,
                     const __global ushort* w_down,
                     const __global ushort* w_up,
#if GR_PRODUCE_WRITE_GATE
                     const __global ushort* w_inject,
#endif
                     __global ushort* mixed,
#if GR_PRODUCE_WRITE_GATE
                     __global ushort* write_gate,
#endif
                     uint num_rows) {
    __local float slm_inv[GR_BRANCHES * GR_RD_TM] __attribute__((aligned(32)));
    __local ushort slm_t[GR_MT * GR_KB * 128] __attribute__((aligned(16)));
#if GR_RD_SLM_GAIN
    __local ushort slm_g[GR_SLM_G_SIZE] __attribute__((aligned(64)));
#endif

    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    const uint t0 = get_group_id(1) * GR_RD_TM;
    if (t0 >= num_rows) {
        return;
    }

#if GR_RD_SLM_GAIN
    {
        // All chunk loads are issued before the SLM stores so the copy costs one load latency.
#if GR_CD % 128 == 0
    #define GR_GNV 8
#else
    #define GR_GNV 2
#endif
#define GR_GIT ((GR_CD + GR_WG_SIZE * GR_GNV * 16 - 1) / (GR_WG_SIZE * GR_GNV * 16))
        GR_UT(GR_GNV) gv[GR_GIT];
#pragma unroll
        for (uint it = 0; it < GR_GIT; it++) {
            gv[it] = GR_BRU(GR_GNV, norm_weight + min((it * GR_WG_SIZE + sg) * GR_GNV * 16, (uint)(GR_CD - GR_GNV * 16)));
        }
#pragma unroll
        for (uint it = 0; it < GR_GIT; it++) {
            const uint i = (it * GR_WG_SIZE + sg) * GR_GNV * 16;
            if (i < GR_CD) {
                const GR_UT(GR_GNV) g = gv[it];
#if GR_RD_SLM_GAIN == 2
                ushort gq[GR_GNV];
                CAT(vstore, GR_GNV)(g, 0, gq);
#pragma unroll
                for (uint q = 0; q < GR_GNV; q++) {
                    GR_BLOCK_WRITEN_SLM(ushort, 2, slm_g + 2 * (i + 16 * q), 0, (ushort2)(gq[q], gq[q]));
                }
#else
                GR_BLOCK_WRITEN_SLM(ushort, GR_GNV, slm_g + 16 + i, 0, g);
#endif
            }
        }
    }
#endif

    // Phase 1: inv_rms[c][row] for 4 rows per task. Rows past num_rows re-read the last row and get 0.
    for (uint p = sg; p < GR_BRANCHES * (GR_RD_TM / 4); p += GR_WG_SIZE) {
        const uint c = p / (GR_RD_TM / 4);
        const uint r0 = (p % (GR_RD_TM / 4)) * 4;
        size_t off[4];
        float sq[4];
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            off[r] = (size_t)min(t0 + r0 + r, num_rows - 1) * GR_CD + c * GR_HIDDEN;
            sq[r] = 0.0f;
        }
        // The next chunk of the 4 rows is in flight while the current one is reduced (the last step re-reads the last chunk).
        GR_UT(GR_NV_KD) x[4];
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            x[r] = GR_BRU(GR_NV_KD, residual + off[r]);
        }
        for (uint d = 0; d < GR_HIDDEN; d += GR_KD) {
            const uint dn = min(d + GR_KD, (uint)(GR_HIDDEN - GR_KD));
            GR_UT(GR_NV_KD) xn[4];
#pragma unroll
            for (uint r = 0; r < 4; r++) {
                xn[r] = GR_BRU(GR_NV_KD, residual + off[r] + dn);
            }
#pragma unroll
            for (uint r = 0; r < 4; r++) {
                const GR_FT(GR_NV_KD) f = GR_U2F(GR_NV_KD, x[r]);
                const GR_FT(GR_NV_KD) q = f * f;
                GR_ACC(GR_NV_KD, sq[r], q);
                x[r] = xn[r];
            }
        }
        float4 inv;
        inv.s0 = t0 + r0 + 0 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[0]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s1 = t0 + r0 + 1 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[1]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s2 = t0 + r0 + 2 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[2]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s3 = t0 + r0 + 3 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[3]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        if (lane == 0) {
            vstore4(inv, 0, slm_inv + c * GR_RD_TM + r0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: down projection, GR_RD_NT low-rank tiles (or the write-gate tile) per task.
    //   acc += half((R[:, c] * inv_rms[:, c]) * gain[c]) @ W[:, c]^T   per branch c
    for (uint task = sg; task < GR_TASKS; task += GR_WG_SIZE) {
#if GR_PRODUCE_WRITE_GATE
        const bool is_wgate = task == GR_TASKS - 1;
        const __global ushort* w = is_wgate ? w_inject : w_down;
        const uint n0 = is_wgate ? 0 : task * GR_RD_NT * 16;
#else
        const __global ushort* w = w_down;
        const uint n0 = task * GR_RD_NT * 16;
#endif
        const __global ushort* brow[GR_RD_NT];
#pragma unroll
        for (uint j = 0; j < GR_RD_NT; j++) {
#if GR_PRODUCE_WRITE_GATE
            const uint row = is_wgate ? min(lane, (uint)(GR_BRANCHES - 1)) : n0 + 16 * j + lane;
#else
            const uint row = n0 + 16 * j + lane;
#endif
            brow[j] = w + (size_t)row * GR_CD;
        }

        float8 acc[GR_RD_NT * GR_MT];
#pragma unroll
        for (uint i = 0; i < GR_RD_NT * GR_MT; i++) {
            acc[i] = (float8)(0.0f);
        }
        for (uint c = 0; c < GR_BRANCHES; c++) {
            float8 invv[GR_MT];
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                invv[m] = vload8(0, slm_inv + c * GR_RD_TM + m * 8);
            }
            const uint kbeg = c * GR_HIDDEN;
            const uint kend = kbeg + GR_HIDDEN;
            ushort at0[2 * GR_RD_TM];
            ushort2 ge0;
            ushort4 gs0;
            uint16 b0[GR_RD_NT];
            for (uint k = kbeg; k < kend; k += 32) {
                GR_P2_LOAD(at0, ge0, gs0, b0, k);
                GR_P2_MMA(at0, ge0, gs0, b0);
            }
        }

        const float inv_c = GR_INV_C;
#if GR_PRODUCE_WRITE_GATE
        if (is_wgate) {
            if (lane < GR_BRANCHES) {
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    // Write-gate range is (0, 2) - it must not be clamped to (0, 1).
                    const float8 s = 2.0f * GR_SIGMOID(acc[m] * inv_c);
                    float tmp[8];
                    vstore8(s, 0, tmp);
#pragma unroll
                    for (uint i = 0; i < 8; i++) {
                        const uint tok = t0 + m * 8 + i;
                        if (tok < num_rows) {
                            write_gate[(size_t)tok * GR_BRANCHES + lane] = as_ushort(convert_half_rte(tmp[i]));
                        }
                    }
                }
            }
            continue;
        }
#endif
#pragma unroll
        for (uint j = 0; j < GR_RD_NT; j++) {
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                const float8 x = acc[j * GR_MT + m] * inv_c;
                const ushort8 tu = as_ushort8(convert_half8_rte(x * GR_SIGMOID(x)));
                GR_BLOCK_WRITEN_SLM(ushort, 8, slm_t + (m * GR_KB + n0 / 16 + j) * 128, 0, tu);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: up projection + sigmoid read gate + branch mix, 16 channels per pass. Up to 4 branches run in one K loop so
    // each SLM read of t feeds several dpas; the mix still adds the branches in order 0..C-1.
#ifndef GR_P3_GB
#define GR_P3_GB (GR_BRANCHES % 4 == 0 ? 4 : 2)
#endif
    for (uint dt = sg; dt < GR_HIDDEN / 16; dt += GR_WG_SIZE) {
        const uint d0 = dt * 16;
        float8 mix[GR_MT];
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            mix[m] = (float8)(0.0f);
        }
        for (uint cg = 0; cg < GR_BRANCHES; cg += GR_P3_GB) {
            float8 acc[GR_P3_GB * GR_MT];
#pragma unroll
            for (uint i = 0; i < GR_P3_GB * GR_MT; i++) {
                acc[i] = (float8)(0.0f);
            }
            const __global ushort* wrow[GR_P3_GB];
#pragma unroll
            for (uint g = 0; g < GR_P3_GB; g++) {
                wrow[g] = w_up + (size_t)((cg + g) * GR_HIDDEN + d0 + lane) * GR_LOWRANK;
            }
            for (uint kb = 0; kb < GR_KB; kb += 2) {
                uint16 b[GR_P3_GB];
#pragma unroll
                for (uint g = 0; g < GR_P3_GB; g++) {
                    b[g] = vload16(0, (const __global uint*)(wrow[g] + kb * 16));
                }
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    const short8 a0 = as_short8(BLOCK_READN_SLM(ushort, 8, slm_t + (m * GR_KB + kb) * 128, 0));
                    const short8 a1 = as_short8(BLOCK_READN_SLM(ushort, 8, slm_t + (m * GR_KB + kb + 1) * 128, 0));
#pragma unroll
                    for (uint g = 0; g < GR_P3_GB; g++) {
                        acc[g * GR_MT + m] = GR_MAD(a0, as_int8(b[g].lo), acc[g * GR_MT + m]);
                        acc[g * GR_MT + m] = GR_MAD(a1, as_int8(b[g].hi), acc[g * GR_MT + m]);
                    }
                }
            }
#pragma unroll
            for (uint g = 0; g < GR_P3_GB; g++) {
                const uint c = cg + g;
                ushort rt[GR_MT][8];
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    GR_RD2D1(8, residual, GR_CD * 2, num_rows, GR_CD * 2, c * GR_HIDDEN + d0, t0 + m * 8, rt[m]);
                }
                const float gain = convert_float(as_half(intel_sub_group_block_read_us(norm_weight + c * GR_HIDDEN + d0)));
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    const float8 invv = vload8(0, slm_inv + c * GR_RD_TM + m * 8);
                    const float8 rbar = (convert_float8(as_half8(vload8(0, rt[m]))) * gain) * invv;
                    mix[m] += GR_SIGMOID(acc[g * GR_MT + m]) * rbar;
                }
            }
        }
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            const ushort8 ou = as_ushort8(convert_half8_rte(mix[m] * GR_INV_C));
            ushort tmp[8];
            vstore8(ou, 0, tmp);
#pragma unroll
            for (uint i = 0; i < 8; i++) {
                const uint tok = t0 + m * 8 + i;
                if (tok < num_rows) {
                    intel_sub_group_block_write_us(mixed + (size_t)tok * GR_HIDDEN + d0, tmp[i]);
                }
            }
        }
    }
}
// [REFERENCE_END]
