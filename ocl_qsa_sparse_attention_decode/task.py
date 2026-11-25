"""KernelFoundry task: decode attention (split-K GQA systolic attention + partition merge) of the OpenVINO intel_gpu SparsePA primitive.

The kernels (kernel.cl / reference.cl) are extracted from sparse_attention_gqa_dpas.cl (QSA_GQA_SPLIT, paged addressing)
and sparse_attention_split_reduce.cl. For the one new token of each sequence they attend to the top-k selected blocks of a
paged f16 / u8 / u4 KV cache plus the incomplete tail block. Model geometry comes from the Qwen4-exp config.json: 24 query
heads, 2 kv heads, head_dim 256, 12 full-attention layers, indexer_compress_ratio 4 (block size) and indexer_budget 2048
tokens (512 selected blocks). The page size (16) is the one used by the SparsePA unit tests. The caches are pre-filled by
the host (the KV write stage is a separate kernel).
"""

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl
import pytest
from kernelfoundry import TestBase  # Tests must derive from TestBase

# Fixtures provided via conftest.py from kernelfoundry: use_reference, ocl_queue, profile_store, request

HERE = Path(__file__).parent
KERNEL_FILE = "kernel.cl"
REFERENCE_FILE = "reference.cl"

# ===================== Workload geometry (config.json) =====================
HEAD_DIM = 256  # text_config.head_dim, used for both K and V
NUM_HEADS = 24  # text_config.num_attention_heads
NUM_KV_HEADS = 2  # text_config.num_key_value_heads
BLOCK_SIZE = 4  # text_config.indexer_compress_ratio: tokens per selectable block
TOPK = 2048 // BLOCK_SIZE  # text_config.indexer_budget / block size: selected blocks per token
PAGE_SIZE = 16  # tokens per physical cache page
SUBGROUP_SIZE = 16  # Xe2 (ptl / lnl / bmg)
WORKERS = 4  # subgroups per work-group (one token x one KV head x one partition)
SPLIT_PARTITIONS = 64  # partitions of the block steps of a token (gqa_split_partitions of the plugin)
REDUCE_COLS = 32  # output columns merged per work-group of the reduce kernel (Dv % 32 == 0)
NUM_FULL_ATTN_LAYERS = 12  # text_config.layer_types.count("full_attention"): one attention call per layer and decode step
SCALE = 1.0 / np.sqrt(HEAD_DIM)
Q_PITCH = NUM_HEADS * HEAD_DIM  # elements per token of the query / output tensors
MIN_SCALE = 1.0 / 16384.0

CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

U32 = np.uint32


# ===================== Cache types =====================
@dataclass(frozen=True)
class CacheQuant:
    mode: int  # 0 = f16, 1 = BY_TOKEN, 2 = BY_CHANNEL
    bits: int = 16  # 8 or 4 when quantized
    sym: bool = False
    group: int = 0  # tokens per parameter group (BY_CHANNEL only)

    @property
    def row_bytes(self) -> int:
        return HEAD_DIM * self.bits // 8

    @property
    def param_bytes(self) -> int:
        if self.mode == 0:
            return 0
        if self.mode == 1:
            return PAGE_SIZE * 4  # [scale f16][zp f16] per token
        return (PAGE_SIZE // self.group) * HEAD_DIM * (1 if self.sym else 2) * 2

    @property
    def page_bytes(self) -> int:
        return (PAGE_SIZE + -(-self.param_bytes // self.row_bytes)) * self.row_bytes

    @property
    def param_offset(self) -> int:
        return PAGE_SIZE * self.row_bytes

    @property
    def zero_sym(self) -> float:
        return 128.0 if self.bits == 8 else 8.0


@dataclass(frozen=True)
class KvConfig:
    name: str
    key: CacheQuant
    value: CacheQuant


F16 = KvConfig("f16", CacheQuant(0), CacheQuant(0))
# Plugin layout for a quantized kv_cache_precision: key BY_CHANNEL (group 16), value BY_TOKEN, asymmetric.
U8 = KvConfig("u8", CacheQuant(2, 8, False, 16), CacheQuant(1, 8, False))
U4 = KvConfig("u4", CacheQuant(2, 4, False, 16), CacheQuant(1, 4, False))
BENCH_CONFIGS = [F16, U8, U4]
ALL_CONFIGS = BENCH_CONFIGS + [
    KvConfig("u8_sym", CacheQuant(2, 8, True, 16), CacheQuant(1, 8, True)),
    KvConfig("tok_u4_sym", CacheQuant(1, 4, True), CacheQuant(1, 4, True)),
]


def _defines(cfg: KvConfig) -> dict:
    d = {
        "SUBGROUP_SIZE": SUBGROUP_SIZE,
        "QSA_HEADS": NUM_HEADS,
        "QSA_KV_HEADS": NUM_KV_HEADS,
        "QSA_K_HEAD_SIZE": HEAD_DIM,
        "QSA_V_HEAD_SIZE": HEAD_DIM,
        "QSA_COMPRESS_RATIO": BLOCK_SIZE,
        "QSA_BLOCK_TOPK": TOPK,
        "QSA_SCALE": f"{SCALE:.9g}f",
        "QSA_IS_CAUSAL": 1,
        "QSA_PAGE_SIZE": PAGE_SIZE,
        "QSA_SPLIT_PARTITIONS": SPLIT_PARTITIONS,
    }
    for name, q in (("K", cfg.key), ("V", cfg.value)):
        d[f"QSA_{name}_MODE"] = q.mode
        d[f"QSA_{name}_BITS"] = q.bits
        d[f"QSA_{name}_SYM"] = int(q.sym)
        d[f"QSA_{name}_GROUP"] = max(1, q.group)
        d[f"QSA_{name}_ROW_BYTES"] = q.row_bytes
        d[f"QSA_{name}_PAGE_BYTES"] = q.page_bytes
        d[f"QSA_{name}_PARAM_OFFSET"] = q.param_offset
    return d


def _reduce_defines() -> dict:
    return {
        "SUBGROUP_SIZE": SUBGROUP_SIZE,
        "QSA_HEADS": NUM_HEADS,
        "QSA_V_HEAD_SIZE": HEAD_DIM,
        "QSA_SPLIT_PARTITIONS": SPLIT_PARTITIONS,
        "QSA_REDUCE_COLS": REDUCE_COLS,
        "QSA_DECODE_REDUCE": 1,
    }


def _build_program(ctx: cl.Context, source: str, defines: dict, large_grf: bool) -> cl.Program:
    options = ["-cl-mad-enable"] + [f"-D{k}={v}" for k, v in defines.items()]
    if large_grf:
        options.append("-cl-intel-256-GRF-per-thread")
    last_error = None
    for std in ("-cl-std=CL3.0", "-cl-std=CL2.0"):  # same selection as the plugin's KernelGenerator
        try:
            return cl.Program(ctx, source).build(options=options + [std])
        except cl.Error as e:
            last_error = e
    raise last_error


# ===================== Kernel launcher =====================
@dataclass
class AttentionInputs:
    num_tokens: int
    num_seqs: int
    query: cl.Buffer
    key_cache: cl.Buffer
    value_cache: cl.Buffer
    past_lens: cl.Buffer
    subsequence_begins: cl.Buffer
    block_indices: cl.Buffer
    block_indices_begins: cl.Buffer
    selected_blocks: cl.Buffer
    selected_counts: cl.Buffer
    partial_acc: cl.Buffer
    partial_ml: cl.Buffer
    output: cl.Buffer


class AttentionOp:
    """Launches the decode stages like the plugin does: split attention (token x kv head x partition), then the merge."""

    def __init__(self, queue: cl.CommandQueue, filename: str, cfg: KvConfig):
        source = (HERE / filename).read_text()
        self.queue = queue
        self.cfg = cfg
        self._split = _build_program(queue.context, source, _defines(cfg), large_grf=True).sparse_attention_paged_gqa_dpas_split
        self._reduce = _build_program(queue.context, source, _reduce_defines(), large_grf=False).sparse_attention_paged_split_reduce

    def __call__(self, layers: list[AttentionInputs]):
        event = None
        for inp in layers:
            tokens = max(1, inp.num_tokens)
            self._split(
                self.queue,
                (tokens * WORKERS * SUBGROUP_SIZE, NUM_KV_HEADS, SPLIT_PARTITIONS),
                (WORKERS * SUBGROUP_SIZE, 1, 1),
                inp.query,
                inp.key_cache,
                inp.value_cache,
                inp.past_lens,
                inp.subsequence_begins,
                inp.block_indices,
                inp.block_indices_begins,
                inp.selected_blocks,
                inp.selected_counts,
                inp.partial_acc,
                inp.partial_ml,
                U32(inp.num_tokens),
                U32(inp.num_seqs),
                U32(0),
                U32(Q_PITCH),
            )
            event = self._reduce(
                self.queue,
                (NUM_HEADS * SUBGROUP_SIZE, tokens, HEAD_DIM // REDUCE_COLS),
                (SUBGROUP_SIZE, 1, 1),
                inp.partial_acc,
                inp.partial_ml,
                inp.output,
                U32(inp.num_tokens),
            )
        return event


_OPS: dict = {}


def build_op(queue: cl.CommandQueue, filename: str, cfg: KvConfig) -> AttentionOp:
    key = (filename, cfg.name, queue.int_ptr)
    if key not in _OPS:
        _OPS[key] = AttentionOp(queue, filename, cfg)
    return _OPS[key]


# ===================== Cache packing (host model of the quantized page format) =====================
def pack_cache(rows: np.ndarray, q: CacheQuant) -> np.ndarray:
    """f16 rows [pages * PAGE_SIZE, Hkv, D] in logical order -> u8 pages [pages, Hkv, page_bytes], quantized like the kv write kernels."""
    pages = rows.shape[0] // PAGE_SIZE
    x = rows.reshape(pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [pages, Hkv, PAGE, D]
    out = np.zeros((pages, NUM_KV_HEADS, q.page_bytes), np.uint8)
    if q.mode == 0:
        out[:, :, : PAGE_SIZE * q.row_bytes] = np.ascontiguousarray(x).view(np.uint8).reshape(pages, NUM_KV_HEADS, -1)
        return out

    qmax = 255.0 if q.bits == 8 else 15.0
    axis = -1 if q.mode == 1 else -2
    view = x.astype(np.float32)
    if q.mode == 2:
        view = view.reshape(pages, NUM_KV_HEADS, PAGE_SIZE // q.group, q.group, HEAD_DIM)
    mn = np.minimum(view.min(axis=axis, keepdims=True), 0.0)  # the range always includes 0
    mx = np.maximum(view.max(axis=axis, keepdims=True), 0.0)
    if q.sym:
        scale = np.maximum(np.maximum(mx, -mn) / (127.0 if q.bits == 8 else 7.0), MIN_SCALE)
        zp = np.full_like(scale, q.zero_sym)
    else:
        scale = np.maximum((mx - mn) / qmax, MIN_SCALE)
        zp = -mn / scale
    scale_h, zp_h = scale.astype(np.float16), zp.astype(np.float16)
    inv = 1.0 / scale_h.astype(np.float32)
    codes = np.floor(np.clip(view * inv + zp_h.astype(np.float32), 0.0, qmax) + 0.5).astype(np.uint8)
    codes = codes.reshape(pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)

    data = codes if q.bits == 8 else (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    out[:, :, : PAGE_SIZE * q.row_bytes] = data.reshape(pages, NUM_KV_HEADS, -1)
    if q.mode == 1:
        params = np.stack([scale_h[..., 0], zp_h[..., 0]], axis=-1)  # [pages, Hkv, PAGE, 2]
    else:
        parts = [scale_h[..., 0, :]] + ([] if q.sym else [zp_h[..., 0, :]])  # each [pages, Hkv, groups, D]
        params = np.stack(parts, axis=3)  # [pages, Hkv, groups, 1 | 2, D]
    out[:, :, q.param_offset : q.param_offset + q.param_bytes] = np.ascontiguousarray(params).view(np.uint8).reshape(pages, NUM_KV_HEADS, -1)
    return out


def decode_cache(pages: np.ndarray, q: CacheQuant) -> np.ndarray:
    """u8 pages [pages, Hkv, page_bytes] -> f32 rows [pages * PAGE_SIZE, Hkv, D], dequantized in the kernels' f16 arithmetic."""
    n = pages.shape[0]
    data = pages[:, :, : PAGE_SIZE * q.row_bytes]
    if q.mode == 0:
        deq = data.copy().view(np.float16).reshape(n, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
    else:
        data = data.reshape(n, NUM_KV_HEADS, PAGE_SIZE, q.row_bytes)
        if q.bits == 8:
            codes = data
        else:
            codes = np.empty((n, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM), np.uint8)
            codes[..., 0::2] = data & 0xF
            codes[..., 1::2] = data >> 4
        params = np.ascontiguousarray(pages[:, :, q.param_offset : q.param_offset + q.param_bytes]).view(np.float16)
        if q.mode == 1:
            params = params.reshape(n, NUM_KV_HEADS, PAGE_SIZE, 2)
            scale = params[..., 0:1]
            zp = np.float16(q.zero_sym) if q.sym else params[..., 1:2]
        else:
            params = params.reshape(n, NUM_KV_HEADS, PAGE_SIZE // q.group, 1 if q.sym else 2, HEAD_DIM)
            scale = np.repeat(params[:, :, :, 0, :], q.group, axis=2)
            zp = np.float16(q.zero_sym) if q.sym else np.repeat(params[:, :, :, 1, :], q.group, axis=2)
        deq = (codes.astype(np.float16) - zp) * scale  # f16 steps, like qsa_q_dq
    return deq.astype(np.float32).transpose(0, 2, 1, 3).reshape(n * PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)


# ===================== Test data =====================
@dataclass(frozen=True)
class SeqSpec:
    past: int
    q_len: int


def make_rows(rng: np.random.Generator, n: int, magnitude: float) -> np.ndarray:
    """f16 [n, Hkv, D] rows with per-channel magnitudes and rare outliers (stresses per-token and per-channel scales)."""
    cols = NUM_KV_HEADS * HEAD_DIM
    pool_n = min(n, 4096)
    x = rng.standard_normal((pool_n, cols), dtype=np.float32)
    x *= magnitude * (1.0 + (np.arange(cols) % 7) * 0.5).astype(np.float32)
    x[rng.random((pool_n, cols)) < 0.005] *= 4.0
    pool = x.astype(np.float16)
    rows = pool if n == pool_n else pool[np.arange(n) % pool_n]  # timing does not depend on the data, so large inputs tile a page-aligned pool
    return rows.reshape(n, NUM_KV_HEADS, HEAD_DIM)


def make_query(rng: np.random.Generator, n: int) -> np.ndarray:
    pool_n = min(n, 4096)
    pool = rng.standard_normal((pool_n, Q_PITCH), dtype=np.float32).astype(np.float16)
    return pool if n == pool_n else pool[np.arange(n) % pool_n]


def make_selection(past: np.ndarray, q_len: np.ndarray, subsequence_begins: np.ndarray):
    """Per token: ascending block ids spread over all complete blocks (at most TOPK of them), like the SparsePA unit tests."""
    tokens = int(subsequence_begins[-1])
    sel = np.zeros((tokens, TOPK), np.int32)
    count = np.zeros(tokens, np.int32)
    j = np.arange(TOPK, dtype=np.int32)[None, :]
    for s in range(len(past)):
        begin, end = int(subsequence_begins[s]), int(subsequence_begins[s + 1])
        n_complete = (past[s] + np.arange(q_len[s], dtype=np.int32) + 1) // BLOCK_SIZE
        cnt = np.minimum(TOPK, n_complete)
        block = (j * n_complete[:, None]) // np.maximum(cnt, 1)[:, None]
        block[j >= cnt[:, None]] = 0
        sel[begin:end] = block
        count[begin:end] = cnt
    return sel, count


class Case:
    """Host-side description of one attention call: sequences, block table, K/V history, queries and selected blocks."""

    def __init__(self, seqs: list[SeqSpec], seed: int):
        self.num_seqs = len(seqs)
        self.past = np.array([s.past for s in seqs], np.int32)
        self.q_len = np.array([s.q_len for s in seqs], np.int32)
        self.subsequence_begins = np.concatenate([[0], np.cumsum(self.q_len)]).astype(np.int32)
        self.num_tokens = int(self.subsequence_begins[-1])
        pages = (self.past + self.q_len + PAGE_SIZE - 1) // PAGE_SIZE
        self.block_indices_begins = np.concatenate([[0], np.cumsum(pages)]).astype(np.int32)
        self.num_pages = int(self.block_indices_begins[-1])
        rng = np.random.default_rng(seed)
        self.block_indices = rng.permutation(self.num_pages).astype(np.int32)  # physical pages are not in logical order
        # K / V history of every sequence, in logical page order (sequence i starts at row block_indices_begins[i] * PAGE_SIZE).
        self.key_rows = make_rows(rng, self.num_pages * PAGE_SIZE, 0.5)
        self.value_rows = make_rows(rng, self.num_pages * PAGE_SIZE, 1.0)
        self.query = make_query(rng, self.num_tokens)
        self.selected_blocks, self.selected_counts = make_selection(self.past, self.q_len, self.subsequence_begins)

    def logical_caches(self, cfg: KvConfig):
        return pack_cache(self.key_rows, cfg.key), pack_cache(self.value_rows, cfg.value)

    def physical(self, logical: np.ndarray) -> np.ndarray:
        physical = np.empty_like(logical)
        physical[self.block_indices] = logical
        return physical


def make_inputs(queue: cl.CommandQueue, case: Case, cfg: KvConfig, layers: int = 1, poison: bool = False) -> list[AttentionInputs]:
    """Uploads a case once per layer: the layers share the small inputs but own their K/V caches and scratch (cold caches)."""
    ctx = queue.context
    mf = cl.mem_flags

    def ro(arr):
        return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr))

    def scratch(nbytes: int):
        if poison:  # NaN: partial rows that are not written show up in the output
            return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.full(nbytes, 0xFF, np.uint8))
        return cl.Buffer(ctx, mf.READ_WRITE, size=nbytes)

    key_logical, value_logical = case.logical_caches(cfg)
    key_physical, value_physical = case.physical(key_logical), case.physical(value_logical)
    shared = dict(
        num_tokens=case.num_tokens,
        num_seqs=case.num_seqs,
        query=ro(case.query),
        past_lens=ro(case.past),
        subsequence_begins=ro(case.subsequence_begins),
        block_indices=ro(case.block_indices),
        block_indices_begins=ro(case.block_indices_begins),
        selected_blocks=ro(case.selected_blocks),
        selected_counts=ro(case.selected_counts),
    )
    rows = case.num_tokens * NUM_HEADS * SPLIT_PARTITIONS
    return [
        AttentionInputs(
            key_cache=ro(key_physical),
            value_cache=ro(value_physical),
            partial_acc=scratch(rows * HEAD_DIM * 4),
            partial_ml=scratch(rows * 2 * 4),
            output=scratch(case.num_tokens * Q_PITCH * 2),
            **shared,
        )
        for _ in range(layers)
    ]


def read_output(queue: cl.CommandQueue, inp: AttentionInputs) -> np.ndarray:
    out = np.empty(inp.num_tokens * Q_PITCH, np.float16)
    cl.enqueue_copy(queue, out, inp.output)
    queue.finish()
    return out.astype(np.float32).reshape(inp.num_tokens, NUM_HEADS, HEAD_DIM)


# ===================== CPU model =====================
def reference_attention(case: Case, key_tokens: np.ndarray, value_tokens: np.ndarray) -> np.ndarray:
    """Softmax attention of every token over its selected blocks plus its incomplete tail block, on dequantized K/V rows."""
    group = NUM_HEADS // NUM_KV_HEADS
    query = case.query.astype(np.float32).reshape(case.num_tokens, NUM_HEADS, HEAD_DIM)
    out = np.zeros((case.num_tokens, NUM_HEADS, HEAD_DIM), np.float32)
    offsets = np.arange(BLOCK_SIZE)[None, :]
    for s in range(case.num_seqs):
        base = int(case.block_indices_begins[s]) * PAGE_SIZE
        for i in range(int(case.q_len[s])):
            t = int(case.subsequence_begins[s]) + i
            abs_pos = int(case.past[s]) + i
            blocks = case.selected_blocks[t, : case.selected_counts[t]].astype(np.int64)
            tail = np.arange((abs_pos + 1) // BLOCK_SIZE * BLOCK_SIZE, abs_pos + 1)
            rows = base + np.concatenate([(blocks[:, None] * BLOCK_SIZE + offsets).reshape(-1), tail])
            k, v = key_tokens[rows], value_tokens[rows]  # [n, Hkv, D]
            for h in range(NUM_KV_HEADS):
                heads = slice(h * group, (h + 1) * group)
                logits = (query[t, heads] @ k[:, h].T) * SCALE
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                out[t, heads] = p @ v[:, h]
    return out


def assert_close(name: str, got: np.ndarray, expected: np.ndarray, atol: float, rtol: float):
    bad = ~(np.abs(got - expected) <= atol + rtol * np.abs(expected))  # NaN (unwritten output) counts as bad
    assert not bad.any(), f"{name}: {int(bad.sum())} of {bad.size} outputs differ, max abs error {np.nanmax(np.abs(got - expected)):.4g}"


SCENARIOS = {
    # one new token per sequence at very different contexts: empty selection (past 0, 3), tails of 0..3 rows, many empty partitions
    "decode_multi": [SeqSpec(p, 1) for p in (15, 100, 1023, 5000, 2047, 3, 0)],
    "decode_batch": [SeqSpec(p, 1) for p in (511, 1023, 2047, 4095, 8191, 100, 17, 3000)],
    # more complete blocks than TOPK: the strided top-k selection is capped
    "decode_16K": [SeqSpec(16383, 1)],
    "decode_64K": [SeqSpec(65535, 1)],
}
CPU_CASES = [(cfg, "decode_multi") for cfg in ALL_CONFIGS] + [(cfg, s) for cfg in BENCH_CONFIGS for s in ("decode_16K", "decode_64K")]
REFERENCE_CASES = [(cfg, "decode_batch") for cfg in BENCH_CONFIGS]


def _case(scenario: str) -> Case:
    return Case(SCENARIOS[scenario], seed=zlib.crc32(scenario.encode()))


# ===================== Fixtures =====================
def _has_gpu() -> bool:
    try:
        return any(p.get_devices(cl.device_type.GPU) for p in cl.get_platforms())
    except cl.Error as e:
        print(f"[build] OpenCL GPU query failed: {type(e).__name__}: {e}")
        return False


def _gpu_queue() -> cl.CommandQueue:
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            return cl.CommandQueue(cl.Context([devs[0]]), properties=cl.command_queue_properties.PROFILING_ENABLE)
    raise RuntimeError("No OpenCL GPU device found")


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
    return _gpu_queue()


@pytest.fixture(scope="session")
def kernel_file(use_reference):
    return REFERENCE_FILE if use_reference else KERNEL_FILE


# ===================== Test class =====================
class TestQsaSparseAttentionDecode(TestBase):
    """SparsePA decode attention (split + merge) over a paged f16 / u8 / u4 KV cache."""

    def build(self, gpu_arch) -> list[str]:
        """Compiles kernel.cl for every cache config of the tests."""
        if not _has_gpu():
            print("[build] No OpenCL GPU device found: skipping the kernel.cl compile check (it is compiled when the tests run on the GPU machine).")
            return
        queue = _gpu_queue()
        for cfg in ALL_CONFIGS:
            build_op(queue, KERNEL_FILE, cfg)

    @pytest.mark.parametrize("cfg, scenario", CPU_CASES, ids=[f"{c.name}-{s}" for c, s in CPU_CASES])
    def test_correctness_wrt_cpu(self, ocl_queue, kernel_file, cfg, scenario):
        """Attention output vs a numpy model that reads the same (dequantized) cache."""
        case = _case(scenario)
        inp = make_inputs(ocl_queue, case, cfg, poison=True)[0]
        build_op(ocl_queue, kernel_file, cfg)([inp])
        ocl_queue.finish()
        got = read_output(ocl_queue, inp)

        key_logical, value_logical = case.logical_caches(cfg)
        expected = reference_attention(case, decode_cache(key_logical, cfg.key), decode_cache(value_logical, cfg.value))
        assert_close("attention output vs CPU model", got, expected, atol=1.5e-2, rtol=1.5e-2)

    @pytest.mark.parametrize("cfg, scenario", REFERENCE_CASES, ids=[f"{c.name}-{s}" for c, s in REFERENCE_CASES])
    def test_correctness_wrt_reference(self, ocl_queue, kernel_file, cfg, scenario):
        """Attention output vs reference.cl on the same inputs."""
        case = _case(scenario)
        inp = make_inputs(ocl_queue, case, cfg, poison=True)[0]
        build_op(ocl_queue, kernel_file, cfg)([inp])
        ocl_queue.finish()
        got = read_output(ocl_queue, inp)

        ref_inp = make_inputs(ocl_queue, case, cfg, poison=True)[0]
        build_op(ocl_queue, REFERENCE_FILE, cfg)([ref_inp])
        ocl_queue.finish()
        assert_close("attention output vs reference.cl", got, read_output(ocl_queue, ref_inp), atol=1e-2, rtol=1e-2)

    @pytest.mark.performance
    @pytest.mark.parametrize("cfg", BENCH_CONFIGS, ids=[c.name for c in BENCH_CONFIGS])
    @pytest.mark.parametrize("length", CONTEXT_LENGTHS, ids=[f"{n // 1024}K" for n in CONTEXT_LENGTHS])
    def test_benchmark_decode(self, ocl_queue, kernel_file, measure_runtime, cfg, length):
        """One decode step of a single sequence at context `length`: all full-attention layers back to back, each with its own cache."""
        op = build_op(ocl_queue, kernel_file, cfg)
        layers = make_inputs(ocl_queue, Case([SeqSpec(length - 1, 1)], seed=length), cfg, layers=NUM_FULL_ATTN_LAYERS)
        measure_runtime(
            op,
            args=(layers,),
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestQsaSparseAttentionDecode()
    task.build(TestBase.get_machine_gpu_arch())
