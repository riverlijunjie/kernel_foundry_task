"""KernelFoundry task: main KV cache write ("KV cache compression") of the OpenVINO intel_gpu SparsePA primitive.

The kernels (kernel.cl / reference.cl) are extracted from sparse_attention_kv_write.cl and quantize the new K / V rows
(u8 / u4, BY_TOKEN or BY_CHANNEL) into a paged cache. Model geometry comes from the Qwen4-exp config.json: 2 kv heads,
head_dim 256, 12 full-attention layers. The page size (16) is the one used by the SparsePA unit tests.
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
NUM_KV_HEADS = 2  # text_config.num_key_value_heads
NUM_FULL_ATTN_LAYERS = 12  # text_config.layer_types.count("full_attention"); one KV write per layer and decode step
PAGE_SIZE = 16  # tokens per physical cache page
SUBGROUP_SIZE = 16  # Xe2 (ptl / lnl / bmg)
ROW_PITCH = NUM_KV_HEADS * HEAD_DIM  # elements per token of the key / value input tensors

# CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]

U32 = np.uint32


# ===================== Cache quantization configs =====================
@dataclass(frozen=True)
class CacheQuant:
    mode: int  # 1 = BY_TOKEN, 2 = BY_CHANNEL
    bits: int  # 8 or 4
    sym: bool
    group: int = 0  # tokens per parameter group (BY_CHANNEL only)

    @property
    def row_bytes(self) -> int:
        return HEAD_DIM * self.bits // 8

    @property
    def param_bytes(self) -> int:
        if self.mode == 1:
            return PAGE_SIZE * 4  # [scale f16][zp f16] per token
        return (PAGE_SIZE // self.group) * HEAD_DIM * (1 if self.sym else 2) * 2

    @property
    def record_bytes(self) -> int:
        """Bytes of one BY_CHANNEL group record: [scale f16 x D] (+ [zp f16 x D] when asymmetric)."""
        return HEAD_DIM * (2 if self.sym else 4)

    @property
    def total_rows(self) -> int:
        return PAGE_SIZE + -(-self.param_bytes // self.row_bytes)

    @property
    def page_bytes(self) -> int:
        return self.total_rows * self.row_bytes

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


# Plugin default for kv_cache_precision=u8: key BY_CHANNEL (group 16), value BY_TOKEN, asymmetric.
PRIMARY = KvConfig("ch16_u8_asym", CacheQuant(2, 8, False, 16), CacheQuant(1, 8, False))
ALL_CONFIGS = [
    PRIMARY,
    KvConfig("ch16_u4_asym", CacheQuant(2, 4, False, 16), CacheQuant(1, 4, False)),
    KvConfig("ch16_u8_sym", CacheQuant(2, 8, True, 16), CacheQuant(1, 8, True)),
    KvConfig("tok_u8_asym", CacheQuant(1, 8, False), CacheQuant(1, 8, False)),
    KvConfig("tok_u4_sym", CacheQuant(1, 4, True), CacheQuant(1, 4, True)),
]


def _defines(cfg: KvConfig, kchannel: bool) -> dict:
    d = {
        "SUBGROUP_SIZE": SUBGROUP_SIZE,
        "QSA_KV_HEADS": NUM_KV_HEADS,
        "QSA_K_HEAD_SIZE": HEAD_DIM,
        "QSA_V_HEAD_SIZE": HEAD_DIM,
        "QSA_PAGE_SIZE": PAGE_SIZE,
    }
    for name, q in (("K", cfg.key), ("V", cfg.value)):
        d[f"QSA_{name}_MODE"] = q.mode
        d[f"QSA_{name}_BITS"] = q.bits
        d[f"QSA_{name}_SYM"] = int(q.sym)
        d[f"QSA_{name}_GROUP"] = max(1, q.group)
        d[f"QSA_{name}_ROW_BYTES"] = q.row_bytes
        d[f"QSA_{name}_PAGE_BYTES"] = q.page_bytes
        d[f"QSA_{name}_PARAM_OFFSET"] = q.param_offset
    if kchannel:
        d["QSA_KV_WRITE_KCHANNEL"] = 1
    return d


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
class WriteInputs:
    num_tokens: int
    num_seqs: int
    key: cl.Buffer
    value: cl.Buffer
    past_lens: cl.Buffer
    block_indices: cl.Buffer
    block_indices_begins: cl.Buffer
    subsequence_begins: cl.Buffer
    key_cache: cl.Buffer
    value_cache: cl.Buffer


class KvWriteOp:
    """Launches sparse_attention_kv_write (+ sparse_attention_kv_write_kchannel for BY_CHANNEL keys) like the plugin does."""

    def __init__(self, queue: cl.CommandQueue, filename: str, cfg: KvConfig):
        source = (HERE / filename).read_text()
        self.queue = queue
        self.cfg = cfg
        self._main = _build_program(queue.context, source, _defines(cfg, False), large_grf=False).sparse_attention_kv_write
        self._kchannel = None
        if cfg.key.mode == 2:
            program = _build_program(queue.context, source, _defines(cfg, True), large_grf=True)
            self._kchannel = program.sparse_attention_kv_write_kchannel

    def write_key_channel(self, key, past_lens, block_indices, block_indices_begins, subsequence_begins, key_cache, num_tokens, num_seqs):
        # A sequence touches at most tokens / group + 2 groups (one partial group at each end).
        work_items = max(1, num_tokens // self.cfg.key.group + 2 * num_seqs)
        return self._kchannel(
            self.queue,
            (work_items * SUBGROUP_SIZE, NUM_KV_HEADS, 1),
            (SUBGROUP_SIZE, 1, 1),
            key,
            past_lens,
            block_indices,
            block_indices_begins,
            subsequence_begins,
            key_cache,
            U32(ROW_PITCH),
            U32(0),
            U32(num_seqs),
        )

    def __call__(self, inp: WriteInputs, repeat: int = 1):
        event = None
        for _ in range(repeat):
            event = self._main(
                self.queue,
                (SUBGROUP_SIZE, NUM_KV_HEADS, max(1, inp.num_tokens)),
                (SUBGROUP_SIZE, 1, 1),
                inp.key,
                inp.value,
                inp.past_lens,
                inp.block_indices,
                inp.block_indices_begins,
                inp.subsequence_begins,
                inp.key_cache,
                inp.value_cache,
                U32(ROW_PITCH),
                U32(0),
                U32(ROW_PITCH),
                U32(0),
                U32(inp.num_seqs),
            )
            if self._kchannel is not None:
                event = self.write_key_channel(
                    inp.key, inp.past_lens, inp.block_indices, inp.block_indices_begins, inp.subsequence_begins, inp.key_cache, inp.num_tokens, inp.num_seqs
                )
        return event


_OPS: dict = {}


def build_op(queue: cl.CommandQueue, filename: str, cfg: KvConfig) -> KvWriteOp:
    key = (filename, cfg.name, queue.int_ptr)
    if key not in _OPS:
        _OPS[key] = KvWriteOp(queue, filename, cfg)
    return _OPS[key]


# ===================== Test data =====================
@dataclass(frozen=True)
class SeqSpec:
    past: int
    q_len: int


def make_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """f16 [n, ROW_PITCH] rows with per-channel magnitudes and rare outliers (stresses per-token and per-channel scales)."""
    pool_n = min(n, 4096)
    x = rng.standard_normal((pool_n, ROW_PITCH), dtype=np.float32)
    x *= (1.0 + (np.arange(ROW_PITCH) % 7) * 0.5).astype(np.float32)
    x[rng.random((pool_n, ROW_PITCH)) < 0.005] *= 4.0
    pool = x.astype(np.float16)
    return pool if n == pool_n else pool[np.arange(n) % pool_n]  # timing does not depend on the data, so large inputs tile a pool


class Case:
    """Host-side description of one write call: sequences, block table and the new K / V rows."""

    def __init__(self, seqs: list[SeqSpec], seed: int):
        self.seed = seed
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
        self.key = make_rows(rng, self.num_tokens)
        self.value = make_rows(rng, self.num_tokens)

    def key_tails(self, group: int):
        """BY_CHANNEL: the cached tokens [g0, past) of the group that the first new token of each sequence lands in."""
        g0 = (self.past // group * group).astype(np.int32)
        tail_len = (self.past - g0).astype(np.int32)
        n = int(tail_len.sum())
        rows = make_rows(np.random.default_rng(self.seed + 7919 * group), n) if n else np.zeros((0, ROW_PITCH), np.float16)
        return g0, tail_len, rows


def make_inputs(queue: cl.CommandQueue, case: Case, cfg: KvConfig, reference_op: KvWriteOp, poison: int = 0xFF) -> WriteInputs:
    """Uploads a case into fresh caches. BY_CHANNEL keys first get a valid cached tail group, written by the reference kernel."""
    ctx = queue.context
    mf = cl.mem_flags

    def ro(arr):
        return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr))

    def cache(q: CacheQuant):
        host = np.full(case.num_pages * NUM_KV_HEADS * q.page_bytes, poison, np.uint8)
        return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host)

    inp = WriteInputs(
        num_tokens=case.num_tokens,
        num_seqs=case.num_seqs,
        key=ro(case.key),
        value=ro(case.value),
        past_lens=ro(case.past),
        block_indices=ro(case.block_indices),
        block_indices_begins=ro(case.block_indices_begins),
        subsequence_begins=ro(case.subsequence_begins),
        key_cache=cache(cfg.key),
        value_cache=cache(cfg.value),
    )
    if cfg.key.mode == 2:
        g0, tail_len, rows = case.key_tails(cfg.key.group)
        n = int(tail_len.sum())
        if n:
            tail_begins = np.concatenate([[0], np.cumsum(tail_len)]).astype(np.int32)
            reference_op.write_key_channel(ro(rows), ro(g0), inp.block_indices, inp.block_indices_begins, ro(tail_begins), inp.key_cache, n, case.num_seqs)
            queue.finish()
    return inp


def read_cache(queue: cl.CommandQueue, buf: cl.Buffer) -> np.ndarray:
    out = np.empty(buf.size, np.uint8)
    cl.enqueue_copy(queue, out, buf)
    queue.finish()
    return out


# ===================== Cache decoding / verification =====================
def decode_cache(raw: np.ndarray, q: CacheQuant, num_pages: int):
    """Returns (dequantized f16, scale f16, codes u8), each [pages, heads, PAGE_SIZE, D] (scale is [.., 1] for BY_TOKEN)."""
    pages = raw.reshape(num_pages, NUM_KV_HEADS, q.page_bytes)
    data = pages[:, :, : PAGE_SIZE * q.row_bytes].reshape(num_pages, NUM_KV_HEADS, PAGE_SIZE, q.row_bytes)
    if q.bits == 8:
        codes = data
    else:
        codes = np.empty((num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM), np.uint8)
        codes[..., 0::2] = data & 0xF
        codes[..., 1::2] = data >> 4
    params = np.ascontiguousarray(pages[:, :, q.param_offset : q.param_offset + q.param_bytes]).view(np.float16)
    if q.mode == 1:
        params = params.reshape(num_pages, NUM_KV_HEADS, PAGE_SIZE, 2)
        scale = params[..., 0:1]
        zp = np.float16(q.zero_sym) if q.sym else params[..., 1:2]
    else:
        params = params.reshape(num_pages, NUM_KV_HEADS, PAGE_SIZE // q.group, 1 if q.sym else 2, HEAD_DIM)
        scale = np.repeat(params[:, :, :, 0, :], q.group, axis=2)
        zp = np.float16(q.zero_sym) if q.sym else np.repeat(params[:, :, :, 1, :], q.group, axis=2)
    deq = (codes.astype(np.float16) - zp) * scale  # f16 steps, like qsa_q_dq
    return deq, scale, codes


def _touched(case: Case, lo: list[int]):
    """(physical page, token in page) of every position [lo[i], past + q_len) of every sequence."""
    pages, toks = [], []
    for i in range(case.num_seqs):
        pos = np.arange(lo[i], int(case.past[i] + case.q_len[i]))
        pages.append(case.block_indices[case.block_indices_begins[i] + pos // PAGE_SIZE])
        toks.append(pos % PAGE_SIZE)
    return np.concatenate(pages), np.concatenate(toks)


def _touched_byte_mask(q: CacheQuant, num_pages: int, pages: np.ndarray, toks: np.ndarray) -> np.ndarray:
    """[pages, page_bytes] mask of the data rows and parameters the write may modify."""
    token_mask = np.zeros((num_pages, PAGE_SIZE), bool)
    token_mask[pages, toks] = True
    mask = np.zeros((num_pages, q.page_bytes), bool)
    data = np.repeat(token_mask, q.row_bytes, axis=1)
    mask[:, : data.shape[1]] = data
    if q.mode == 1:
        params = np.repeat(token_mask, 4, axis=1)
    else:
        group_mask = np.zeros((num_pages, PAGE_SIZE // q.group), bool)
        group_mask[pages, toks // q.group] = True
        params = np.repeat(group_mask, q.record_bytes, axis=1)
    mask[:, q.param_offset : q.param_offset + params.shape[1]] = params
    return mask


def verify_cache(name, case, q, lo, truth, initial, result, reference):
    """Checks the cache `result` of the kernel under test against the input rows and against the reference kernel's cache."""
    n_pages = case.num_pages
    pages, toks = _touched(case, lo)

    # 1. nothing outside the written tokens / groups changed
    mask = _touched_byte_mask(q, n_pages, pages, toks)
    shape = (n_pages, NUM_KV_HEADS, q.page_bytes)
    outside = np.broadcast_to(~mask[:, None, :], shape)
    assert np.array_equal(result.reshape(shape)[outside], initial.reshape(shape)[outside]), f"{name} cache: bytes outside the written tokens / groups were modified"

    with np.errstate(all="ignore"):
        deq, scale, codes = decode_cache(result, q, n_pages)
        got = deq[pages, :, toks, :].astype(np.float32)  # [n, heads, D]
        step = scale[pages, :, toks, :].astype(np.float32)
        expected = np.concatenate([t[l:] for t, l in zip(truth, lo)]).reshape(-1, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)

        # 2. dequantized rows reproduce the input within the quantization step (cached rows of a group are re-quantized once)
        err = np.abs(got - expected)
        bound = 1.25 * step + 4e-3 * np.abs(expected) + 1e-3
        assert np.all(err <= bound), f"{name} cache: dequantization error {np.nanmax(err):.4g} exceeds the quantization step (bad codes or parameters)"

        # 3. same result as the reference kernel up to rare one-code rounding differences
        ref_deq, ref_scale, ref_codes = decode_cache(reference, q, n_pages)
        ref_got = ref_deq[pages, :, toks, :].astype(np.float32)
        ref_step = ref_scale[pages, :, toks, :].astype(np.float32)
        assert np.all(np.abs(got - ref_got) <= 1.5 * ref_step + 2e-3 * np.abs(ref_got) + 1e-3), f"{name} cache: differs from the reference kernel"
        mismatch = np.mean(codes[pages, :, toks, :] != ref_codes[pages, :, toks, :])
        assert mismatch < 0.005, f"{name} cache: {mismatch:.2%} of the codes differ from the reference kernel"


def build_truth(case: Case, cfg: KvConfig):
    """Per sequence f16 [past + q_len, ROW_PITCH] input rows (only positions >= lo are meaningful) and first touched position."""
    g0, tail_len, tail_rows = case.key_tails(cfg.key.group) if cfg.key.mode == 2 else (None, None, None)
    key_truth, key_lo, value_truth, value_lo = [], [], [], []
    tail_off = 0
    for i in range(case.num_seqs):
        past, q_len, begin = int(case.past[i]), int(case.q_len[i]), int(case.subsequence_begins[i])
        k = np.zeros((past + q_len, ROW_PITCH), np.float16)
        v = np.zeros_like(k)
        k[past:] = case.key[begin : begin + q_len]
        v[past:] = case.value[begin : begin + q_len]
        if cfg.key.mode == 2:
            n = int(tail_len[i])
            k[g0[i] : past] = tail_rows[tail_off : tail_off + n]
            tail_off += n
            key_lo.append(int(g0[i]))
        else:
            key_lo.append(past)
        value_lo.append(past)
        key_truth.append(k)
        value_truth.append(v)
    return key_truth, key_lo, value_truth, value_lo


SCENARIOS = {
    # mixed partial groups: sequence start, start inside a group, page-aligned start
    "prefill_multi": [SeqSpec(0, 77), SeqSpec(37, 200), SeqSpec(64, 16)],
    # one token per sequence: tail of 15 / 4 / 15 / 0 cached tokens in the group
    "decode_multi": [SeqSpec(15, 1), SeqSpec(100, 1), SeqSpec(1023, 1), SeqSpec(16, 1)],
    "prefill_4K": [SeqSpec(0, 4096)],
    "decode_16K": [SeqSpec(16383, 1)],
}
CORRECTNESS_CASES = [(cfg, s) for cfg in ALL_CONFIGS for s in ("prefill_multi", "decode_multi")] + [(PRIMARY, "prefill_4K"), (PRIMARY, "decode_16K")]


def run_case(queue: cl.CommandQueue, kernel_file: str, cfg: KvConfig, scenario: str):
    case = Case(SCENARIOS[scenario], seed=zlib.crc32(scenario.encode()))
    reference_op = build_op(queue, REFERENCE_FILE, cfg)
    op = build_op(queue, kernel_file, cfg)

    ref_inp = make_inputs(queue, case, cfg, reference_op)
    initial_key, initial_value = read_cache(queue, ref_inp.key_cache), read_cache(queue, ref_inp.value_cache)
    reference_op(ref_inp)
    queue.finish()
    ref_key, ref_value = read_cache(queue, ref_inp.key_cache), read_cache(queue, ref_inp.value_cache)

    inp = make_inputs(queue, case, cfg, reference_op)
    op(inp)
    queue.finish()
    out_key, out_value = read_cache(queue, inp.key_cache), read_cache(queue, inp.value_cache)

    key_truth, key_lo, value_truth, value_lo = build_truth(case, cfg)
    verify_cache("key", case, cfg.key, key_lo, key_truth, initial_key, out_key, ref_key)
    verify_cache("value", case, cfg.value, value_lo, value_truth, initial_value, out_value, ref_value)


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


@pytest.fixture(scope="session")
def reference_op(ocl_queue):
    return build_op(ocl_queue, REFERENCE_FILE, PRIMARY)


@pytest.fixture(scope="session")
def kernel(kernel_file, ocl_queue):
    """Callable kernel(inputs, repeat) for the benchmarked (primary) cache config."""
    return build_op(ocl_queue, kernel_file, PRIMARY)


# ===================== Test class =====================
class TestQsaKvCacheCompression(TestBase):
    """SparsePA main KV cache write: quantize new K / V rows into the paged cache."""

    def build(self, gpu_arch) -> list[str]:
        """Compiles kernel.cl for every cache config of the tests (both programs of BY_CHANNEL configs)."""
        if not _has_gpu():
            print("[build] No OpenCL GPU device found: skipping the kernel.cl compile check (it is compiled when the tests run on the GPU machine).")
            return
        queue = _gpu_queue()
        for cfg in ALL_CONFIGS:
            build_op(queue, KERNEL_FILE, cfg)

    @pytest.mark.parametrize("cfg, scenario", CORRECTNESS_CASES, ids=[f"{c.name}-{s}" for c, s in CORRECTNESS_CASES])
    def test_correctness(self, ocl_queue, kernel_file, cfg, scenario):
        """Cache bytes written by the kernel vs the input rows and vs reference.cl, for several quantization configs."""
        run_case(ocl_queue, kernel_file, cfg, scenario)

    @pytest.mark.performance
    @pytest.mark.parametrize("phase", ["prefill", "decode"])
    @pytest.mark.parametrize("length", CONTEXT_LENGTHS, ids=[f"{n // 1024}K" for n in CONTEXT_LENGTHS])
    def test_benchmark(self, kernel, reference_op, ocl_queue, measure_runtime, phase, length):
        """Prefill: one sequence of `length` new tokens. Decode: one new token at context `length`, written for all
        full-attention layers (back-to-back launches hide the host launch overhead of the tiny kernels)."""
        if phase == "prefill":
            seqs, repeat = [SeqSpec(0, length)], 1
        else:
            seqs, repeat = [SeqSpec(length - 1, 1)], NUM_FULL_ATTN_LAYERS
        case = Case(seqs, seed=length)
        inp = make_inputs(ocl_queue, case, PRIMARY, reference_op, poison=0)
        measure_runtime(
            kernel,
            args=(inp, repeat),
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestQsaKvCacheCompression()
    task.build(TestBase.get_machine_gpu_arch())
