"""KernelFoundry task: selected-block GQA systolic attention (sparse_attention_gqa_dpas) of the OpenVINO intel_gpu SparsePA primitive.

The kernels (kernel.cl / reference.cl) are extracted from sparse_attention_gqa_dpas.cl (paged addressing, prefill variant)
and attend, per token and KV head, to the top-k selected blocks of a paged f16 / u8 / u4 KV cache plus the incomplete
tail block. Model geometry comes from the Qwen4-exp config.json: 24 query heads, 2 kv heads, head_dim 256,
indexer_compress_ratio 4 (block size) and indexer_budget 2048 tokens (512 selected blocks). The page size (16) is the one
used by the SparsePA unit tests. The caches are pre-filled by the host (the KV write stage is a separate kernel).

How the kernel is launched is not fixed by this file: kernel.cl declares it in a [LAUNCH_START] / [LAUNCH_END] comment block
(entry point, argument list, dispatch geometry, subgroup size, 256-GRF mode, extra build options), see parse_launch().
"""

import ast
import re
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
SUBGROUP_SIZE = 16  # Xe2 (ptl / lnl / bmg); the default of the launch description
SCALE = 1.0 / np.sqrt(HEAD_DIM)
Q_PITCH = NUM_HEADS * HEAD_DIM  # elements per token of the query / output tensors
MIN_SCALE = 1.0 / 16384.0

# CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]

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


def _defines(cfg: KvConfig, subgroup_size: int) -> dict:
    d = {
        "SUBGROUP_SIZE": subgroup_size,
        "QSA_HEADS": NUM_HEADS,
        "QSA_KV_HEADS": NUM_KV_HEADS,
        "QSA_K_HEAD_SIZE": HEAD_DIM,
        "QSA_V_HEAD_SIZE": HEAD_DIM,
        "QSA_COMPRESS_RATIO": BLOCK_SIZE,
        "QSA_BLOCK_TOPK": TOPK,
        "QSA_SCALE": f"{SCALE:.9g}f",
        "QSA_IS_CAUSAL": 1,
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
    return d


# ===================== Launch description (declared by the kernel source) =====================
BUFFER_ARGS = (
    "query",
    "key_cache",
    "value_cache",
    "past_lens",
    "subsequence_begins",
    "block_indices",
    "block_indices_begins",
    "selected_blocks",
    "selected_counts",
    "output",
)
SCALAR_ARGS = ("num_tokens", "num_seqs", "q_base", "q_pitch", "scale")  # uint32 except `scale` (float32)
DISPATCH_VARS = ("tokens", "num_seqs", "sg", "heads", "kv_heads", "head_dim", "topk", "block_size", "page_size")

# Used when kernel.cl has no [LAUNCH_START] block or omits a key: the launch of the original plugin kernel.
DEFAULT_LAUNCH = {
    "kernel": "sparse_attention_paged_gqa_dpas",
    "args": ", ".join(BUFFER_ARGS + ("num_tokens", "num_seqs", "q_base", "q_pitch")),
    "gws": "tokens * 4 * sg, kv_heads, 1",
    "lws": "4 * sg, 1, 1",
    "subgroup_size": str(SUBGROUP_SIZE),
    "grf_256": "1",
    "build_options": "",
}

_LAUNCH_BLOCK = re.compile(r"//\s*\[LAUNCH_START\](.*?)//\s*\[LAUNCH_END\]", re.S)
_LAUNCH_LINE = re.compile(r"^\s*//\s*(kernel|args|gws|lws|subgroup_size|grf_256|build_options)\s*=\s*(.*?)\s*$")
_BUILD_OPTION = re.compile(r"^(-cl-[A-Za-z0-9=.\-]+|-D[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.+\-]*)?)$")
_DISPATCH_FUNCS = {"min": min, "max": max, "cdiv": lambda a, b: -(-a // b)}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _compile_expr(text: str, what: str):
    """Compiles an integer dispatch expression; only numbers, DISPATCH_VARS, + - * // % and min / max / cdiv are allowed."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"launch {what}: invalid expression {text!r}: {e}") from e
    for node in ast.walk(tree):
        bad = (
            not isinstance(node, _ALLOWED_NODES)
            or (isinstance(node, ast.Constant) and type(node.value) is not int)
            or (isinstance(node, ast.Name) and node.id not in DISPATCH_VARS and node.id not in _DISPATCH_FUNCS)
            or (isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _DISPATCH_FUNCS or node.keywords))
        )
        if bad:
            raise ValueError(f"launch {what}: unsupported syntax in {text!r} (variables: {', '.join(DISPATCH_VARS)}; operators + - * // %; min, max, cdiv)")
    return compile(tree, "<launch>", "eval")


@dataclass
class LaunchConfig:
    kernel: str
    args: tuple
    gws: tuple  # three compiled expressions
    lws: tuple
    subgroup_size: int
    grf_256: bool
    build_options: tuple


def parse_launch(source: str) -> LaunchConfig:
    """Reads the `key = value` lines of the [LAUNCH_START] / [LAUNCH_END] comment block of a kernel source."""
    values = dict(DEFAULT_LAUNCH)
    block = _LAUNCH_BLOCK.search(source)
    if block:
        for line in block.group(1).splitlines():
            match = _LAUNCH_LINE.match(line)
            if match:
                values[match.group(1)] = match.group(2)

    kernel = values["kernel"]
    if not re.fullmatch(r"[A-Za-z_]\w*", kernel):
        raise ValueError(f"launch kernel: invalid entry point name {kernel!r}")
    args = tuple(a for a in _split_top_level(values["args"]) if a)
    for a in args:
        if a not in BUFFER_ARGS + SCALAR_ARGS:
            raise ValueError(f"launch args: unknown argument {a!r} (buffers: {', '.join(BUFFER_ARGS)}; scalars: {', '.join(SCALAR_ARGS)})")
    dims = {}
    for key in ("gws", "lws"):
        exprs = _split_top_level(values[key])
        if len(exprs) != 3:
            raise ValueError(f"launch {key}: expected 3 comma separated expressions, got {values[key]!r}")
        dims[key] = tuple(_compile_expr(e, key) for e in exprs)
    if values["subgroup_size"] not in ("8", "16", "32"):
        raise ValueError(f"launch subgroup_size: must be 8, 16 or 32, got {values['subgroup_size']!r}")
    options = tuple(values["build_options"].split())
    for option in options:
        if not _BUILD_OPTION.match(option):
            raise ValueError(f"launch build_options: {option!r} is not allowed (only -cl-* options and -DNAME[=VALUE])")
    return LaunchConfig(kernel, args, dims["gws"], dims["lws"], int(values["subgroup_size"]), values["grf_256"].strip().lower() in ("1", "true", "yes"), options)


def _build_program(ctx: cl.Context, source: str, defines: dict, launch: LaunchConfig) -> cl.Program:
    options = ["-cl-mad-enable"] + [f"-D{k}={v}" for k, v in defines.items()] + list(launch.build_options)
    if launch.grf_256:
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
    output: cl.Buffer


class AttentionOp:
    """Launches the attention kernel as described by the LAUNCH block of its source (by default one work-group per (token, kv head))."""

    def __init__(self, queue: cl.CommandQueue, filename: str, cfg: KvConfig):
        source = (HERE / filename).read_text()
        self.queue = queue
        self.cfg = cfg
        self.launch = parse_launch(source)
        program = _build_program(queue.context, source, _defines(cfg, self.launch.subgroup_size), self.launch)
        try:
            self._kernel = getattr(program, self.launch.kernel)
        except AttributeError as e:
            raise ValueError(f"launch kernel: entry point {self.launch.kernel!r} not found in {filename}") from e
        self._plans: dict = {}

    def _plan(self, inp: AttentionInputs):
        """(gws, lws, [(is_buffer, buffer attribute | scalar)]) for the token / sequence counts of `inp`."""
        key = (inp.num_tokens, inp.num_seqs)
        plan = self._plans.get(key)
        if plan is None:
            env = dict(
                tokens=max(1, inp.num_tokens),
                num_seqs=inp.num_seqs,
                sg=self.launch.subgroup_size,
                heads=NUM_HEADS,
                kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                topk=TOPK,
                block_size=BLOCK_SIZE,
                page_size=PAGE_SIZE,
            )
            namespace = {"__builtins__": {}, **_DISPATCH_FUNCS}
            gws = tuple(int(eval(code, namespace, env)) for code in self.launch.gws)
            lws = tuple(int(eval(code, namespace, env)) for code in self.launch.lws)
            if min(gws + lws) <= 0 or any(g % l for g, l in zip(gws, lws)):
                raise ValueError(f"launch: gws {gws} must be positive multiples of lws {lws}")
            scalars = {
                "num_tokens": U32(inp.num_tokens),
                "num_seqs": U32(inp.num_seqs),
                "q_base": U32(0),
                "q_pitch": U32(Q_PITCH),
                "scale": np.float32(SCALE),
            }
            args = [(True, a) if a in BUFFER_ARGS else (False, scalars[a]) for a in self.launch.args]
            plan = self._plans[key] = (gws, lws, args)
        return plan

    def __call__(self, inp: AttentionInputs):
        gws, lws, args = self._plan(inp)
        return self._kernel(self.queue, gws, lws, *[getattr(inp, v) if is_buffer else v for is_buffer, v in args])


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


def make_inputs(queue: cl.CommandQueue, case: Case, cfg: KvConfig, poison_output: bool = False) -> AttentionInputs:
    ctx = queue.context
    mf = cl.mem_flags

    def ro(arr):
        return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr))

    key_logical, value_logical = case.logical_caches(cfg)
    out_elems = case.num_tokens * Q_PITCH
    if poison_output:
        output = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.full(out_elems, 0xFFFF, np.uint16))  # NaN: unwritten outputs show up
    else:
        output = cl.Buffer(ctx, mf.READ_WRITE, size=out_elems * 2)
    return AttentionInputs(
        num_tokens=case.num_tokens,
        num_seqs=case.num_seqs,
        query=ro(case.query),
        key_cache=ro(case.physical(key_logical)),
        value_cache=ro(case.physical(value_logical)),
        past_lens=ro(case.past),
        subsequence_begins=ro(case.subsequence_begins),
        block_indices=ro(case.block_indices),
        block_indices_begins=ro(case.block_indices_begins),
        selected_blocks=ro(case.selected_blocks),
        selected_counts=ro(case.selected_counts),
        output=output,
    )


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
    # sequence start, history, a decode-like single token and a tiny sequence; every block is selected
    "multi": [SeqSpec(0, 300), SeqSpec(130, 200), SeqSpec(37, 1), SeqSpec(0, 5)],
    # more complete blocks than TOPK: the strided top-k selection is capped
    "prefill_2560": [SeqSpec(0, 2560)],
    "prefill_8K": [SeqSpec(0, 8192)],
}
CPU_CASES = [(cfg, "multi") for cfg in ALL_CONFIGS] + [(cfg, "prefill_2560") for cfg in BENCH_CONFIGS]
REFERENCE_CASES = [(cfg, "prefill_8K") for cfg in BENCH_CONFIGS]


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
class TestQsaSparseAttentionGqaDpas(TestBase):
    """SparsePA selected-block attention over a paged f16 / u8 / u4 KV cache (prefill)."""

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
        inp = make_inputs(ocl_queue, case, cfg, poison_output=True)
        build_op(ocl_queue, kernel_file, cfg)(inp)
        ocl_queue.finish()
        got = read_output(ocl_queue, inp)

        key_logical, value_logical = case.logical_caches(cfg)
        expected = reference_attention(case, decode_cache(key_logical, cfg.key), decode_cache(value_logical, cfg.value))
        assert_close("attention output vs CPU model", got, expected, atol=1.5e-2, rtol=1.5e-2)

    @pytest.mark.parametrize("cfg, scenario", REFERENCE_CASES, ids=[f"{c.name}-{s}" for c, s in REFERENCE_CASES])
    def test_correctness_wrt_reference(self, ocl_queue, kernel_file, cfg, scenario):
        """Attention output vs reference.cl on the same inputs."""
        case = _case(scenario)
        inp = make_inputs(ocl_queue, case, cfg, poison_output=True)
        build_op(ocl_queue, kernel_file, cfg)(inp)
        ocl_queue.finish()
        got = read_output(ocl_queue, inp)

        ref_inp = make_inputs(ocl_queue, case, cfg, poison_output=True)
        build_op(ocl_queue, REFERENCE_FILE, cfg)(ref_inp)
        ocl_queue.finish()
        assert_close("attention output vs reference.cl", got, read_output(ocl_queue, ref_inp), atol=1e-2, rtol=1e-2)

    @pytest.mark.performance
    @pytest.mark.parametrize("cfg", BENCH_CONFIGS, ids=[c.name for c in BENCH_CONFIGS])
    @pytest.mark.parametrize("length", CONTEXT_LENGTHS, ids=[f"{n // 1024}K" for n in CONTEXT_LENGTHS])
    def test_benchmark_prefill(self, ocl_queue, kernel_file, measure_runtime, cfg, length):
        """Prefill of one sequence of `length` tokens (past = 0): every token attends to up to 512 selected blocks + its tail."""
        op = build_op(ocl_queue, kernel_file, cfg)
        inp = make_inputs(ocl_queue, Case([SeqSpec(0, length)], seed=length), cfg)
        measure_runtime(
            op,
            args=(inp,),
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestQsaSparseAttentionGqaDpas()
    task.build(TestBase.get_machine_gpu_arch())
