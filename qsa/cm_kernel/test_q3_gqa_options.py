"""CPU-only GQA option/dispatch contracts; never import or compile real clops.

Real wrappers, Q2 planning and Pipeline wiring run against CPU tensors and a
recording compiler. These tests do not emulate GPU arithmetic or memory guards.
"""
from dataclasses import replace
from itertools import product
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import qsa_attention_gqa as GQA
import qsa_config
import qsa_harness as H
import qsa_score_chunked as Q2
import test_qsa_pipeline as T


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data).copy()

    def numpy(self):
        return self.data.copy()


class CPUCase(unittest.TestCase):
    def setUp(self):
        self.cfg = qsa_config.load()
        self.events = []
        self.cl = SimpleNamespace(tensor=mock.Mock(side_effect=FakeTensor))
        # Unexpected device discovery, compilation or synchronization must fail,
        # rather than silently succeeding through an unrestricted Mock.
        for name in ("kernels", "finish", "dev_info", "profiling"):
            setattr(self.cl, name, mock.Mock(side_effect=AssertionError("GPU API forbidden: " + name)))
        module = ModuleType("clops")
        setattr(module, "cl", self.cl)
        self.enterContext(mock.patch.dict(sys.modules, {"clops": module}))
        self.enterContext(mock.patch.dict(Q2._KERNELS, {}, clear=True))
        self.enterContext(mock.patch.dict(T.COMPILED, {}, clear=True))
        self.enterContext(mock.patch.dict(T.SCORE_OPTIONS, {}, clear=True))

        def build(name, jit):
            handle = mock.Mock(spec=["enqueue"])
            handle.kernel_file, handle.jit = name, jit.copy()
            handle.enqueue.side_effect = lambda *args, **kw: self.events.append((args, kw))
            return handle

        self.builder = self.enterContext(mock.patch.object(H, "build_kernel", side_effect=build))
        self.inputs = tuple(FakeTensor([i]) for i in range(9))
        self.output, self.token_map = FakeTensor([0]), FakeTensor([[0, 0]])

    def launch(self, attention, **options):
        c = attention.cfg
        layout = dict(num_tokens=3, num_seqs=2, q_base=16,
                      q_pitch=c.heads_num * (c.k_head_size + c.v_head_size),
                      q_head_stride=c.k_head_size + c.v_head_size,
                      gate_delta=c.k_head_size, token_map=self.token_map)
        layout.update(options)
        attention.enqueue(self.inputs, self.output, **layout)

    def pipeline(self, *, phase="prefill", optimized=True, attention_options=None,
                 score_options=None, seqs=None):
        # Small fake fixture: production buffer/planner/dispatch code is real;
        # tensor contents are deliberately not an attention correctness oracle.
        c = self.cfg
        seqs = seqs if seqs is not None else [(0, 3 if phase == "prefill" else 1)]
        fixture = SimpleNamespace(cfg=c, seed=71,
            constants=[FakeTensor([0]) for _ in range(4)],
            batch=lambda calls: H.make_batch(calls, c.pa_block_size),
            data=[dict(x=np.zeros((p + q, c.hidden_size), np.float16),
                       key=np.zeros((p + q, c.kv_heads_num, c.k_head_size), np.float16),
                       value=np.zeros((p + q, c.kv_heads_num, c.v_head_size), np.float16))
                  for p, q in seqs])
        state = T.State({name: np.zeros(1, np.float16) for name in ("kc", "vc", "raw", "summary")})
        return T.Pipeline(fixture, seqs, state, optimized=optimized, phase=phase,
                          guard=True, attention_options=attention_options,
                          score_options=score_options)


class GQAOptionsTests(CPUCase):
    def test_flags_require_builtin_bool(self):
        for flag in ("enabled", "head_major", "vector_meta"):
            for value in (0, 1, -1, 0.0, 1.0, None, "true", [], {}, np.bool_(True), np.int64(1)):
                with self.subTest(flag=flag, value=value):
                    options = dict(enabled=True)
                    options[flag] = value
                    with self.assertRaisesRegex(ValueError, flag + " must be bool"):
                        GQA.Q3GQAAttention(self.cfg, **options)
        self.builder.assert_not_called()

    def test_fallback_choices_are_explicit(self):
        for value in (None, "", "auto", "SCALAR", True, 1):
            with self.subTest(fallback=value), self.assertRaisesRegex(ValueError, "fallback"):
                GQA.Q3GQAAttention(self.cfg, fallback=value)
        self.builder.assert_not_called()

    def test_default_remains_real_scalar_wrapper_with_fake_compiler(self):
        attention = GQA.Q3GQAAttention(self.cfg)
        self.assertFalse(attention.enabled)
        self.assertEqual(attention.reason, "not opted in")
        self.assertIsInstance(attention.scalar, GQA.Q3Attention)
        self.assertIsNone(attention.scalar.partitions)
        self.assertEqual(attention.scalar.scratch_elements(3), (0, 0))
        self.assertEqual(attention.scalar.kernel.kernel_file, "qsa_sparse_attention.cm")
        self.assertNotIn("QSA_GQA_HEAD_MAJOR", attention.scalar.kernel.jit)
        self.assertNotIn("QSA_GQA_VECTOR_META", attention.scalar.kernel.jit)
        self.launch(attention, token_map=None, guard_items=True, q_base=2)
        attention.scalar.kernel.enqueue.assert_called_once_with(
            "qsa_sparse_attention", [25, 4, 1], [1, 1, 1], *self.inputs,
            self.output, 3, 2, 2, 12288, 512, 256)
        self.assertEqual(self.builder.call_count, 1)

    def test_tuning_flags_do_not_enable_gqa(self):
        for head, vector in product((False, True), repeat=2):
            attention = GQA.Q3GQAAttention(self.cfg, head_major=head, vector_meta=vector)
            self.assertFalse(attention.enabled)
            self.assertEqual(attention.scalar.name, "qsa_sparse_attention")

    def test_all_jit_flag_combinations_are_distinct(self):
        keys = set()
        for head, vector in product((False, True), repeat=2):
            attention = GQA.Q3GQAAttention(self.cfg, enabled=True, head_major=head, vector_meta=vector)
            self.assertTrue(attention.enabled)
            self.assertIsNone(attention.reason)
            self.assertIsNone(attention.scalar)
            jit = attention.kernel.jit
            self.assertEqual(jit["QSA_GQA_HEAD_MAJOR"], int(head))
            self.assertEqual(jit["QSA_GQA_VECTOR_META"], int(vector))
            self.assertIs(type(jit["QSA_GQA_HEAD_MAJOR"]), int)
            self.assertIs(type(jit["QSA_GQA_VECTOR_META"]), int)
            self.assertEqual(attention.kernel.kernel_file, attention.name + ".cm")
            keys.add(tuple(jit.items()))
        self.assertEqual(len(keys), 4)

    def test_unsupported_geometry_rejected_before_compilation(self):
        cases = ({"heads_num": 0}, {"kv_heads_num": 0}, {"heads_num": 23},
                 {"heads_num": 34}, {"k_head_size": 0}, {"v_head_size": 64},
                 {"k_head_size": 384}, {"compress_ratio": 1}, {"pa_block_size": 0},
                 {"pa_block_size": 6}, {"block_topk": -1}, {"is_causal": False})
        for changes in cases:
            with self.subTest(changes=changes):
                cfg = replace(self.cfg, **changes)
                self.assertIsNotNone(GQA.Q3GQAAttention.unsupported_reason(cfg))
                with self.assertRaises(ValueError):
                    GQA.Q3GQAAttention(cfg, enabled=True)
        self.builder.assert_not_called()

    def test_supported_geometry_boundaries(self):
        for rep, dim, ratio, k, gate in product((1, 16), (128, 256), (2, 4, 8, 16),
                                               (0, 1, 512), (False, True)):
            cfg = replace(self.cfg, heads_num=rep * 2, k_head_size=dim, v_head_size=dim,
                          compress_ratio=ratio, block_topk=k, has_output_gate=gate)
            self.assertIsNone(GQA.Q3GQAAttention.unsupported_reason(cfg))

    def test_explicit_fallback_uses_scalar_and_forwards_layout(self):
        cfg = replace(self.cfg, k_head_size=64)
        attention = GQA.Q3GQAAttention(cfg, enabled=True, fallback="scalar",
                                       head_major=True, vector_meta=True)
        self.assertFalse(attention.enabled)
        self.assertEqual(attention.reason, GQA.Q3GQAAttention.unsupported_reason(cfg))
        self.launch(attention, token_map=None, guard_items=True, q_base=2)
        attention.scalar.kernel.enqueue.assert_called_once_with(
            "qsa_sparse_attention", [25, 4, 1], [1, 1, 1], *self.inputs,
            self.output, 3, 2, 2, 7680, 320, 64)

    def test_fallback_does_not_accept_noncausal_or_invalid_scalar_geometry(self):
        for enabled, fallback in product((False, True), ("error", "scalar")):
            with self.assertRaisesRegex(ValueError, "noncausal"):
                GQA.Q3GQAAttention(replace(self.cfg, is_causal=False),
                                   enabled=enabled, fallback=fallback)
        with self.assertRaisesRegex(ValueError, "unsupported Q3"):
            GQA.Q3GQAAttention(replace(self.cfg, heads_num=23), enabled=True, fallback="scalar")
        self.builder.assert_not_called()


class GQADispatchTests(CPUCase):
    def test_baseline_head_major_grids_guards_and_unchanged_abi(self):
        for head, vector, guard in product((False, True), repeat=3):
            with self.subTest(head=head, vector=vector, guard=guard):
                attention = GQA.Q3GQAAttention(self.cfg, enabled=True,
                                               head_major=head, vector_meta=vector)
                self.launch(attention, guard_items=guard)
                g = int(guard)
                grid = [(3 + g) * 4, 2 + g, 1] if head else [(2 + g) * 4, 3 + g, 1]
                attention.kernel.enqueue.assert_called_once_with(
                    attention.name, grid, [4, 1, 1], *self.inputs, self.token_map,
                    self.output, 3, 16, 12288, 512, 256)

    def test_enqueue_has_no_allocation_compilation_or_synchronization(self):
        attention = GQA.Q3GQAAttention(self.cfg, enabled=True)
        self.builder.reset_mock()
        for _ in range(2):
            self.launch(attention)
        self.builder.assert_not_called()
        for api in vars(self.cl).values():
            api.assert_not_called()

    def test_empty_batch_is_noop_in_both_paths(self):
        for enabled in (False, True):
            attention = GQA.Q3GQAAttention(self.cfg, enabled=enabled)
            attention.enqueue(None, None, 0, 0, 0, 0, 0, 0, guard_items=True)
        self.assertEqual(self.events, [])

    def test_layout_requires_builtin_uint32_including_empty_batches(self):
        fields = ("num_tokens", "num_seqs", "q_base", "q_pitch", "q_head_stride", "gate_delta")
        for enabled in (False, True):
            attention = GQA.Q3GQAAttention(self.cfg, enabled=enabled)
            for field, bad in product(fields, (-1, 2**32, True, 1.0, np.int64(1), None)):
                with self.subTest(enabled=enabled, field=field, value=bad):
                    with self.assertRaisesRegex(ValueError, "uint32 integers"):
                        self.launch(attention, **{field: bad})
            with self.assertRaisesRegex(ValueError, "uint32 integers"):
                self.launch(attention, num_tokens=0, q_base=-1)
        self.assertEqual(self.events, [])

    def test_guard_requires_builtin_bool_even_for_empty_batch(self):
        for enabled, nt, value in product((False, True), (0, 3), (0, 1, None, "true", np.bool_(True))):
            attention = GQA.Q3GQAAttention(self.cfg, enabled=enabled)
            with self.assertRaisesRegex(ValueError, "guard_items must be bool"):
                self.launch(attention, num_tokens=nt, guard_items=value)
        self.assertEqual(self.events, [])

    def test_missing_inputs_sequences_output_and_token_map_rejected(self):
        attention = GQA.Q3GQAAttention(self.cfg, enabled=True)
        for inputs, output, ns in ((None, self.output, 1), ((), self.output, 1),
                                  (self.inputs[:-1], self.output, 1),
                                  (self.inputs + (self.output,), self.output, 1),
                                  (self.inputs, None, 1), (self.inputs, self.output, 0)):
            with self.assertRaisesRegex(ValueError, "nine inputs"):
                attention.enqueue(inputs, output, 3, ns, 0, 12288, 512, 256, token_map=self.token_map)
        with self.assertRaisesRegex(ValueError, "map required"):
            self.launch(attention, token_map=None)
        self.assertEqual(self.events, [])

    def test_alignment_overlap_and_byte_range_guards(self):
        attention = GQA.Q3GQAAttention(self.cfg, enabled=True)
        for changes, message in (({"q_base": 2}, "aligned"), ({"q_pitch": 12290}, "aligned"),
                                 ({"q_head_stride": 514}, "aligned"), ({"gate_delta": 257}, "aligned"),
                                 ({"q_head_stride": 496}, "overlaps"), ({"q_pitch": 12272}, "overlaps"),
                                 ({"gate_delta": 258}, "overlaps"),
                                 ({"q_base": 2**31 - 12288, "num_tokens": 1}, "byte-offset"),
                                 ({"num_tokens": 174763}, "byte-offset")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.launch(attention, **changes)
        self.assertEqual(self.events, [])
        self.launch(attention, num_tokens=1, q_base=2**31 - 12288 - 16)
        self.assertEqual(len(self.events), 1)

    def test_ungated_layout_does_not_require_gate_storage(self):
        attention = GQA.Q3GQAAttention(replace(self.cfg, has_output_gate=False), enabled=True)
        self.launch(attention, q_head_stride=256, q_pitch=6144, gate_delta=512)
        attention.kernel.enqueue.assert_called_once()


class PipelineIntegrationTests(CPUCase):
    def test_prefill_options_labels_and_dispatch_match(self):
        for head, vector in product((False, True), repeat=2):
            p = self.pipeline(attention_options=dict(head_major=head, vector_meta=vector),
                              score_options={"finalizer": "fast-wg16-cached", "row_cap": 1})
            expected = "Q3-gqa-W4" + ("-head-major" if head else "") + ("-vector-meta" if vector else "")
            self.assertEqual(p.labels[-1], expected)
            self.assertEqual(p.q2_labels, ["Q2-dpas", "Q2-fast-WG16-cached"])
            self.assertEqual(len(p.q2.plan.chunks), 3)
            self.assertEqual(p.q2.cache_capacity, 64)
            np.testing.assert_array_equal(p.token_map.numpy(), [[0, 0], [0, 1], [0, 2]])
            for guard in (True, False):
                p.guard = guard
                self.events.clear()
                builds, tensors = self.builder.call_count, self.cl.tensor.call_count
                p.enqueue()
                self.assertEqual((self.builder.call_count, self.cl.tensor.call_count), (builds, tensors))
                self.assertEqual([args[0] for args, _ in self.events],
                    ["qsa_kv_cache_update", "qsa_project_dpas", "qsa_prepare_projected"] +
                    ["qsa_score_tile_dpas", "qsa_topk_radix_wg"] * 3 + [p.q3.name])
                self.assertEqual(len(self.events), len(p.labels))
                g = int(guard)
                grid = [(3 + g) * 4, 2 + g, 1] if head else [(2 + g) * 4, 3 + g, 1]
                p.q3.kernel.enqueue.assert_called_with(p.q3.name, grid, [4, 1, 1],
                    p.qdev, p.state.dev("kc"), p.state.dev("vc"), *p.meta,
                    p.sel.device, p.count.device, p.token_map, p.out.device, 3, 0, p.pitch, p.qhs, 256)

    def test_pipeline_cache_distinguishes_options_and_canonicalizes_order(self):
        variants = []
        for head, vector in product((False, True), repeat=2):
            p = self.pipeline(attention_options=dict(head_major=head, vector_meta=vector))
            repeat = self.pipeline(attention_options=dict(vector_meta=vector, head_major=head))
            self.assertIs(p.q3, repeat.q3)
            self.assertIs(p.q2.scorer, repeat.q2.scorer)
            variants.append(p.q3)
        self.assertEqual(len({id(q3) for q3 in variants}), 4)
        gqa_builds = [call for call in self.builder.call_args_list
                      if call.args[0] == "qsa_sparse_attention_gqa_dpas.cm"]
        self.assertEqual(len(gqa_builds), 4)

    def test_pipeline_cache_distinguishes_configuration(self):
        baseline = self.pipeline()
        self.cfg = replace(self.cfg, block_topk=1)
        changed = self.pipeline()
        self.assertIsNot(baseline.q3, changed.q3)
        self.assertEqual(changed.q3.kernel.jit["QSA_BLOCK_TOPK"], 1)

    def test_omitted_options_retain_token_major_scalar_metadata_gqa(self):
        p = self.pipeline()
        repeat = self.pipeline(attention_options={})
        self.assertIs(p.q3, repeat.q3)
        self.assertTrue(p.q3.enabled)
        self.assertFalse(p.q3.head_major)
        self.assertFalse(p.q3.vector_meta)
        self.assertEqual(p.labels[-1], "Q3-gqa-W4")

    def test_unknown_options_and_nonboolean_cache_misses_rejected(self):
        for options in ({"auto": True}, {"enabled": True}, {"fallback": "scalar"},
                        {"head_major": 1}, {"vector_meta": np.bool_(True)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.pipeline(attention_options=options)

    def test_warm_pipeline_cache_must_not_bypass_strict_bool_validation(self):
        # True == 1 in tuple keys: validate before the compiler-cache lookup.
        rejected = []
        for flag in ("head_major", "vector_meta"):
            self.pipeline(attention_options={flag: True})
            try:
                self.pipeline(attention_options={flag: 1})
            except ValueError:
                rejected.append(flag)
        self.assertEqual(rejected, ["head_major", "vector_meta"])

    def test_existing_unoptimized_prefill_and_decode_routes_unchanged(self):
        for phase, optimized in (("prefill", False), ("decode", False), ("decode", True)):
            with self.subTest(phase=phase, optimized=optimized):
                p = self.pipeline(phase=phase, optimized=optimized,
                                  attention_options={"head_major": True, "vector_meta": True})
                self.assertNotIsInstance(p.q3, GQA.Q3GQAAttention)
                self.events.clear()
                p.enqueue()
                expected = "qsa_sparse_attention_dpas" if phase == "prefill" else "qsa_sparse_attention"
                self.assertEqual(self.events[-1][0][0], expected)
                self.assertEqual(len(self.events), len(p.labels))
                self.assertNotIn("head-major", p.labels[-1])
                self.assertNotIn("vector-meta", p.labels[-1])

    def test_empty_pipeline_dispatches_nothing(self):
        p = self.pipeline(seqs=[(0, 0), (7, 0)],
                          attention_options={"head_major": True, "vector_meta": True})
        p.enqueue()
        self.assertEqual(self.events, [])
        self.assertEqual(p.q2.plan.chunks, ())


if __name__ == "__main__":
    unittest.main()