"""CPU-only explicit finalizer selection contract; no GPU dependency."""
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import qsa_config
import qsa_harness as H
import qsa_score_chunked as Q2
from qsa_score_chunked import finalizer_spec, topk_cache_capacity
from validate_topk_fast import Finalizers, oracle


class FinalizerOptionsTests(unittest.TestCase):
    def test_defaults_and_legacy(self):
        for coop in (False, True):
            for wg in (8, 16):
                expected = ("qsa_topk_cooperative", wg) if coop else ("qsa_topk_finalization", 1)
                for mode in (None, "legacy"):
                    self.assertEqual(finalizer_spec(mode, cooperative=coop, wg=wg), expected)

    def test_explicit_modes(self):
        for coop in (False, True):
            for wg in (8, 16):
                for mode, name, width in (("scalar", "qsa_topk_finalization", 1),
                    ("cooperative", "qsa_topk_cooperative", wg), ("fast", "qsa_topk_radix_fast", 1),
                    ("fast-wg", "qsa_topk_radix_wg", wg), ("fast-wg8", "qsa_topk_radix_wg", 8),
                    ("fast-wg16", "qsa_topk_radix_wg", 16),
                    ("fast-wg16-cached", "qsa_topk_radix_wg", 16)):
                    self.assertEqual(finalizer_spec(mode, cooperative=coop, wg=wg), (name, width))

    def test_reject_unvalidated_automatic_policy(self):
        for mode in ("auto", "auto-fast", "", "fast-wg32", "fast-wg8-cached", "fast-cached"):
            with self.assertRaises(ValueError):
                finalizer_spec(mode)
        with self.assertRaises(ValueError):
            finalizer_spec("fast-wg16", wg=4)
        with self.assertRaises(ValueError):
            finalizer_spec("fast-wg16-cached", wg=4)

    def test_cache_capacity_rounding_and_ceiling(self):
        for mb, expected in ((8, 64), (64, 64), (1024, 64), (1025, 128), (1032, 128),
                             (16384, 1024), (16392, 1088), (32768, 2048),
                             (32769, 2048), (32776, 2048), (131072, 2048)):
            with self.subTest(max_blocks=mb):
                self.assertEqual(topk_cache_capacity(mb), expected)
        # Every padded width fits the full aligned chunk until the hard ceiling.
        for mb in range(8, 32769, 8):
            capacity = topk_cache_capacity(mb)
            self.assertEqual(capacity % 64, 0)
            self.assertLessEqual(capacity, 2048)
            self.assertGreaterEqual(capacity * 16, mb)
            self.assertLess((capacity - 64) * 16, mb)
        self.assertEqual(topk_cache_capacity(np.int64(1032)), 128)
        for mb in (0, -1):
            with self.assertRaises(ValueError):
                topk_cache_capacity(mb)
        with self.assertRaises(TypeError):
            topk_cache_capacity(1024.5)

    def test_worker_cache_fit_boundary(self):
        # Independent host arithmetic for the documented row-uniform decision.
        for n, expected_chunk, fits in ((32768, 2048, True), (32769, 2112, False)):
            chunk = ((n + 16 * 64 - 1) // (16 * 64)) * 64
            capacity = topk_cache_capacity(H.ceil8(n) + 8)
            self.assertEqual(chunk, expected_chunk)
            self.assertEqual(chunk <= capacity, fits)
        for n in (1, 63, 64, 65, 513, 1024, 1025, 16384, 32768):
            capacity = topk_cache_capacity(H.ceil8(n))
            chunk = ((n + 1023) // 1024) * 64
            visited = []
            for worker in range(16):
                begin, end = worker * chunk, min((worker + 1) * chunk, n)
                for i in range(begin, end, 64):
                    self.assertLessEqual(i - begin + 64, capacity)
                    visited.extend(range(i, min(i + 64, end)))
            self.assertEqual(visited, list(range(n)))


class HostDispatchTests(unittest.TestCase):
    """Real construction/JIT/dispatch logic; tensors and compiler are CPU fakes."""
    def setUp(self):
        def tensor(a):
            data = a.copy()
            return SimpleNamespace(numpy=lambda: data.copy())

        # No clops import, device discovery, compilation or GPU enqueue occurs.
        self.enterContext(mock.patch.dict(sys.modules, {"clops": SimpleNamespace(
            cl=SimpleNamespace(tensor=tensor))}))
        self.enterContext(mock.patch.dict(Q2._KERNELS, {}, clear=True))
        import validate_topk_fast as V
        self.enterContext(mock.patch.dict(V.CACHE, {}, clear=True))

        def build(name, jit):
            handle = mock.Mock()
            handle.kernel_file, handle.jit = name, jit.copy()
            return handle

        self.builder = self.enterContext(mock.patch.object(H, "build_kernel", side_effect=build))

    def make(self, mb=1032, *, k=512, begins=(0, 0, 2, 5), **options):
        batch = SimpleNamespace(subsequence_begins=begins, num_seqs=len(begins) - 1)
        return Q2.ChunkedQ2(qsa_config.load(block_topk=k), batch, mb, **options)

    def test_existing_modes_never_enable_cache(self):
        for mode in (None, *Q2.FINALIZER_CHOICES):
            for coop in (False, True):
                for wg in (8, 16):
                    with self.subTest(mode=mode, cooperative=coop, wg=wg):
                        q2 = self.make(finalizer=mode, cooperative=coop, wg=wg)
                        name, width = finalizer_spec(mode, cooperative=coop, wg=wg)
                        self.assertEqual((q2.final_name, q2.wg), (name, width))
                        self.assertNotIn("QSA_TOPK_CACHE_CAPACITY", q2.scorer.jit)
                        if mode == "fast-wg16-cached":
                            self.assertEqual(q2.cache_capacity, 128)
                            self.assertEqual(q2.finalizer.jit["QSA_TOPK_CACHE_CAPACITY"], 128)
                        else:
                            self.assertEqual(q2.cache_capacity, 0)
                            self.assertNotIn("QSA_TOPK_CACHE_CAPACITY", q2.finalizer.jit)

    def test_build_cache_separates_capacity_but_reuses_scorer(self):
        base = self.make(finalizer="fast-wg16", wg=8)
        small = self.make(8, finalizer="fast-wg16-cached", wg=8)
        same = self.make(1024, finalizer="fast-wg16-cached", wg=8)
        bigger = self.make(1032, finalizer="fast-wg16-cached", wg=8)
        full = self.make(32768, finalizer="fast-wg16-cached", wg=8)
        overflow = self.make(32776, finalizer="fast-wg16-cached", wg=8)
        self.assertIsNot(base.finalizer, small.finalizer)
        self.assertIs(small.finalizer, same.finalizer)
        self.assertIsNot(small.finalizer, bigger.finalizer)
        self.assertIsNot(bigger.finalizer, full.finalizer)
        self.assertIs(full.finalizer, overflow.finalizer)
        for q2 in (small, same, bigger, full, overflow):
            self.assertIs(q2.scorer, base.scorer)
            self.assertEqual(q2.finalizer.jit["QSA_TOPK_WG"], 16)
        self.assertEqual([q.cache_capacity for q in (small, same, bigger, full, overflow)],
                         [64, 64, 128, 2048, 2048])
        self.assertEqual(self.builder.call_count, 5)  # scorer + four finalizer JITs

    def test_chunked_and_unchunked_dispatch_preserves_abi_and_guards(self):
        for backend in ("partition", "dpas"):
            for unchunked in (False, True):
                for bypass in (False, True):
                    with self.subTest(backend=backend, unchunked=unchunked, bypass=bypass):
                        q2 = self.make(finalizer="fast-wg16-cached", wg=8, backend=backend,
                                       unchunked=unchunked, row_cap=None if unchunked else 2,
                                       dense_bypass=bypass, partition_blocks=32)
                        q2.scorer.reset_mock()
                        q2.finalizer.reset_mock()
                        for kernel in (q2.scorer, q2.finalizer):
                            self.assertEqual(kernel.jit["QSA_SCORE_ROW_CHUNK"], int(not unchunked))
                            self.assertEqual(kernel.jit["QSA_DENSE_BYPASS"], int(bypass))
                            self.assertEqual(kernel.jit["QSA_DENSE_SCORE_BYPASS"], int(bypass))
                            self.assertEqual(kernel.jit["QSA_PARTITION_BLOCKS"], 32)
                        order = mock.Mock()
                        order.attach_mock(q2.scorer, "score")
                        order.attach_mock(q2.finalizer, "finalize")
                        inputs = ("iq", "summary", "past", "begins", "pages", "page_begins")
                        q2.enqueue(inputs, "selected", "counts", guard_groups=2)
                        self.assertEqual([call[0] for call in order.mock_calls],
                                         ["score.enqueue", "finalize.enqueue"] * len(q2.plan.chunks))
                        for chunk, call in zip(q2.plan.chunks, q2.finalizer.enqueue.call_args_list):
                            args = call.args
                            self.assertEqual(args[:3], ("qsa_topk_radix_wg",
                                [(chunk.row_end - chunk.row_begin + 2) * 16, 1, 1], [16, 1, 1]))
                            self.assertIs(args[3], q2.scores)
                            self.assertEqual(args[4:8], ("past", "begins", "selected", "counts"))
                            self.assertEqual(args[8:], (5, 3, 1032) + (() if unchunked else
                                             (chunk.row_begin, chunk.row_end)))
                        # Scratch remains row-local, including its checked suffix.
                        self.assertEqual(q2.read().shape, (5 if unchunked else 2, 1032))

    def test_k0_dense_shapes_and_empty_dispatch(self):
        for k in (0, 1, 512):
            q2 = self.make(8, k=k, finalizer="fast-wg16-cached")
            self.assertEqual(q2.finalizer.jit["QSA_BLOCK_TOPK"], k)
            self.assertEqual(q2.cache_capacity, 64)  # capacity is not a function of K
            empty = self.make(8, k=k, begins=(0, 0, 0), finalizer="fast-wg16-cached")
            empty.scorer.reset_mock()
            empty.finalizer.reset_mock()
            empty.enqueue([], None, None, guard_groups=2)
            empty.scorer.enqueue.assert_not_called()
            empty.finalizer.enqueue.assert_not_called()
            self.assertEqual(empty.read().shape, (0, 8))

    def test_remote_validator_pins_cached_wg_and_specializes_capacity(self):
        for n, capacity in ((1024, 64), (1032, 128), (32768, 2048), (32769, 2048)):
            f = Finalizers(np.zeros((1, n), np.float32), [(4 * n - 1, 1)], wg=8,
                           variants=("fast-wg", "fast-wg16-cached"))
            self.assertEqual(f.widths, {"fast-wg": 8, "fast-wg16-cached": 16})
            self.assertEqual(f.key_capacities["fast-wg16-cached"], capacity)
            for mode in f.variants:
                f.enqueue(mode, guard=2)
                kernel = f.kernels[mode]
                width = f.widths[mode]
                self.assertEqual(kernel.jit["QSA_TOPK_WG"], width)
                self.assertEqual(kernel.enqueue.call_args.args[1:3], ([3 * width, 1, 1], [width, 1, 1]))
                self.assertEqual(kernel.jit.get("QSA_TOPK_CACHE_CAPACITY", 0), f.key_capacities[mode])
        f = Finalizers(np.zeros((1, 1025), np.float32), [(4099, 1)], key_capacity=64,
                       variants=("fast-wg16-cached",))
        self.assertEqual(f.kernels["fast-wg16-cached"].jit["QSA_TOPK_CACHE_CAPACITY"], 64)


class OrderedKeysOracleTests(unittest.TestCase):
    def test_signed_zeros_subnormals_and_extremes(self):
        bits = np.array([[0x80000000, 0, 0x80000001, 1, 0xbf800000, 0x3f800000,
                          0xff7fffff, 0x7f7fffff, 0, 0x80000000]], np.uint32)
        scores = bits.view(np.float32)
        # Explicit descending order, including +0 > -0 and ascending tied IDs.
        ranked = [7, 5, 3, 1, 8, 0, 9, 2, 4, 6]
        for k in range(11):
            selected, counts = oracle(scores, np.array([10]), k)
            np.testing.assert_array_equal(selected[0], sorted(ranked[:k]))
            np.testing.assert_array_equal(counts, [k])

    def test_independent_numeric_sort_at_capacity_boundary(self):
        rng = np.random.default_rng(371)
        for n in (32768, 32769):
            scores = rng.choice(np.array([-2, -1, -0.0, 0.0, 1, 2], np.float32), (1, n))
            # Numeric Python sort with an explicit zero sign tie key: independent
            # of the uint bit transform used by oracle() and the CM kernel.
            ranked = sorted(range(n), key=lambda i: (-float(scores[0, i]),
                            bool(np.signbit(scores[0, i])) if scores[0, i] == 0 else False, i))
            for k in (1, 512, 2049, n - 1):
                selected, counts = oracle(scores, np.array([n]), k)
                np.testing.assert_array_equal(selected[0], sorted(ranked[:k]))
                np.testing.assert_array_equal(counts, [k])

    def test_poison_bypasses_and_sparse_nonfinite_rejection(self):
        poison = np.full((3, 8), np.nan, np.float32)
        selected, counts = oracle(poison, np.array([0, 1, 8]), 0)
        self.assertEqual(selected.shape, (3, 0))
        np.testing.assert_array_equal(counts, 0)
        selected, counts = oracle(poison, np.array([0, 1, 8]), 8)
        np.testing.assert_array_equal(selected[0], -1)
        np.testing.assert_array_equal(selected[1], [0, -1, -1, -1, -1, -1, -1, -1])
        np.testing.assert_array_equal(selected[2], np.arange(8))
        np.testing.assert_array_equal(counts, [0, 1, 8])
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                oracle(np.array([[bad, 1]], np.float32), np.array([2]), 1)


if __name__ == "__main__":
    unittest.main()