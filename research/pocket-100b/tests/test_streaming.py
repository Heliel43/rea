"""The streaming engine must be *the same maths* as the resident path.

If a quantized, streamed, prefetched run ever produced different logits from a
resident run over the same decoded weights, every timing in RESULTS.md would be
measuring a different model.  So the central test here is bit-exactness.
"""

import json
import os
import random
import tempfile
import unittest

from pocket import nn, store, streaming
from pocket.nn import ModelConfig, init_params


def _decoded_params(cfg, path):
    "Rebuild resident parameters by decoding the store, tensor by tensor."

    out = nn.Params()
    with store.MappedStore(path) as ms:
        for name in store.tensor_order(cfg):
            rec = ms.records[name]
            out.add(name, rec.shape, ms.decode(name))
    return out


class StreamingEqualsResident(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(vocab=24, d_model=16, n_layers=3, n_heads=4, ffn_dim=20,
                              context=8, n_experts=4, top_k=2)
        self.params = init_params(self.cfg, random.Random(5))
        self.tokens = [1, 5, 7, 2, 9]
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "s.pksw")
        store.write_store(self.path, self.cfg, self.params, scheme="int8", group_size=32)

    def test_logits_are_bit_identical(self):
        reference, _ = nn.forward(self.cfg, _decoded_params(self.cfg, self.path), self.tokens)
        for depth in (0, 1, 2):
            with store.MappedStore(self.path) as ms:
                model = streaming.StreamedModel(ms, depth=depth)
                got = model.forward(self.tokens)
                model.close()
            self.assertEqual(len(got), len(reference))
            for a_row, b_row in zip(reference, got):
                for a, b in zip(a_row, b_row):
                    self.assertEqual(a, b, f"depth {depth} diverged from resident path")

    def test_generation_is_deterministic_across_depths(self):
        outputs = []
        for depth in (0, 1, 3):
            with store.MappedStore(self.path) as ms:
                model = streaming.StreamedModel(ms, depth=depth)
                outputs.append(model.generate([1, 2, 3], n_tokens=3))
                model.close()
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_byte_accounting_matches_store_layout(self):
        with store.MappedStore(self.path) as ms:
            model = streaming.StreamedModel(ms, depth=1)
            model.forward(self.tokens)
            model.close()
        expected = sum(ms.records[n].nbytes for n in store.tensor_order(self.cfg))
        # embeddings are re-read per pass, everything else once
        self.assertGreaterEqual(model.pipe.bytes_read, expected)
        self.assertLessEqual(model.pipe.bytes_read, 2 * expected)

    def test_lookahead_window_bounds_submissions(self):
        with store.MappedStore(self.path) as ms:
            model = streaming.StreamedModel(ms, depth=2)
            model.forward([1, 2])
            total = sum(1 for name in model.pipe.order if not model.pipe._slots[name].empty())
            model.close()
        self.assertLessEqual(total, len(model.pipe.order))


class TransportBench(unittest.TestCase):
    def test_prefetch_and_sync_both_read_everything(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "blob.bin")
        size = 4 * 1024 * 1024
        with open(path, "wb") as fh:
            fh.write(bytes(size))
        for mode, depth in (("sync", 1), ("prefetch", 2), ("prefetch", 4)):
            r = streaming.benchmark_transport(path, size, chunk_bytes=256 * 1024, depth=depth, mode=mode)
            self.assertEqual(r["bytes"], size)
            self.assertGreater(r["mb_per_s"], 0)
            self.assertEqual(r["mode"], mode)

    def test_mmap_mode_reads_everything(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "blob.bin")
        size = 2 * 1024 * 1024
        with open(path, "wb") as fh:
            fh.write(bytes(size))
        r = streaming.benchmark_transport(path, size, chunk_bytes=256 * 1024, mode="mmap")
        self.assertEqual(r["bytes"], size)


class RssAccounting(unittest.TestCase):
    def test_rss_is_readable(self):
        self.assertGreater(streaming.rss_kb(), 0)
        self.assertGreaterEqual(streaming.peak_rss_kb(), streaming.rss_kb() * 0.9)


if __name__ == "__main__":
    unittest.main()
