"""Codec and container tests: sizes must match what is written, and what is read
must decode to what was encoded."""

import math
import os
import random
import tempfile
import unittest

from pocket import quant, store
from pocket.nn import ModelConfig, init_params


class QuantCodec(unittest.TestCase):
    def test_round_trip_error_ordering(self):
        rng = random.Random(0)
        vals = [rng.gauss(0, 0.05) for _ in range(512)]
        errors = {}
        for scheme in quant.SCHEMES:
            decoded = quant.decode(quant.encode(vals, scheme, 32), len(vals), scheme, 32)
            self.assertEqual(len(decoded), len(vals))
            errors[scheme] = quant.reconstruction_error(vals, decoded)["relative_rmse"]
        self.assertLess(errors["fp16"], 1e-3)
        self.assertLess(errors["int8"], errors["int4"])
        self.assertLess(errors["int4"], errors["int2"])
        self.assertLess(errors["int2"], 0.6)
        self.assertLess(errors["ternary"], 0.7)
        # and the ordering must survive at the bit budgets people actually ship
        self.assertLess(errors["int4"], 0.15)

    def test_encoded_size_matches_encoder(self):
        rng = random.Random(1)
        for count in (1, 5, 32, 33, 1000):
            vals = [rng.gauss(0, 1) for _ in range(count)]
            for scheme in quant.SCHEMES:
                blob = quant.encode(vals, scheme, 32)
                self.assertEqual(len(blob), quant.nbytes_for(count, scheme, 32),
                                 f"{scheme} count={count}")

    def test_bits_per_weight_accounts_for_metadata(self):
        # 2-bit with 32-wide groups spends 8 bytes of scale+zero per 32 weights
        self.assertAlmostEqual(quant.bits_per_weight(320, "int4", 32), 5.0, places=1)
        self.assertEqual(quant.bits_per_weight(320, "fp16", 32), 16.0)
        # the point of the test: at 32-wide groups, 8 bytes of scale+zero per group
        # make "2-bit" cost exactly as much as 4 bits per weight.
        self.assertAlmostEqual(quant.bits_per_weight(32, "int2", 32), 4.0, places=2)
        self.assertLess(quant.bits_per_weight(128, "int2", 128), 2.6)

    def test_int8_is_near_exact(self):
        rng = random.Random(2)
        vals = [rng.uniform(-1, 1) for _ in range(256)]
        decoded = quant.decode(quant.encode(vals, "int8", 32), len(vals), "int8", 32)
        self.assertLess(quant.reconstruction_error(vals, decoded)["max_abs"], 0.01)

    def test_zeros_and_constant_groups(self):
        for scheme in ("int8", "int4", "int2", "ternary"):
            vals = [0.0] * 64
            decoded = quant.decode(quant.encode(vals, scheme, 32), 64, scheme, 32)
            self.assertTrue(all(abs(v) < 1e-6 for v in decoded), scheme)

    def test_rejects_unknown_scheme(self):
        with self.assertRaises(ValueError):
            quant.nbytes_for(10, "int3")


class StoreFormat(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(vocab=24, d_model=16, n_layers=2, n_heads=4, ffn_dim=20,
                               context=8, n_experts=3, top_k=2)
        self.params = init_params(self.cfg, random.Random(3))
        self.tmp = tempfile.mkdtemp()

    def test_round_trip_all_tensors(self):
        path = os.path.join(self.tmp, "m.pksw")
        records = store.write_store(path, self.cfg, self.params, scheme="int8", group_size=32)
        self.assertEqual(len(records), len(store.tensor_order(self.cfg)))
        with store.MappedStore(path) as ms:
            for name, rec in ms.records.items():
                decoded = ms.decode(name)
                self.assertEqual(len(decoded), rec.count, name)
                err = quant.reconstruction_error(self.params.data[name], decoded)["relative_rmse"]
                self.assertLess(err, 0.02, name)

    def test_offsets_are_aligned_and_ordered(self):
        path = os.path.join(self.tmp, "a.pksw")
        store.write_store(path, self.cfg, self.params, scheme="int4", group_size=32)
        with store.MappedStore(path) as ms:
            offsets = [ms.records[n].offset for n in store.tensor_order(self.cfg)]
            self.assertEqual(offsets, sorted(offsets))
            for off in offsets:
                self.assertEqual(off % 64, 0)
            self.assertLessEqual(ms.file_size, ms.records[list(ms.records)[-1]].offset
                                 + ms.records[list(ms.records)[-1]].nbytes + 64)

    def test_pread_and_mmap_agree(self):
        path = os.path.join(self.tmp, "b.pksw")
        store.write_store(path, self.cfg, self.params, scheme="int4", group_size=32)
        with store.MappedStore(path) as a, store.MappedStore(path, use_mmap=True) as b:
            for name in a.records:
                self.assertEqual(a.decode(name), b.decode(name), name)

    def test_quantized_store_is_smaller(self):
        sizes = {}
        for scheme in ("fp16", "int8", "int4"):
            path = os.path.join(self.tmp, f"{scheme}.pksw")
            store.write_store(path, self.cfg, self.params, scheme=scheme, group_size=32)
            sizes[scheme] = os.path.getsize(path)
        self.assertLess(sizes["int8"], sizes["fp16"])
        self.assertLess(sizes["int4"], sizes["int8"])

    def test_checkpoint_round_trip(self):
        path = os.path.join(self.tmp, "c.pkck")
        store.save_checkpoint(path, self.cfg, self.params, {"note": "hello"})
        cfg, params, meta = store.load_checkpoint(path)
        self.assertEqual(cfg.as_dict(), self.cfg.as_dict())
        self.assertEqual(meta["note"], "hello")
        for name, vals in self.params.data.items():
            for a, b in zip(vals, params.data[name]):
                self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
    unittest.main()
