"""Turning a file declaration into a plan.

The fixtures are schema-shaped documents (REA's ``modelArtifact`` contract) so these
tests need neither a 40 GB checkpoint nor a built CLI.  `pocket/gguf.py` can also
write a real GGUF v3 file, which is exercised here to keep the fixture writer honest.
"""

import json
import os
import tempfile
import unittest

from pocket import catalog, fixtures, gguf, reaadapter


class Extraction(unittest.TestCase):
    def test_finds_tensor_rows_and_metadata(self):
        doc = fixtures.inspection_document("moe-120b")
        rows = reaadapter.find_tensor_rows(doc)
        self.assertEqual(len(rows), len(doc["tensors"]))
        meta = reaadapter.find_metadata(doc)
        self.assertEqual(meta["llama.block_count"], 36)
        self.assertEqual(meta["llama.expert_count"], 128)

    def test_tolerates_nesting_and_extra_fields(self):
        doc = {"result": {"view": {"kind": "model-artifact", "rows": fixtures.inspection_document("dense-70b")["tensors"]}},
               "noise": [1, 2, 3]}
        rows = reaadapter.find_tensor_rows(doc)
        self.assertGreater(len(rows), 100)

    def test_missing_rows_is_an_error(self):
        with self.assertRaises(ValueError):
            reaadapter.ArtifactShape({"artifact": {"path": "/x"}})


class ShapeDerivation(unittest.TestCase):
    def test_dense_shape(self):
        doc = fixtures.inspection_document("dense-70b")
        shape = reaadapter.ArtifactShape(doc)
        self.assertEqual(shape.layers, 80)
        self.assertEqual(shape.expert_count, 0)
        self.assertAlmostEqual(shape.params / 1e9, 70.6, delta=1.0)
        self.assertAlmostEqual(shape.effective_bits(), 4.0, delta=0.1)

    def test_moe_shape_uses_metadata_not_guessed_axis(self):
        doc = fixtures.inspection_document("moe-120b")
        shape = reaadapter.ArtifactShape(doc)
        self.assertEqual(shape.expert_count, 128)   # not 2880 from the ffn axis
        self.assertEqual(shape.top_k, 4)
        spec = shape.to_model_spec()
        self.assertAlmostEqual(spec.active_params_b, 5.1, delta=1.5)

    def test_kv_bytes_per_token(self):
        doc = fixtures.inspection_document("dense-70b")
        shape = reaadapter.ArtifactShape(doc)
        # 2 * 8 kv heads * 128 head_dim * 80 layers * 2 bytes
        self.assertEqual(shape.kv_bytes_per_token, 2 * 8 * 128 * 80 * 2)


class Planning(unittest.TestCase):
    def test_plan_contains_schedule_and_warnings(self):
        doc = fixtures.inspection_document("moe-120b")
        out = reaadapter.plan_from_inspection(doc, catalog.device("laptop-16gb"))
        self.assertEqual(len(out["schedule"]["layer_bytes_mb"]), 36)
        self.assertGreater(out["plan"]["tok_s"], 0)
        self.assertTrue(any("token embedding" in w for w in out["warnings"]))
        self.assertTrue(any("experts per layer" in w for w in out["warnings"]))
        self.assertGreaterEqual(out["schedule"]["prefetch_depth"], 2)

    def test_storage_override_changes_throughput(self):
        doc = fixtures.inspection_document("dense-70b")
        slow = reaadapter.plan_from_inspection(doc, catalog.device("laptop-16gb"), storage_bw_gbs=1.0)
        fast = reaadapter.plan_from_inspection(doc, catalog.device("laptop-16gb"), storage_bw_gbs=8.0)
        self.assertGreater(fast["plan"]["tok_s"], slow["plan"]["tok_s"])

    def test_cli_plumbing(self):
        import contextlib
        import io

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "inspection.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(fixtures.inspection_document("dense-70b"), fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = reaadapter.main([path, "--device", "desktop-32gb-4090", "--json"])
        self.assertEqual(code, 0)
        plan = json.loads(buf.getvalue())
        self.assertIn("plan", plan)
        self.assertIn("schedule", plan)


class GgufFixture(unittest.TestCase):
    def test_writes_a_real_gguf_header(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "f.gguf")
        size = gguf.llama_fixture(path, layers=2, d=64, ffn=128, vocab=32, experts=4)
        self.assertEqual(size, os.path.getsize(path))
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(4), b"GGUF")
            version = int.from_bytes(fh.read(4), "little")
            self.assertEqual(version, 3)
            tensors = int.from_bytes(fh.read(8), "little")
            self.assertGreater(tensors, 20)


if __name__ == "__main__":
    unittest.main()
