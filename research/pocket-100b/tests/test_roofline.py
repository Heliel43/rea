"""The cost model's invariants.

A planner that quietly returns a flattering number is worse than no planner, so
these tests pin the arithmetic: the published active-parameter counts must be
reproduced from the derived expert share, a streaming plan must equal bandwidth over
bytes, and speculation must agree with a simulation of the process it models.
"""

import math
import random
import unittest

from pocket import catalog, roofline


class CatalogConsistency(unittest.TestCase):
    def test_active_params_match_published_figures(self):
        for key, published in (
            ("mixtral-8x7b", 12.9),
            ("mixtral-8x22b", 39.0),
            ("qwen3-235b-a22b", 22.0),
            ("gpt-oss-120b", 5.1),
            ("deepseek-v3", 37.0),
        ):
            m = catalog.model(key)
            self.assertAlmostEqual(m.active_params_b, published, delta=published * 0.02,
                                   msg=f"{key} active params drifted")

    def test_dense_models_have_no_expert_share(self):
        m = catalog.model("llama-3.1-70b")
        self.assertEqual(m.expert_share, 0.0)
        self.assertAlmostEqual(m.active_params_b, 70.6)

    def test_weight_bytes(self):
        m = catalog.model("llama-3.1-8b")
        self.assertAlmostEqual(m.weight_bytes(16) / 1e9, 16.06, delta=0.2)
        self.assertAlmostEqual(m.weight_bytes(4) / 1e9, 4.02, delta=0.1)


class PlanArithmetic(unittest.TestCase):
    def test_stream_throughput_is_bandwidth_over_bytes(self):
        m = catalog.model("dense-100b")
        dev = catalog.device("laptop-16gb")
        p = roofline.plan(m, dev, bits=4.0, strategy="stream")
        expected = (dev.storage_bw_gbs * 1e9) / m.weight_bytes(4)
        self.assertAlmostEqual(p["tok_s"] / expected, 1.0, delta=0.02)
        self.assertEqual(p["bottleneck"], "storage")
        self.assertTrue(p["needs_streaming"])

    def test_resident_needs_capacity(self):
        m = catalog.model("llama-3.1-70b")
        dev = catalog.device("laptop-16gb")
        with self.assertRaises(ValueError):
            roofline.plan(m, dev, bits=4.0, strategy="ram")
        small = catalog.model("llama-3.1-8b")
        p = roofline.plan(small, dev, bits=4.0, strategy="ram")
        self.assertTrue(p["fits_in_ram"])

    def test_vram_strategy_rejected_without_vram(self):
        with self.assertRaises(ValueError):
            roofline.plan(catalog.model("llama-3.1-8b"), catalog.device("phone-8gb"),
                          strategy="vram")

    def test_auto_never_worse_than_any_alternative(self):
        for mk in ("llama-3.1-8b", "llama-3.1-70b", "gpt-oss-120b", "qwen3-235b-a22b", "dense-100b"):
            for dk in ("phone-8gb", "laptop-16gb", "desktop-32gb-4090", "mac-studio-96gb"):
                p = roofline.plan(catalog.model(mk), catalog.device(dk), bits=4.0, ctx=4096)
                self.assertGreater(p["tok_s"], 0, f"{mk}/{dk}")
                for alt in p.get("alternatives", []):
                    self.assertLessEqual(alt["tok_s"], p["tok_s"] + 1e-9, f"{mk}/{dk}/{alt['strategy']}")

    def test_bytes_per_token_never_exceeds_model_size_for_dense(self):
        m = catalog.model("llama-3.1-70b")
        for dk in ("laptop-16gb", "desktop-32gb-4090"):
            p = roofline.plan(m, catalog.device(dk), bits=4.0, ctx=2048)
            self.assertLessEqual(p["bytes_per_token"], m.weight_bytes(4) + 1)

    def test_moe_reads_less_than_the_file(self):
        m = catalog.model("gpt-oss-120b")
        p = roofline.plan(m, catalog.device("laptop-16gb"), bits=4.0, strategy="stream", ctx=2048)
        self.assertLess(p["bytes_per_token"], m.weight_bytes(4))

    def test_kv_cache_scales_with_context(self):
        m = catalog.model("llama-3.1-8b")
        dev = catalog.device("laptop-16gb")
        a = roofline.plan(m, dev, bits=4.0, strategy="ram", ctx=1024)
        b = roofline.plan(m, dev, bits=4.0, strategy="ram", ctx=4096)
        self.assertGreater(b["kv_gb_at_ctx"], a["kv_gb_at_ctx"])
        self.assertAlmostEqual(b["kv_gb_at_ctx"] / a["kv_gb_at_ctx"], 4.0, delta=0.05)

    def test_speculation_amortises_bytes(self):
        m = catalog.model("gpt-oss-120b")
        dev = catalog.device("laptop-16gb")
        base = roofline.plan(m, dev, bits=4.0, strategy="stream", ctx=2048)
        spec = roofline.plan(m, dev, bits=4.0, strategy="stream", ctx=2048, alpha=0.7, draft_tokens=4)
        expected = (1 - 0.7 ** 5) / (1 - 0.7)
        self.assertAlmostEqual(spec["tokens_per_weight_pass"], expected, places=6)
        self.assertAlmostEqual(spec["tok_s"] / base["tok_s"], spec["tokens_per_weight_pass"], delta=0.15)


class SpeculationFormula(unittest.TestCase):
    def test_closed_form_matches_simulation(self):
        rng = random.Random(0)
        for alpha in (0.0, 0.3, 0.5, 0.75):
            for k in (1, 2, 3, 6):
                trials, total = 3000, 0
                for _ in range(trials):
                    accepted = 0
                    for _ in range(k):
                        if rng.random() <= alpha:
                            accepted += 1
                        else:
                            break
                    total += accepted + 1
                self.assertAlmostEqual(total / trials, roofline.speculation_tokens(alpha, k), delta=0.12)

    def test_edges(self):
        self.assertEqual(roofline.speculation_tokens(0.0, 5), 1.0)
        self.assertEqual(roofline.speculation_tokens(0.7, 0), 1.0)
        self.assertAlmostEqual(roofline.speculation_tokens(1.0, 3), 4.0)


class HitRateCurve(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(roofline.default_hit_rate(0.0), 0.0)
        self.assertAlmostEqual(roofline.default_hit_rate(1.0), 1.0)
        last = -1.0
        for i in range(11):
            v = roofline.default_hit_rate(i / 10)
            self.assertGreaterEqual(v, last)
            last = v
        # concave: the first 10% of the pool should carry more than 10% of requests
        self.assertGreater(roofline.default_hit_rate(0.1), 0.15)


class LeverStack(unittest.TestCase):
    def test_stack_is_complete_and_monotone_in_strategy(self):
        rows = roofline.lever_stack(catalog.model("gpt-oss-120b"), catalog.device("desktop-32gb-4090"))
        labels = [r["technique"] for r in rows]
        self.assertEqual(labels[0], "fp16 resident")
        self.assertTrue(any("speculation" in l for l in labels))
        self.assertTrue(all(r["bytes_per_token"] >= 0 for r in rows))

    def test_measured_device_from_probe(self):
        profile = {
            "cpu": {"ram_mb": 2048, "cpu_count": 2},
            "derived": {"storage_seq_mb_s": 500.0, "storage_random_mb_s": 50.0, "copy_gbs": 2.0},
            "python_matvec": {"m_mac_per_s": 40.0},
        }
        dev = catalog.device_from_probe(profile, "probe")
        self.assertAlmostEqual(dev.ram_gb, 2.0)
        self.assertAlmostEqual(dev.storage_bw_gbs, 0.5)
        self.assertGreater(dev.ram_bw_gbs, 0)


if __name__ == "__main__":
    unittest.main()
