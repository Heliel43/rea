"""Unit tests for the reference implementation.

The gradient check is the load-bearing test in this repository: every quality
number in RESULTS.md is produced by the trainer in ``train.py``, which is only
meaningful if ``backward`` is exactly the transpose of ``forward``.
"""

import math
import random
import unittest

from pocket.nn import ForwardTrace, ModelConfig, Params, forward, init_params, loss_and_grads


class GradientCheck(unittest.TestCase):
    def _check(self, cfg, seed=7, steps=6):
        rng = random.Random(seed)
        p = init_params(cfg, rng)
        tokens = [rng.randrange(cfg.vocab) for _ in range(3)]
        targets = [rng.randrange(cfg.vocab) for _ in range(3)]
        _, grads = loss_and_grads(cfg, p, tokens, targets)

        eps = 1e-4
        worst = 0.0
        worst_name = ""
        names = list(p.data)
        rng.shuffle(names)
        checked = 0
        for name in names[: steps if steps else len(names)]:
            flat = p.data[name]
            for idx in range(min(4, len(flat))):
                orig = flat[idx]
                flat[idx] = orig + eps
                lp, _ = loss_and_grads(cfg, p, tokens, targets)
                flat[idx] = orig - eps
                lm, _ = loss_and_grads(cfg, p, tokens, targets)
                flat[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                analytic = grads.data[name][idx]
                denom = max(1.0, abs(numeric), abs(analytic))
                err = abs(numeric - analytic) / denom
                checked += 1
                if err > worst:
                    worst, worst_name = err, f"{name}[{idx}] numeric={numeric:.6g} analytic={analytic:.6g}"
        self.assertGreater(checked, 0)
        self.assertLess(worst, 1e-4, f"worst relative gradient error {worst:.3e} at {worst_name}")

    def test_dense(self):
        self._check(ModelConfig(vocab=11, d_model=8, n_layers=2, n_heads=2, ffn_dim=12, context=8))

    def test_moe_top2(self):
        self._check(
            ModelConfig(
                vocab=11, d_model=8, n_layers=2, n_heads=2, ffn_dim=12, context=8, n_experts=3, top_k=2
            )
        )

    def test_moe_top1(self):
        self._check(
            ModelConfig(
                vocab=9, d_model=8, n_layers=1, n_heads=4, ffn_dim=10, context=8, n_experts=4, top_k=1
            )
        )


class ForwardContract(unittest.TestCase):
    def test_causality(self):
        """Changing a future token must not change earlier logits."""

        cfg = ModelConfig(vocab=13, d_model=8, n_layers=2, n_heads=2, ffn_dim=10, context=8)
        rng = random.Random(3)
        p = init_params(cfg, rng)
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 9, 7]
        la, _ = forward(cfg, p, a)
        lb, _ = forward(cfg, p, b)
        for t in range(3):
            for i in range(cfg.vocab):
                self.assertAlmostEqual(la[t][i], lb[t][i], places=10)

    def test_trace_matches_forward(self):
        cfg = ModelConfig(
            vocab=10, d_model=8, n_layers=2, n_heads=2, ffn_dim=12, context=8, n_experts=4, top_k=2
        )
        rng = random.Random(5)
        p = init_params(cfg, rng)
        tokens = [1, 4, 2, 7]
        trace = ForwardTrace()
        forward(cfg, p, tokens, trace=trace)
        self.assertEqual(len(trace.router_choices), cfg.n_layers)
        for layer in trace.router_choices:
            self.assertEqual(len(layer), len(tokens))
            for chosen in layer:
                self.assertEqual(len(chosen), 2)
                self.assertEqual(len(set(chosen)), 2)
        # every recorded probability vector sums to 1
        for layer in trace.router_probs:
            for weights in layer:
                self.assertAlmostEqual(sum(weights), 1.0, places=9)

    def test_no_nans(self):
        cfg = ModelConfig(vocab=17, d_model=12, n_layers=3, n_heads=3, ffn_dim=16, context=6)
        rng = random.Random(11)
        p = init_params(cfg, rng)
        logits, _ = forward(cfg, p, [1, 2, 3, 4, 5, 6])
        for row in logits:
            for v in row:
                self.assertTrue(math.isfinite(v))


class ParamAccounting(unittest.TestCase):
    def test_counts_match_formula(self):
        cfg = ModelConfig(
            vocab=32, d_model=16, n_layers=2, n_heads=4, ffn_dim=24, context=8, n_experts=3, top_k=1
        )
        p = init_params(cfg, random.Random(1))
        d, ff, e = cfg.d_model, cfg.ffn_dim, cfg.n_experts
        expected = cfg.vocab * d + cfg.context * d + d
        for _ in range(cfg.n_layers):
            expected += d  # attn_norm
            expected += d * d + 2 * (cfg.d_kv * d) + d * d
            expected += d  # ffn_norm
            expected += e * d  # router
            expected += 2 * (e * ff * d) + e * d * ff
        self.assertEqual(p.count(), expected)


if __name__ == "__main__":
    unittest.main()
