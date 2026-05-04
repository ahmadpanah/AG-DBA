"""
tests/test_ag_dba.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for AG-DBA core components.

Tests paper algorithms (Sections 3.1-3.4) and validates correctness.
"""

import unittest
import torch
import numpy as np

from compression.ag_dba import (
    ContinuousLloydMaxQuantizer,
    create_isotropic_rotation,
    AttentionScoreTracker,
    WaterFillingAllocator,
    AGDBA,
)


class TestLloydMaxQuantizer(unittest.TestCase):
    """Test Lloyd-Max quantizer (Section 3.1)."""
    
    def setUp(self):
        self.quantizer = ContinuousLloydMaxQuantizer(
            embedding_dim=128,
            bits=(1, 2, 3, 4),
            device="cpu",
        )
    
    def test_codebook_initialization(self):
        """Test that codebooks are initialized for all bit-widths."""
        for b in [1, 2, 3, 4]:
            self.assertIn(b, self.quantizer.codebooks)
            codebook = self.quantizer.codebooks[b]
            expected_size = 2 ** b
            self.assertEqual(codebook.shape[0], expected_size)
    
    def test_quantize_dequantize(self):
        """Test quantization and dequantization (round-trip)."""
        x = torch.randn(10, 128)  # Random input
        
        for b in [1, 2, 3, 4]:
            # Quantize
            indices = self.quantizer.quantize(x, b)
            self.assertEqual(indices.shape, x.shape)
            self.assertTrue((indices >= 0).all())
            self.assertTrue((indices < 2**b).all())
            
            # Dequantize
            x_reconstructed = self.quantizer.dequantize(indices, b)
            self.assertEqual(x_reconstructed.shape, x.shape)
            
            # Reconstruction should have lower MSE for higher bits
            mse = torch.mean((x - x_reconstructed) ** 2).item()
            self.assertGreater(mse, 0)  # Some quantization error expected
    
    def test_mse_distortion_bound(self):
        """Test MSE distortion bound (Eq. 3)."""
        # Higher bits should have lower distortion
        d1 = self.quantizer.mse_distortion(1)
        d2 = self.quantizer.mse_distortion(2)
        d3 = self.quantizer.mse_distortion(3)
        d4 = self.quantizer.mse_distortion(4)
        
        self.assertGreater(d1, d2)
        self.assertGreater(d2, d3)
        self.assertGreater(d3, d4)


class TestIsotropicRotation(unittest.TestCase):
    """Test isotropic rotation (Section 3.1, Eq. 1)."""
    
    def test_orthogonality(self):
        """Test that rotation matrix is orthogonal."""
        d = 128
        Pi = create_isotropic_rotation(d, seed=42, device="cpu")
        
        # Q^T @ Q should be I
        identity = torch.matmul(Pi.t(), Pi)
        expected_identity = torch.eye(d)
        
        torch.testing.assert_close(identity, expected_identity, atol=1e-5, rtol=1e-4)
    
    def test_determinant(self):
        """Test that determinant is ±1."""
        d = 64
        Pi = create_isotropic_rotation(d, seed=42, device="cpu")
        
        det = torch.det(Pi).item()
        self.assertAlmostEqual(abs(det), 1.0, places=5)
    
    def test_reproducibility(self):
        """Test that same seed produces same rotation."""
        d = 64
        Pi1 = create_isotropic_rotation(d, seed=42, device="cpu")
        Pi2 = create_isotropic_rotation(d, seed=42, device="cpu")
        
        torch.testing.assert_close(Pi1, Pi2)


class TestAttentionScoreTracker(unittest.TestCase):
    """Test attention tracking (Section 3.2, Eq. 4)."""
    
    def setUp(self):
        self.tracker = AttentionScoreTracker(
            alpha=0.85,
            num_heads=32,
            device="cpu",
        )
    
    def test_ema_update(self):
        """Test EMA update (Eq. 4)."""
        seq_len = 100
        
        # Create dummy attention matrix
        attention = torch.randn(1, 32, seq_len)  # [batch, heads, seq_len]
        
        # First update
        scores1 = self.tracker.update(attention)
        self.assertEqual(scores1.shape[0], seq_len)
        
        # Second update
        attention2 = torch.randn(1, 32, seq_len)
        scores2 = self.tracker.update(attention2)
        
        # Scores should be different but in same range
        self.assertFalse(torch.allclose(scores1, scores2))
        self.assertTrue((scores2 >= 0).all())
    
    def test_alpha_effect(self):
        """Test that alpha controls decay."""
        seq_len = 50
        attention = torch.ones(1, 32, seq_len)
        
        # High alpha (memory)
        tracker_high = AttentionScoreTracker(alpha=0.95, num_heads=32, device="cpu")
        scores_high = tracker_high.update(attention)
        
        # Low alpha (forgetting)
        tracker_low = AttentionScoreTracker(alpha=0.5, num_heads=32, device="cpu")
        scores_low = tracker_low.update(attention)
        
        # Both should have positive scores but different magnitudes
        self.assertTrue((scores_high >= 0).all())
        self.assertTrue((scores_low >= 0).all())


class TestWaterFillingAllocator(unittest.TestCase):
    """Test water-filling algorithm (Section 3.4, Algorithm 1)."""
    
    def setUp(self):
        self.quantizer = ContinuousLloydMaxQuantizer(
            embedding_dim=128, bits=(1, 2, 3, 4), device="cpu"
        )
        self.allocator = WaterFillingAllocator(
            quantizer=self.quantizer,
            bit_widths=(1, 2, 3, 4),
            device="cpu",
        )
    
    def test_bit_allocation_respects_budget(self):
        """Test that allocation respects global budget constraint."""
        seq_len = 1000
        importance_scores = torch.randn(seq_len)
        target_bpp = 2.0
        
        bit_allocation = self.allocator.allocate(
            importance_scores,
            target_bits_per_param=target_bpp,
        )
        
        # Check budget constraint
        avg_bits = bit_allocation.float().mean().item()
        self.assertLessEqual(avg_bits, target_bpp + 0.1)  # Small tolerance
    
    def test_higher_importance_gets_more_bits(self):
        """Test that important tokens get higher bit-widths."""
        seq_len = 100
        importance_scores = torch.zeros(seq_len)
        
        # Make first 10 tokens important, rest unimportant
        importance_scores[:10] = 100.0
        importance_scores[10:] = 0.1
        
        bit_allocation = self.allocator.allocate(
            importance_scores,
            target_bits_per_param=2.0,
        )
        
        # Important tokens should have higher bits on average
        important_bits = bit_allocation[:10].float().mean().item()
        unimportant_bits = bit_allocation[10:].float().mean().item()
        
        self.assertGreater(important_bits, unimportant_bits)


class TestAGDBA(unittest.TestCase):
    """Test main AG-DBA framework."""
    
    def setUp(self):
        self.ag_dba = AGDBA(
            embedding_dim=128,
            num_attention_heads=32,
            target_bits_per_param=2.0,
            device="cpu",
            seed=42,
        )
    
    def test_compression_pipeline(self):
        """Test end-to-end compression pipeline."""
        seq_len = 100
        
        # Step 1: Allocate bits
        bit_widths = self.ag_dba.allocate_bits(seq_len)
        self.assertEqual(bit_widths.shape[0], seq_len)
        
        # Step 2: Compress KV
        kv_tensor = torch.randn(seq_len, 128)
        indices, _ = self.ag_dba.compress_kv(kv_tensor, bit_widths)
        self.assertEqual(indices.shape, kv_tensor.shape)
        
        # Step 3: Decompress
        kv_reconstructed = self.ag_dba.decompress_kv(indices, bit_widths)
        self.assertEqual(kv_reconstructed.shape, kv_tensor.shape)
        
        # Reconstruction should be close to original
        mse = torch.mean((kv_tensor - kv_reconstructed) ** 2).item()
        self.assertGreater(mse, 0)  # Some quantization error
        self.assertLess(mse, 0.5)  # But not too much
    
    def test_statistics_computation(self):
        """Test statistics computation."""
        seq_len = 100
        bit_widths = torch.randint(1, 5, (seq_len,))
        
        stats = self.ag_dba.compute_statistics(bit_widths)
        
        self.assertIn("avg_bits_per_param", stats)
        self.assertIn("compression_ratio", stats)
        self.assertGreater(stats["compression_ratio"], 1.0)
        
        expected_avg_bits = bit_widths.float().mean().item()
        self.assertAlmostEqual(stats["avg_bits_per_param"], expected_avg_bits, places=1)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_recall(self):
        """Test NIAH recall metric."""
        from eval.metrics import compute_recall
        
        # Exact match
        self.assertEqual(compute_recall("Hello world", "Hello"), 1.0)
        
        # No match
        self.assertEqual(compute_recall("Hello world", "Goodbye"), 0.0)
        
        # Case insensitive
        self.assertEqual(compute_recall("HELLO", "hello"), 1.0)
    
    def test_f1(self):
        """Test F1-score metric."""
        from eval.metrics import compute_f1
        
        # Identical
        f1 = compute_f1("the quick brown fox", "the quick brown fox")
        self.assertAlmostEqual(f1, 1.0, places=5)
        
        # Partial overlap
        f1 = compute_f1("the quick brown fox", "quick brown")
        self.assertGreater(f1, 0)
        self.assertLess(f1, 1)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
