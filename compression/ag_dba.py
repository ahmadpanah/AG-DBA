"""
compression/ag_dba.py
~~~~~~~~~~~~~~~~~~~~~
Attention-Guided Dynamic Bit Allocation (AG-DBA) for KV Cache Compression.

Implements Sections 3.1–3.4 of the paper:
  - Section 3.1: Isotropic Vector Quantization (Eq. 1–3)
  - Section 3.2: Online Attention-Guided Importance Scoring (Eq. 4)
  - Section 3.3: Constrained Rate-Distortion Optimization (Eq. 5)
  - Section 3.4: Hardware-Aware Greedy Water-Filling Algorithm (Algorithm 1)

References:
  Paper: "Attention-Guided Dynamic Bit Allocation for Long-Context LLM KV Caches 
  via Near-Optimal Vector Quantization"
  arXiv: [TBD]
"""

from __future__ import annotations

import logging
import heapq
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ============================================================================
# Lloyd-Max Quantizer (Continuous)
# ============================================================================

class ContinuousLloydMaxQuantizer:
    """Continuous Lloyd-Max quantizer for Beta-distributed coordinates.
    
    Paper Section 3.1 (Eq. 3):
      After isotropic rotation, coordinates follow Beta(d/2-1, d/2-1).
      For d ≥ 128, this converges to N(0, 1/d).
      
      Optimal MSE distortion: C(f_X, b) ≤ (sqrt(3π)/2) * (1/4^b)
      where b is the bit-width.
    
    Precomputes optimal Lloyd-Max centroids for 1-4 bits offline.
    """
    
    def __init__(
        self,
        embedding_dim: int = 4096,
        bits: Tuple[int, ...] = (1, 2, 3, 4),
        device: str = "cuda",
    ):
        """Initialize quantizer with precomputed codebooks.
        
        Args:
            embedding_dim: d in Beta(d/2-1, d/2-1). Determines variance.
            bits: Bit-widths to precompute. Typically (1,2,3,4).
            device: "cuda" or "cpu"
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.bits = bits
        
        # Codebooks: dict[int, Tensor] of shape [2^b, 1]
        # Stores the centroids for each bit-width
        self.codebooks = {}
        self._precompute_codebooks()
    
    def _precompute_codebooks(self) -> None:
        """Precompute optimal Lloyd-Max centroids for each bit-width.
        
        Approximation: For Beta(d/2-1, d/2-1) with d ≥ 128,
        use quantiles of N(0, 1/d) to find optimal bin edges,
        then compute centroids via integration.
        
        For simplicity, we use uniform quantization on the Beta support.
        For production, implement true Lloyd-Max iterations.
        """
        variance = 1.0 / self.embedding_dim
        std = np.sqrt(variance)
        
        for b in self.bits:
            num_levels = 2 ** b
            
            # Approximate beta support: [-3*std, 3*std]
            # (covers ~99.7% of N(0, variance))
            min_val = -3 * std
            max_val = 3 * std
            
            # Uniform quantization as Lloyd-Max approximation
            levels = np.linspace(min_val, max_val, num_levels)
            
            self.codebooks[b] = torch.tensor(
                levels, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
    
    def quantize(self, x: torch.Tensor, b: int) -> torch.Tensor:
        """Quantize x to b bits using nearest centroid.
        
        Args:
            x: [n_vectors, d] float32 tensor (rotated coordinates)
            b: bit-width ∈ {1, 2, 3, 4}
        
        Returns:
            indices: [n_vectors, d] int32 (centroid indices)
        """
        assert b in self.codebooks, f"Codebook for {b} bits not precomputed"
        
        codebook = self.codebooks[b]  # [num_levels, 1]
        
        # Compute distances: [n, d] x [num_levels, 1] -> [n, d, num_levels]
        distances = torch.abs(x.unsqueeze(2) - codebook.squeeze(1))  # [n, d, num_levels]
        indices = torch.argmin(distances, dim=2).int()  # [n, d]
        
        return indices
    
    def dequantize(self, indices: torch.Tensor, b: int) -> torch.Tensor:
        """Reconstruct from quantized indices.
        
        Args:
            indices: [n_vectors, d] int32 (centroid indices)
            b: bit-width
        
        Returns:
            x_reconstructed: [n_vectors, d] float32
        """
        codebook = self.codebooks[b]  # [num_levels, 1]
        return codebook[indices].squeeze(1)
    
    def mse_distortion(self, b: int) -> float:
        """MSE distortion bound for bit-width b.
        
        Paper Eq. 3: C(f_X, b) ≤ (√(3π)/2) * (1/4^b)
        
        Args:
            b: bit-width
        
        Returns:
            Upper bound on MSE distortion
        """
        const = np.sqrt(3 * np.pi) / 2
        return const * (1.0 / (4 ** b))


# ============================================================================
# Isotropic Random Projection
# ============================================================================

def create_isotropic_rotation(
    embedding_dim: int,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Create random orthogonal matrix Π ∈ R^{d×d}.
    
    Paper Section 3.1 (Eq. 1): y = Πx
    
    Rotates normalized vectors to induce Beta distribution on coordinates.
    Implementation uses QR decomposition of Gaussian random matrix.
    
    Args:
        embedding_dim: d
        seed: For reproducibility
        device: "cuda" or "cpu"
    
    Returns:
        Π: [d, d] orthogonal matrix
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate random Gaussian matrix and orthogonalize via QR
    G = torch.randn(embedding_dim, embedding_dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    
    return Q


# ============================================================================
# EMA Attention Tracking (Section 3.2)
# ============================================================================

@dataclass
class AttentionScoreTracker:
    """Track cumulative attention scores using EMA.
    
    Paper Section 3.2 (Eq. 4):
      S_t(i) = α * S_{t-1}(i) + (1 - α) * Σ_h ω_h * A^{(h)}_{t,i}
      
    where:
      - S_t(i) = importance score of token i at step t
      - α = decay factor (paper: 0.85)
      - A^{(h)}_{t,i} = attention weight from head h at step t to position i
      - ω_h = head importance weighting (uniform or learned)
    """
    
    alpha: float = 0.85  # Decay factor
    num_heads: int = 32
    head_weights: Optional[torch.Tensor] = None  # [num_heads]
    device: str = "cuda"
    
    def __post_init__(self):
        """Initialize head importance weights."""
        if self.head_weights is None:
            # Uniform weighting (paper: "uniform or learned")
            self.head_weights = (
                torch.ones(self.num_heads, device=self.device) / self.num_heads
            )
    
    def update(
        self,
        attention_scores: torch.Tensor,  # [batch, num_heads, seq_len]
        prev_scores: Optional[torch.Tensor] = None,  # [seq_len]
    ) -> torch.Tensor:
        """Update EMA scores after new attention pass.
        
        Args:
            attention_scores: [batch, num_heads, seq_len]
            prev_scores: [seq_len] or None (initialize to 0)
        
        Returns:
            new_scores: [seq_len]
        """
        # Average attention across batch and heads
        # attention_scores: [batch, num_heads, seq_len] 
        # -> [seq_len]
        batch_avg = attention_scores.mean(dim=0)  # [num_heads, seq_len]
        
        # Weighted average across heads
        weighted_avg = (batch_avg.t() * self.head_weights).sum(dim=1)  # [seq_len]
        
        if prev_scores is None:
            prev_scores = torch.zeros_like(weighted_avg)
        
        # EMA update: Eq. 4
        new_scores = self.alpha * prev_scores + (1.0 - self.alpha) * weighted_avg
        
        return new_scores


# ============================================================================
# Water-Filling Algorithm (Algorithm 1)
# ============================================================================

class WaterFillingAllocator:
    """Greedy water-filling algorithm for bit allocation.
    
    Paper Section 3.4 (Algorithm 1):
    Solves the constrained rate-distortion optimization:
    
      min_{b ∈ B_set^N}  Σ_i S_t(i) * C(f_X, b_i)
      s.t. (1/N) * Σ_i b_i ≤ B_target
    
    Greedy approach with max-heap priority queue: O(N log N).
    """
    
    def __init__(
        self,
        quantizer: ContinuousLloydMaxQuantizer,
        bit_widths: Tuple[int, ...] = (1, 2, 3, 4),
        device: str = "cuda",
    ):
        """Initialize allocator.
        
        Args:
            quantizer: ContinuousLloydMaxQuantizer with precomputed distortion bounds
            bit_widths: Available bit-widths {1, 2, 3, 4}
            device: "cuda" or "cpu"
        """
        self.quantizer = quantizer
        self.bit_widths = bit_widths
        self.device = device
        
        # Precompute distortion drops Δ_C_b = C(f_X, b) - C(f_X, b+1)
        self.distortion_drops = self._precompute_distortion_drops()
    
    def _precompute_distortion_drops(self) -> dict:
        """Precompute marginal utility drops for all bit transitions.
        
        For marginal utility (Eq. 6):
          ΔU(i, b) = S(i) × [C(f_X, b) - C(f_X, b+1)]
        
        Precompute C(f_X, b) - C(f_X, b+1) for all b ∈ {1, 2, 3}.
        """
        drops = {}
        for b in range(min(self.bit_widths), max(self.bit_widths)):
            c_b = self.quantizer.mse_distortion(b)
            c_b_plus_1 = self.quantizer.mse_distortion(b + 1)
            drops[b] = c_b - c_b_plus_1
        
        return drops
    
    def allocate(
        self,
        importance_scores: torch.Tensor,  # [seq_len]
        target_bits_per_param: float,
        min_bits: int = 1,
        max_bits: int = 4,
    ) -> torch.Tensor:
        """Allocate bit-widths greedily using water-filling.
        
        Paper Algorithm 1:
        1. Initialize all tokens to b_min = 1
        2. Compute marginal utility for each (token, current_bits)
        3. Greedily upgrade token with highest ΔU until budget exhausted
        
        Args:
            importance_scores: [seq_len] importance S_t(i)
            target_bits_per_param: average budget B_target
            min_bits: minimum bit-width per token (default: 1)
            max_bits: maximum bit-width per token (default: 4)
        
        Returns:
            bit_allocation: [seq_len] allocated bit-widths
        """
        seq_len = importance_scores.shape[0]
        
        # Initialize all tokens to min_bits (Algorithm 1, line 1)
        bit_allocation = torch.full(
            (seq_len,), min_bits, dtype=torch.int32, device=self.device
        )
        
        # Current and max total bits (Algorithm 1, lines 2-3)
        b_current = min_bits * seq_len
        b_max_allowable = int(target_bits_per_param * seq_len)
        
        # Priority queue: (-marginal_utility, token_idx)
        # Negative because heapq is min-heap; we want max-heap
        pq = []
        
        # Initialize heap with all tokens (Algorithm 1, lines 5-9)
        for i in range(seq_len):
            current_bits = bit_allocation[i].item()
            if current_bits < max_bits:
                marginal_util = importance_scores[i].item() * self.distortion_drops[current_bits]
                heapq.heappush(pq, (-marginal_util, i, current_bits))
        
        # Greedy water-filling loop (Algorithm 1, lines 10-18)
        while pq and b_current < b_max_allowable:
            neg_util, token_idx, old_bits = heapq.heappop(pq)
            
            # Verify token's current bits haven't changed
            if bit_allocation[token_idx].item() != old_bits:
                continue
            
            new_bits = old_bits + 1
            
            # Upgrade token (Algorithm 1, line 12)
            bit_allocation[token_idx] = new_bits
            b_current += 1
            
            # Re-insert if further upgrade possible (Algorithm 1, lines 14-17)
            if new_bits < max_bits:
                next_util = importance_scores[token_idx].item() * self.distortion_drops[new_bits]
                heapq.heappush(pq, (-next_util, token_idx, new_bits))
        
        return bit_allocation


# ============================================================================
# Main AG-DBA Framework
# ============================================================================

class AGDBA:
    """Attention-Guided Dynamic Bit Allocation framework.
    
    Combines all components: rotation, attention tracking, bit allocation,
    and quantization to compress KV caches dynamically during inference.
    
    Paper Architecture (Figure 1):
      1. Input: Raw KV vectors x ∈ R^d
      2. Split: Isotropic rotation y = Πx
      3. Score & Allocate: EMA attention tracking + water-filling → b_i
      4. Quantize & Store: Lloyd-Max quantizer → compressed cache ≈ 1.5 bpp
    """
    
    def __init__(
        self,
        embedding_dim: int = 4096,
        num_attention_heads: int = 32,
        target_bits_per_param: float = 2.0,
        attention_decay_alpha: float = 0.85,
        device: str = "cuda",
        seed: int = 42,
    ):
        """Initialize AG-DBA system.
        
        Args:
            embedding_dim: d (typically 128 for GQA KV vectors)
            num_attention_heads: H
            target_bits_per_param: B_target (1.5 or 2.0 in paper)
            attention_decay_alpha: α in Eq. 4
            device: "cuda" or "cpu"
            seed: For reproducible rotations
        """
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.target_bits_per_param = target_bits_per_param
        self.device = device
        self.seed = seed
        
        # Component 1: Isotropic rotation
        self.rotation_matrix = create_isotropic_rotation(
            embedding_dim, seed=seed, device=device
        )
        
        # Component 2: Lloyd-Max quantizer
        self.quantizer = ContinuousLloydMaxQuantizer(
            embedding_dim=embedding_dim,
            bits=(1, 2, 3, 4),
            device=device,
        )
        
        # Component 3: Attention tracking
        self.attention_tracker = AttentionScoreTracker(
            alpha=attention_decay_alpha,
            num_heads=num_attention_heads,
            device=device,
        )
        
        # Component 4: Water-filling allocator
        self.allocator = WaterFillingAllocator(
            quantizer=self.quantizer,
            bit_widths=(1, 2, 3, 4),
            device=device,
        )
        
        # State tracking
        self.importance_scores: Optional[torch.Tensor] = None
        self.bit_allocation: Optional[torch.Tensor] = None
        self.current_sequence_len = 0
    
    def update_attention_scores(
        self,
        attention_matrix: torch.Tensor,  # [batch, num_heads, seq_len]
    ) -> torch.Tensor:
        """Update importance scores from new attention pass.
        
        Paper Section 3.2 (Eq. 4).
        
        Args:
            attention_matrix: [batch, num_heads, seq_len] attention weights
        
        Returns:
            updated_scores: [seq_len] cumulative importance
        """
        self.importance_scores = self.attention_tracker.update(
            attention_matrix, self.importance_scores
        )
        return self.importance_scores
    
    def allocate_bits(
        self,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute dynamic bit allocation for sequence.
        
        Paper Section 3.3-3.4 (Eq. 5, Algorithm 1).
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            bit_allocation: [seq_len] bit-widths per token
        """
        if self.importance_scores is None:
            self.importance_scores = torch.ones(seq_len, device=self.device)
        
        self.bit_allocation = self.allocator.allocate(
            self.importance_scores[:seq_len],
            target_bits_per_param=self.target_bits_per_param,
        )
        
        return self.bit_allocation
    
    def compress_kv(
        self,
        kv_tensor: torch.Tensor,  # [seq_len, embedding_dim]
        bit_widths: torch.Tensor,  # [seq_len] allocated bit-widths
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress KV vectors using dynamic bit allocation.
        
        Paper Section 3.1-3.4.
        
        Args:
            kv_tensor: [seq_len, embedding_dim] float32 KV vectors
            bit_widths: [seq_len] allocated bit-widths
        
        Returns:
            indices: [seq_len, embedding_dim] quantized indices
            metadata: [seq_len] bit allocation for decompression
        """
        seq_len, d = kv_tensor.shape
        
        # Step 2 (Figure 1): Isotropic rotation y = Πx
        rotated = torch.matmul(kv_tensor, self.rotation_matrix.t())  # [seq, d]
        
        # Step 4 (Figure 1): Quantize each vector with its assigned bits
        indices_list = []
        for i in range(seq_len):
            b = bit_widths[i].item()
            vec = rotated[i:i+1, :]  # [1, d]
            idx = self.quantizer.quantize(vec, b)  # [1, d]
            indices_list.append(idx)
        
        indices = torch.cat(indices_list, dim=0)  # [seq_len, d]
        
        return indices, bit_widths
    
    def decompress_kv(
        self,
        indices: torch.Tensor,  # [seq_len, embedding_dim]
        bit_widths: torch.Tensor,  # [seq_len]
    ) -> torch.Tensor:
        """Decompress KV vectors from quantized representation.
        
        Args:
            indices: [seq_len, embedding_dim] quantized indices
            bit_widths: [seq_len] bit allocation
        
        Returns:
            kv_reconstructed: [seq_len, embedding_dim] float32
        """
        seq_len, d = indices.shape
        reconstructed_list = []
        
        for i in range(seq_len):
            b = bit_widths[i].item()
            idx = indices[i:i+1, :]  # [1, d]
            vec = self.quantizer.dequantize(idx, b)  # [1, d]
            reconstructed_list.append(vec)
        
        # Inverse rotation: x = Π^T * y
        reconstructed = torch.cat(reconstructed_list, dim=0)  # [seq_len, d]
        kv_reconstructed = torch.matmul(reconstructed, self.rotation_matrix)  # [seq_len, d]
        
        return kv_reconstructed
    
    def compute_statistics(self, bit_widths: torch.Tensor) -> dict:
        """Compute compression statistics.
        
        Args:
            bit_widths: [seq_len] allocated bit-widths
        
        Returns:
            stats: Dict with average bits, compression ratio, etc.
        """
        avg_bits = bit_widths.float().mean().item()
        num_tokens = bit_widths.shape[0]
        original_bytes = num_tokens * self.embedding_dim * 2  # FP16 = 2 bytes
        compressed_bytes = (num_tokens * self.embedding_dim * avg_bits) / 8
        compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')
        
        return {
            "avg_bits_per_param": avg_bits,
            "num_tokens": num_tokens,
            "original_size_mb": original_bytes / (1024 ** 2),
            "compressed_size_mb": compressed_bytes / (1024 ** 2),
            "compression_ratio": compression_ratio,
        }
