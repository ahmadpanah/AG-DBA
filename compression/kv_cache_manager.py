"""
compression/kv_cache_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KV Cache Management and Integration with AG-DBA.

This module handles:
1. Capturing attention scores during model inference
2. Managing compressed KV caches during generation
3. Decompression on attention computation
4. VRAM/throughput tracking

Integration point with transformers library (Section 4.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

from compression.ag_dba import AGDBA

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Storage
# ============================================================================

@dataclass
class CompressedKVCache:
    """Stores compressed KV cache in quantized form.
    
    Attributes:
        indices_k: [seq_len, num_heads, head_dim] quantized indices (keys)
        indices_v: [seq_len, num_heads, head_dim] quantized indices (values)
        bit_widths_k: [seq_len] bit allocation for keys
        bit_widths_v: [seq_len] bit allocation for values
        attention_scores: [seq_len] cumulative attention importance scores
        metadata: Dict with compression info (compression ratio, avg bits, etc.)
    """
    
    indices_k: torch.Tensor
    indices_v: torch.Tensor
    bit_widths_k: torch.Tensor
    bit_widths_v: torch.Tensor
    attention_scores: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# KV Cache Manager
# ============================================================================

class KVCacheManager:
    """Manages compression and decompression of KV caches.
    
    Responsibilities:
    1. Compress KV vectors using AG-DBA
    2. Store compressed representation
    3. Decompress on demand (before attention computation)
    4. Update attention tracking for dynamic bit allocation
    5. Track performance metrics
    
    Usage:
        manager = KVCacheManager(
            num_heads=32,
            head_dim=128,
            target_bits_per_param=2.0,
        )
        
        # During prefill phase
        manager.compress_and_store(keys, values, attention_scores)
        
        # During decode phase (before attention)
        keys_decompressed, values_decompressed = manager.decompress()
        
        # After attention
        manager.update_attention_tracking(new_attention_scores)
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int = None,  # For GQA (Section 2.1)
        target_bits_per_param: float = 2.0,
        attention_decay_alpha: float = 0.85,
        device: str = "cuda",
        seed: int = 42,
    ):
        """Initialize KV cache manager.
        
        Args:
            num_heads: Total number of query heads
            head_dim: Dimension of each head (typically 128 for 8B models)
            num_key_value_heads: Number of KV heads (for GQA); defaults to num_heads
            target_bits_per_param: B_target for bit allocation (1.5 or 2.0)
            attention_decay_alpha: α for EMA (paper: 0.85)
            device: "cuda" or "cpu"
            seed: Reproducibility seed
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.device = device
        self.seed = seed
        
        # Create separate AG-DBA instances for K and V
        self.ag_dba_k = AGDBA(
            embedding_dim=head_dim,
            num_attention_heads=self.num_key_value_heads,
            target_bits_per_param=target_bits_per_param,
            attention_decay_alpha=attention_decay_alpha,
            device=device,
            seed=seed,
        )
        
        self.ag_dba_v = AGDBA(
            embedding_dim=head_dim,
            num_attention_heads=self.num_key_value_heads,
            target_bits_per_param=target_bits_per_param,
            attention_decay_alpha=attention_decay_alpha,
            device=device,
            seed=seed + 1,  # Different rotation for V
        )
        
        # Cache storage
        self.cache: Optional[CompressedKVCache] = None
        
        # Metrics
        self.metrics = {
            "total_compressed_tokens": 0,
            "total_original_size_mb": 0.0,
            "total_compressed_size_mb": 0.0,
        }
    
    def compress_and_store(
        self,
        keys: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        values: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        attention_scores_k: torch.Tensor,  # [batch, num_heads, seq_len]
        attention_scores_v: torch.Tensor,  # [batch, num_heads, seq_len]
    ) -> None:
        """Compress and store KV cache (prefill phase).
        
        Paper Section 3.4: "reallocation done asynchronously during prefill phase".
        
        Args:
            keys: [batch, seq_len, num_kv_heads, head_dim]
            values: [batch, seq_len, num_kv_heads, head_dim]
            attention_scores_k: [batch, num_heads, seq_len] attention weights to K
            attention_scores_v: [batch, num_heads, seq_len] attention weights to V
        """
        batch_size, seq_len, num_kv_heads, head_dim = keys.shape
        
        # Update importance scores
        self.ag_dba_k.update_attention_scores(attention_scores_k)
        self.ag_dba_v.update_attention_scores(attention_scores_v)
        
        # Allocate bits based on cumulative attention
        bit_widths_k = self.ag_dba_k.allocate_bits(seq_len)
        bit_widths_v = self.ag_dba_v.allocate_bits(seq_len)
        
        # Reshape for compression: [seq_len * num_kv_heads, head_dim]
        keys_flat = keys.reshape(batch_size * seq_len * num_kv_heads, head_dim)
        values_flat = values.reshape(batch_size * seq_len * num_kv_heads, head_dim)
        
        # Expand bit_widths to match flattened shape (repeat for each head)
        bit_widths_k_expanded = bit_widths_k.repeat_interleave(num_kv_heads)
        bit_widths_v_expanded = bit_widths_v.repeat_interleave(num_kv_heads)
        
        # Compress
        indices_k, _ = self.ag_dba_k.compress_kv(keys_flat, bit_widths_k_expanded)
        indices_v, _ = self.ag_dba_v.compress_kv(values_flat, bit_widths_v_expanded)
        
        # Reshape back
        indices_k = indices_k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        indices_v = indices_v.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        
        # Compute statistics
        stats_k = self.ag_dba_k.compute_statistics(bit_widths_k)
        stats_v = self.ag_dba_v.compute_statistics(bit_widths_v)
        
        # Store compressed cache
        self.cache = CompressedKVCache(
            indices_k=indices_k.cpu() if self.device == "cuda" else indices_k,
            indices_v=indices_v.cpu() if self.device == "cuda" else indices_v,
            bit_widths_k=bit_widths_k.cpu() if self.device == "cuda" else bit_widths_k,
            bit_widths_v=bit_widths_v.cpu() if self.device == "cuda" else bit_widths_v,
            attention_scores=self.ag_dba_k.importance_scores.cpu() if self.device == "cuda" else self.ag_dba_k.importance_scores,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "stats_k": stats_k,
                "stats_v": stats_v,
            },
        )
        
        # Update metrics
        self.metrics["total_compressed_tokens"] += seq_len * batch_size
        self.metrics["total_original_size_mb"] += (
            keys.numel() * 2 / (1024 ** 2) + values.numel() * 2 / (1024 ** 2)
        )
        self.metrics["total_compressed_size_mb"] += (
            (stats_k["compressed_size_mb"] * num_kv_heads) +
            (stats_v["compressed_size_mb"] * num_kv_heads)
        )
        
        logger.info(
            f"Compressed KV cache: seq_len={seq_len}, "
            f"avg_bits_k={stats_k['avg_bits_per_param']:.2f}, "
            f"avg_bits_v={stats_v['avg_bits_per_param']:.2f}, "
            f"compression_ratio_k={stats_k['compression_ratio']:.2f}x"
        )
    
    def decompress(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress full KV cache (decode phase).
        
        Called before attention computation: Q @ K^T, then attention @ V.
        
        Returns:
            keys: [batch, seq_len, num_kv_heads, head_dim]
            values: [batch, seq_len, num_kv_heads, head_dim]
        """
        if self.cache is None:
            raise RuntimeError("No compressed cache available. Call compress_and_store first.")
        
        batch_size = self.cache.metadata["batch_size"]
        seq_len = self.cache.metadata["seq_len"]
        num_kv_heads = self.cache.metadata["num_kv_heads"]
        head_dim = self.cache.metadata["head_dim"]
        
        # Move to device for decompression
        indices_k = self.cache.indices_k.to(self.device)
        indices_v = self.cache.indices_v.to(self.device)
        bit_widths_k = self.cache.bit_widths_k.to(self.device)
        bit_widths_v = self.cache.bit_widths_v.to(self.device)
        
        # Flatten for decompression
        indices_k_flat = indices_k.reshape(batch_size * seq_len * num_kv_heads, head_dim)
        indices_v_flat = indices_v.reshape(batch_size * seq_len * num_kv_heads, head_dim)
        bit_widths_k_expanded = bit_widths_k.repeat_interleave(num_kv_heads)
        bit_widths_v_expanded = bit_widths_v.repeat_interleave(num_kv_heads)
        
        # Decompress
        keys_decompressed = self.ag_dba_k.decompress_kv(indices_k_flat, bit_widths_k_expanded)
        values_decompressed = self.ag_dba_v.decompress_kv(indices_v_flat, bit_widths_v_expanded)
        
        # Reshape back
        keys_decompressed = keys_decompressed.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        values_decompressed = values_decompressed.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        
        return keys_decompressed, values_decompressed
    
    def update_attention_tracking(
        self,
        attention_scores_k: torch.Tensor,  # [batch, num_heads, seq_len]
        attention_scores_v: torch.Tensor,  # [batch, num_heads, seq_len]
    ) -> None:
        """Update attention tracking during decode phase.
        
        Paper Section 3.4: "updated in small chunks during decoding phase".
        
        Args:
            attention_scores_k: New attention scores to keys
            attention_scores_v: New attention scores to values
        """
        self.ag_dba_k.update_attention_scores(attention_scores_k)
        self.ag_dba_v.update_attention_scores(attention_scores_v)
    
    def get_statistics(self) -> dict:
        """Get compression statistics.
        
        Returns:
            stats: Dict with compression ratio, sizes, etc.
        """
        total_tokens = self.metrics["total_compressed_tokens"]
        if total_tokens == 0:
            return {}
        
        total_original = self.metrics["total_original_size_mb"]
        total_compressed = self.metrics["total_compressed_size_mb"]
        
        return {
            "total_tokens_compressed": total_tokens,
            "total_original_size_mb": total_original,
            "total_compressed_size_mb": total_compressed,
            "overall_compression_ratio": (
                total_original / total_compressed
                if total_compressed > 0
                else float('inf')
            ),
            "cache_metadata": self.cache.metadata if self.cache else {},
        }
