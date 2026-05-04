"""
eval/data_loaders.py
~~~~~~~~~~~~~~~~~~~~
Data loading stubs for the two evaluation benchmarks used in the paper.

Paper Section 4.2:
  - Needle-In-A-Haystack (NIAH): deterministic factual retrieval benchmark.
  - LongBench-E: six-domain generative comprehension suite (10k–30k tokens).

Both loaders return a unified SampleBatch dataclass so the evaluation
harness (eval/evaluator.py) can treat them identically.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dataclass
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A single evaluation sample.

    Attributes:
        sample_id:    Unique identifier string.
        context:      The long input context (the "haystack" or document).
        question:     The query / instruction appended after the context.
        answer:       Ground-truth answer string (for metric computation).
        context_len:  Approximate token count of *context* (pre-tokenisation estimate).
        meta:         Arbitrary metadata dict (needle depth, task name, etc.).
    """

    sample_id: str
    context: str
    question: str
    answer: str
    context_len: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NIAH Loader
# ---------------------------------------------------------------------------

# The exact needle string used in the public NIAH benchmark:
# https://github.com/gkamradt/LLMTest_NeedleInAHaystack  (Section 4.2, Ref [12])
_DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a "
    "sunny day."
)

_NIAH_QUESTION = (
    "What is the best thing to do in San Francisco? "
    "Be very specific and concise."
)

_HAYSTACK_FILLER = (
    "Huey, Dewey, and Louie went to the market. They bought apples, bananas, and carrots. "
    "The sun was shining brightly as they walked home along the river. "
)


def _build_haystack(target_words: int, needle: str, depth_pct: float) -> str:
    """Build a synthetic haystack string of ~*target_words* words.

    The *needle* is inserted at position *depth_pct* % of the haystack.

    Args:
        target_words: Approximate word count of the full document.
        needle:       The needle sentence to hide inside the haystack.
        depth_pct:    Float in [0, 100] indicating insertion depth.

    Returns:
        Full haystack string (needle embedded).
    """
    # Repeat the filler sentence until we have enough words.
    filler_words = _HAYSTACK_FILLER.split()
    repeats = max(1, target_words // len(filler_words) + 1)
    words = (filler_words * repeats)[:target_words]

    insert_idx = int(len(words) * depth_pct / 100.0)
    needle_words = needle.split()
    words = words[:insert_idx] + needle_words + words[insert_idx:]
    return " ".join(words)


class NIAHDataLoader:
    """Needle-In-A-Haystack data loader.

    Reproduces the evaluation protocol from Section 4.2:
      - Context lengths: [4k, 16k, 32k, 64k, 100k, 104k] tokens.
      - Needle depths: 10 evenly-spaced depth percentiles (0%–100%).
      - Metric: binary recall (exact-match of needle string).

    NOTE: Real token counts depend on the tokenizer.  We use a words→tokens
    approximation of 0.75 tokens/word (English average) for stub purposes.
    Replace `_build_haystack` with actual dataset downloads for full fidelity.

    Args:
        context_lengths: List of target context lengths in tokens.
        depth_percentiles: List of depth values in [0, 100].
        needle: The "needle" sentence to retrieve.
        words_per_token: Approximate words-per-token ratio for haystack sizing.
        seed: Random seed for filler shuffling.
    """

    WORDS_PER_TOKEN: float = 0.75  # empirical English approximation

    def __init__(
        self,
        context_lengths: List[int] | None = None,
        depth_percentiles: List[int] | None = None,
        needle: str = _DEFAULT_NEEDLE,
        words_per_token: float = WORDS_PER_TOKEN,
        seed: int = 42,
    ) -> None:
        self.context_lengths = context_lengths or [4000, 16000, 32000, 64000, 100000, 104000]
        self.depth_percentiles = depth_percentiles or list(range(0, 101, 10))
        self.needle = needle
        self.words_per_token = words_per_token
        self.seed = seed

    def __iter__(self) -> Iterator[Sample]:
        """Yield one Sample per (context_length, depth) combination."""
        rng = random.Random(self.seed)
        _ = rng  # available for future use (e.g., random needle positions)

        for ctx_len in self.context_lengths:
            for depth in self.depth_percentiles:
                target_words = int(ctx_len * self.words_per_token)
                haystack = _build_haystack(target_words, self.needle, float(depth))
                sample_id = f"niah_ctx{ctx_len}_depth{depth}"
                yield Sample(
                    sample_id=sample_id,
                    context=haystack,
                    question=_NIAH_QUESTION,
                    answer=self.needle,
                    context_len=ctx_len,
                    meta={"depth_pct": depth, "benchmark": "niah"},
                )

    def __len__(self) -> int:
        return len(self.context_lengths) * len(self.depth_percentiles)


# ---------------------------------------------------------------------------
# LongBench-E Loader
# ---------------------------------------------------------------------------

# Task definitions matching Section 4.2 / Table 2 domains.
_LONGBENCH_E_TASKS = [
    "single_qa",
    "multi_qa",
    "summarization",
    "few_shot",
    "synthetic",
    "code",
]

# Metric per task (Section 4.5)
LONGBENCH_E_METRICS: dict[str, str] = {
    "single_qa": "f1",
    "multi_qa": "f1",
    "summarization": "rouge",
    "few_shot": "em",
    "synthetic": "em",
    "code": "em",
}


class LongBenchEDataLoader:
    """LongBench-E data loader stub.

    Paper Section 4.2: six reasoning/comprehension tasks, context lengths
    10k–30k tokens.  This stub generates synthetic placeholder samples so
    that the evaluation pipeline can be validated end-to-end.

    For production use, replace `_synthetic_sample` with HuggingFace
    dataset loading:

        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", name=task_name, split="test")

    Args:
        tasks: Subset of LongBench-E tasks to load.
        n_samples_per_task: Number of samples to yield per task (stub only).
        context_len_range: (min, max) token range for synthetic context.
        seed: Random seed.
    """

    def __init__(
        self,
        tasks: List[str] | None = None,
        n_samples_per_task: int = 5,
        context_len_range: tuple[int, int] = (10_000, 30_000),
        seed: int = 42,
    ) -> None:
        self.tasks = tasks or _LONGBENCH_E_TASKS
        unknown = set(self.tasks) - set(_LONGBENCH_E_TASKS)
        if unknown:
            raise ValueError(f"Unknown LongBench-E tasks: {unknown}")
        self.n_samples_per_task = n_samples_per_task
        self.context_len_range = context_len_range
        self.seed = seed

    def _synthetic_sample(
        self, task: str, idx: int, ctx_len: int, rng: random.Random
    ) -> Sample:
        """Generate a dummy sample for pipeline testing.

        REPLACE THIS with real dataset loading in production.
        """
        words_per_token = 0.75
        n_words = int(ctx_len * words_per_token)

        # Synthetic context: random ASCII words.
        context_words = [
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
            for _ in range(n_words)
        ]
        context = " ".join(context_words)
        question = f"[STUB] Task={task} idx={idx}: What is the main topic?"
        answer = "[STUB_ANSWER]"

        return Sample(
            sample_id=f"longbench_e_{task}_{idx}",
            context=context,
            question=question,
            answer=answer,
            context_len=ctx_len,
            meta={"task": task, "benchmark": "longbench_e", "metric": LONGBENCH_E_METRICS[task]},
        )

    def __iter__(self) -> Iterator[Sample]:
        """Yield samples across all tasks."""
        rng = random.Random(self.seed)
        lo, hi = self.context_len_range
        for task in self.tasks:
            for idx in range(self.n_samples_per_task):
                ctx_len = rng.randint(lo, hi)
                yield self._synthetic_sample(task, idx, ctx_len, rng)

    def __len__(self) -> int:
        return len(self.tasks) * self.n_samples_per_task