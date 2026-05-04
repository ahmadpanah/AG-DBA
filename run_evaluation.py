"""
run_evaluation.py
~~~~~~~~~~~~~~~~~
Quick-start evaluation script for AG-DBA.

Usage:
    python run_evaluation.py --config configs/ag_dba_config.yaml --benchmark niah
    python run_evaluation.py --config configs/ag_dba_config.yaml --benchmark longbench_e

Paper: "Attention-Guided Dynamic Bit Allocation for Long-Context LLM KV Caches 
via Near-Optimal Vector Quantization"
"""

import argparse
import logging
import sys
from pathlib import Path

from eval.evaluator import AGDBAEvaluator
from utils.config import set_reproducibility

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="AG-DBA Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate NIAH benchmark at multiple context lengths
  python run_evaluation.py --benchmark niah --context-lengths 4000 16000 32000

  # Evaluate LongBench-E with custom task subset
  python run_evaluation.py --benchmark longbench_e --tasks single_qa multi_qa

  # Run all evaluations
  python run_evaluation.py --all-benchmarks
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "ag_dba_config.yaml",
        help="Path to AG-DBA config YAML",
    )
    
    parser.add_argument(
        "--benchmark",
        choices=["niah", "longbench_e"],
        help="Benchmark to run",
    )
    
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run all benchmarks",
    )
    
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[4000, 16000, 32000, 64000, 100000, 104000],
        help="Context lengths for NIAH",
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["single_qa", "multi_qa", "summarization", "few_shot", "synthetic", "code"],
        help="Tasks for LongBench-E",
    )
    
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID",
    )
    
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device (cuda:0, cpu, etc.)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable deterministic behavior",
    )
    
    args = parser.parse_args()
    
    # Set reproducibility
    set_reproducibility(seed=args.seed, deterministic=not args.no_deterministic)
    
    # Create evaluator
    logger.info(f"Initializing evaluator with model: {args.model}")
    evaluator = AGDBAEvaluator(
        model_name=args.model,
        config_path=args.config,
        device=args.device,
    )
    
    # Run benchmarks
    if args.all_benchmarks or args.benchmark == "niah":
        logger.info("Running NIAH benchmark...")
        try:
            evaluator.evaluate_niah(
                context_lengths=args.context_lengths,
                depth_percentiles=list(range(0, 101, 10)),
            )
        except Exception as e:
            logger.error(f"NIAH evaluation failed: {e}", exc_info=True)
    
    if args.all_benchmarks or args.benchmark == "longbench_e":
        logger.info("Running LongBench-E benchmark...")
        try:
            evaluator.evaluate_longbench_e(
                tasks=args.tasks,
                n_samples_per_task=5,
            )
        except Exception as e:
            logger.error(f"LongBench-E evaluation failed: {e}", exc_info=True)
    
    # Benchmark latency
    if args.all_benchmarks or args.benchmark:
        logger.info("Benchmarking latency...")
        try:
            latency_stats = evaluator.benchmark_latency()
            logger.info(f"Throughput: {latency_stats['throughput_tokens_per_sec']:.1f} tokens/sec")
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}", exc_info=True)
    
    # Report results
    evaluator.report_results()
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
