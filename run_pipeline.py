"""
Pipeline orchestrator — runs all stages in sequence.

Usage:
    python run_pipeline.py
"""

import sys
from pipeline.train import train


def main():
    print("=" * 50)
    print("MLOps Pipeline — Wine Quality Classifier")
    print("=" * 50)

    print("\n[1/1] Training model with MLflow tracking...")
    metrics = train()

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print(f"  Train accuracy: {metrics['train_accuracy']:.1%}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.1%}")
    print(f"  Top feature:    {metrics['top_feature']}")
    print("=" * 50)

    # Exit with error if accuracy is below threshold (used in CI)
    if metrics["test_accuracy"] < 0.70:
        print(f"\nFAIL: test accuracy {metrics['test_accuracy']:.1%} is below 70% threshold")
        sys.exit(1)

    print("\nPASS: model meets quality threshold")


if __name__ == "__main__":
    main()
