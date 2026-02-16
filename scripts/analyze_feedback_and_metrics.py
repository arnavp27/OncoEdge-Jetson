#!/usr/bin/env python3
"""Analyze collected feedback and metrics from OncoEdge pipeline.

This script demonstrates all analytics capabilities:
1. Feedback analysis (doctor agreement rate)
2. Confusion matrix computation
3. False Negative Rate (FNR) calculation
4. Confidence distribution statistics
5. Latency performance metrics
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feedback import FeedbackStore
from src.utils.metrics import MetricsTracker
from src.config import load_config
import numpy as np


def analyze_feedback():
    """Analyze doctor feedback data."""
    print("=" * 60)
    print("FEEDBACK ANALYSIS")
    print("=" * 60)

    config = load_config()
    store = FeedbackStore(feedback_dir=config.paths.feedback_dir)

    # Load all feedback
    all_feedback = store.load_all_feedback()
    print(f"\nTotal feedback entries: {len(all_feedback)}")

    if not all_feedback:
        print("  No feedback data yet. Submit feedback via Streamlit UI.")
        return

    # Compute agreement rate
    agreement = store.compute_doctor_agreement_rate()
    print(f"\nDoctor-AI Agreement:")
    print(f"  Total assessments: {agreement['total']}")
    print(f"  Agreed: {agreement['agreed']}")
    print(f"  Disagreed: {agreement['disagreed']}")
    print(f"  Agreement rate: {agreement['agreement_rate']:.1%}")

    # Show disagreements
    disagreements = [f for f in all_feedback if not f.get('doctor_agrees')]
    if disagreements:
        print(f"\nDisagreement cases ({len(disagreements)}):")
        for f in disagreements:
            print(f"  Inference {f['inference_id']}:")
            print(f"    AI predicted: {f['ai_prediction']} ({f['ai_risk_level']})")
            print(f"    Doctor says: {f.get('doctor_classification', 'N/A')}")
            if f.get('doctor_notes'):
                print(f"    Notes: {f['doctor_notes']}")

    # Load ground truth annotations
    ground_truths = store.load_ground_truths()
    print(f"\nGround truth annotations: {len(ground_truths)}")
    if ground_truths:
        for gt in ground_truths:
            print(f"  {gt['inference_id']}: {gt['ground_truth_class']} (verified by {gt['verification_method']})")


def analyze_metrics():
    """Analyze inference metrics (confusion matrix, FNR, confidence, latency)."""
    print("\n" + "=" * 60)
    print("METRICS ANALYSIS")
    print("=" * 60)

    config = load_config()
    tracker = MetricsTracker(config=config.metrics)

    # Check if log file exists
    if not Path(config.metrics.log_file).exists():
        print(f"\nNo metrics log found at: {config.metrics.log_file}")
        print("Run some inferences via Streamlit to collect metrics.")
        return

    # Load metrics from log
    with open(config.metrics.log_file, 'r') as f:
        import json
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"\nTotal inference entries: {len(entries)}")

    if not entries:
        return

    # Show recent inferences
    print(f"\nRecent inferences (last 5):")
    for entry in entries[-5:]:
        print(f"  {entry['timestamp']} | {entry['inference_id']}")
        print(f"    Risk: {entry['risk_level']} (score: {entry['risk_score']:.2f})")
        print(f"    Detections: {entry['num_detections']}")
        if entry.get('timing'):
            print(f"    Latency: {entry['timing'].get('pipeline_total_ms', 0):.0f}ms")

    # Compute confusion matrix (if ground truths are available)
    has_ground_truth = any(e.get('ground_truth') is not None for e in entries)
    if has_ground_truth:
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX (with ground truth)")
        print("-" * 60)
        cm = tracker.get_confusion_matrix()
        class_names = ["OSCC", "OPMD", "Normal"]

        # Print header
        print("\n             " + "".join(f"{cn:>10}" for cn in class_names) + " (predicted)")
        print("             " + "-" * (10 * len(class_names)))

        # Print rows
        for i, true_class in enumerate(class_names):
            row = "".join(f"{cm[i, j]:>10}" for j in range(len(class_names)))
            print(f"{true_class:>10} | {row}")

        print("\n(rows = true class, columns = predicted class)")

        # False Negative Rate
        fnr = tracker.get_false_negative_rate(positive_classes=["OSCC", "OPMD"])
        print(f"\nFalse Negative Rate (OSCC/OPMD): {fnr:.2%}")
        print("  (Critical metric: how often do we miss cancer/precancer?)")
    else:
        print("\nNo ground truth annotations yet.")
        print("Use FeedbackStore.record_ground_truth() to add pathology results.")

    # Confidence distribution
    print("\n" + "-" * 60)
    print("CONFIDENCE DISTRIBUTION")
    print("-" * 60)
    conf_stats = tracker.get_confidence_distribution()
    if conf_stats:
        print(f"  Count: {conf_stats['count']}")
        print(f"  Mean: {conf_stats['mean']:.3f}")
        print(f"  Std: {conf_stats['std']:.3f}")
        print(f"  Min: {conf_stats['min']:.3f}")
        print(f"  Median: {conf_stats['median']:.3f}")
        print(f"  Max: {conf_stats['max']:.3f}")
        print(f"  P5-P95: [{conf_stats['p5']:.3f}, {conf_stats['p95']:.3f}]")

    # Latency statistics
    print("\n" + "-" * 60)
    print("LATENCY STATISTICS")
    print("-" * 60)
    latency_stats = tracker.get_latency_stats()
    if latency_stats:
        for metric, stats in latency_stats.items():
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.1f}ms")
            print(f"  Median: {stats['median']:.1f}ms")
            print(f"  P95: {stats['p95']:.1f}ms")
            print(f"  Max: {stats['max']:.1f}ms")


def main():
    """Run all analyses."""
    print("\nOncoEdge Analytics Report")
    print("=" * 60)

    analyze_feedback()
    analyze_metrics()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
