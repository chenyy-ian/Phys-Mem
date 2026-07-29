import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from evaluation.visualization.plot_case_study import plot_case_study
    from evaluation.visualization.plot_evidence import (
        plot_confidence_curve,
        plot_decision_confidence_histogram,
        plot_evidence_score_curve,
        plot_policy_confidence_distribution,
        plot_unified_memory_score,
    )
    from evaluation.visualization.plot_memory import (
        plot_memory_lifetime,
        plot_memory_window_timeline,
        plot_window_evolution,
    )
    from evaluation.visualization.plot_runtime import plot_memory_size_curve, plot_runtime_curve
    from evaluation.visualization.plot_strategy import (
        plot_action_magnitude_vs_strategy,
        plot_geometry_vs_semantic,
        plot_similarity_vs_strategy,
        plot_strategy_distribution,
        plot_strategy_timeline,
        plot_transition_matrix,
    )
    from evaluation.visualization.plot_utils import (
        apply_paper_style,
        ensure_output_dirs,
        read_records,
        write_summary,
    )
else:
    from .plot_case_study import plot_case_study
    from .plot_evidence import (
        plot_confidence_curve,
        plot_decision_confidence_histogram,
        plot_evidence_score_curve,
        plot_policy_confidence_distribution,
        plot_unified_memory_score,
    )
    from .plot_memory import plot_memory_lifetime, plot_memory_window_timeline, plot_window_evolution
    from .plot_runtime import plot_memory_size_curve, plot_runtime_curve
    from .plot_strategy import (
        plot_action_magnitude_vs_strategy,
        plot_geometry_vs_semantic,
        plot_similarity_vs_strategy,
        plot_strategy_distribution,
        plot_strategy_timeline,
        plot_transition_matrix,
    )
    from .plot_utils import apply_paper_style, ensure_output_dirs, read_records, write_summary


def generate_all_figures(records, output_dir: str):
    apply_paper_style()
    output_dirs = ensure_output_dirs(output_dir)

    plot_strategy_timeline(records, output_dirs)
    plot_memory_window_timeline(records, output_dirs)
    plot_unified_memory_score(records, output_dirs)
    plot_evidence_score_curve(records, output_dirs)
    plot_strategy_distribution(records, output_dirs)
    plot_transition_matrix(records, output_dirs)
    plot_memory_lifetime(records, output_dirs)
    plot_window_evolution(records, output_dirs)
    plot_runtime_curve(records, output_dirs)
    plot_confidence_curve(records, output_dirs)
    plot_memory_size_curve(records, output_dirs)
    plot_decision_confidence_histogram(records, output_dirs)
    plot_similarity_vs_strategy(records, output_dirs)
    plot_geometry_vs_semantic(records, output_dirs)
    plot_action_magnitude_vs_strategy(records, output_dirs)
    plot_policy_confidence_distribution(records, output_dirs)
    plot_case_study(records, output_dirs)

    return write_summary(records, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from ExperimentRecord.csv only.")
    parser.add_argument("--records", required=True, help="Path to ExperimentRecord.csv")
    parser.add_argument("--output_dir", default="evaluation/results/figures")
    args = parser.parse_args()

    records = read_records(args.records)
    if not records:
        raise ValueError(f"No ExperimentRecord rows found: {args.records}")
    summary = generate_all_figures(records, args.output_dir)
    print(f"Generated figures in {args.output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
