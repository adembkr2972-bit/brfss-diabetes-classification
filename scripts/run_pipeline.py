import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP_DEFINITIONS = [
    (2, "Data audit", PROJECT_ROOT / "scripts" / "preprocessing" / "2_data_audit.py"),
    (3, "Target construction", PROJECT_ROOT / "scripts" / "preprocessing" / "3_target_construction.py"),
    (4, "Feature selection", PROJECT_ROOT / "scripts" / "preprocessing" / "4_feature_selection.py"),
    (5, "Build pipelines", PROJECT_ROOT / "scripts" / "preprocessing" / "5_build_pipelines.py"),
    (6, "Split data", PROJECT_ROOT / "scripts" / "preprocessing" / "6_split_data.py"),
    (7, "Train baseline models", PROJECT_ROOT / "scripts" / "training" / "7_train_baseline_models.py"),
    (8, "Hyperparameter tuning", PROJECT_ROOT / "scripts" / "training" / "8_hyperparameter_tuning.py"),
    (9, "Evaluate models", PROJECT_ROOT / "scripts" / "validation" / "9_evaluate_models.py"),
    (10, "Model elimination", PROJECT_ROOT / "scripts" / "validation" / "10_model_elimination.py"),
    (11, "Final test evaluation", PROJECT_ROOT / "scripts" / "experiments" / "11_final_evaluation.py"),
    (12, "Error analysis", PROJECT_ROOT / "scripts" / "experiments" / "12_error_analysis.py"),
    (13, "Comparison report", PROJECT_ROOT / "scripts" / "experiments" / "13_comparison_report.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BRFSS diabetes pipeline in order."
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=2,
        help="First numbered pipeline step to run.",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=10,
        help="Last numbered pipeline step to run. Default stops before final test use.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Allow steps 11-13 to run. Use this only when you are ready for the one-time final test stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps that would run without executing them.",
    )
    parser.add_argument(
        "--dry-run-on-validation",
        action="store_true",
        help="Run validation-only equivalents of steps 11-13 under outputs/dry_run; this never reads the holdout.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    valid_steps = {step for step, _, _ in STEP_DEFINITIONS}
    if args.from_step not in valid_steps:
        raise SystemExit(f"Invalid --from-step: {args.from_step}")
    if args.to_step not in valid_steps:
        raise SystemExit(f"Invalid --to-step: {args.to_step}")
    if args.from_step > args.to_step:
        raise SystemExit("--from-step cannot be greater than --to-step")
    if args.to_step >= 11 and not args.include_test:
        raise SystemExit(
            "Steps 11-13 touch the held-out test set. Re-run with --include-test to confirm you want that stage."
        )


def selected_steps(from_step: int, to_step: int) -> list[tuple[int, str, Path]]:
    return [
        step_def
        for step_def in STEP_DEFINITIONS
        if from_step <= step_def[0] <= to_step
    ]


def run_step(step_number: int, step_name: str, script_path: Path) -> None:
    started = time.time()
    print(f"\n[{step_number}] {step_name}")
    print(f"Running: {script_path}")
    result = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    duration = time.time() - started
    if result.returncode != 0:
        raise SystemExit(
            f"Step {step_number} failed with exit code {result.returncode} after {duration:.1f}s: {script_path}"
        )
    print(f"Completed step {step_number} in {duration:.1f}s")


def run_validation_dry_run(script_path: Path) -> None:
    started = time.time()
    print("\n[11-13] Validation-only rehearsal of evaluation, error analysis, and comparison")
    print(f"Running: {script_path}")
    result = subprocess.run(
        [sys.executable, "-u", "-B", str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    duration = time.time() - started
    if result.returncode != 0:
        raise SystemExit(
            f"Validation-only steps 11-13 rehearsal failed with exit code {result.returncode} after {duration:.1f}s: {script_path}"
        )
    print(f"Completed validation-only steps 11-13 rehearsal in {duration:.1f}s")


def main() -> None:
    args = parse_args()

    if args.dry_run_on_validation:
        if args.include_test or args.from_step >= 11 or args.to_step >= 11:
            raise SystemExit(
                "--dry-run-on-validation is validation-only; do not combine it with holdout-stage options."
            )
        dry_run_script = PROJECT_ROOT / "scripts" / "experiments" / "dry_run_validation.py"
        print("Validation-only dry run: steps 11-13 equivalents; holdout access is disabled.")
        run_validation_dry_run(dry_run_script)
        return

    validate_args(args)
    steps = selected_steps(args.from_step, args.to_step)

    print("BRFSS diabetes pipeline runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Step range: {args.from_step} -> {args.to_step}")
    if args.to_step >= 11:
        print("Test-stage execution enabled: yes")
    else:
        print("Test-stage execution enabled: no")

    if args.dry_run:
        for step_number, step_name, script_path in steps:
            print(f"[dry-run] {step_number}: {step_name} -> {script_path}")
        return

    total_started = time.time()
    for step_number, step_name, script_path in steps:
        run_step(step_number, step_name, script_path)
    total_duration = time.time() - total_started

    print(f"\nPipeline completed successfully in {total_duration / 60:.2f} minutes.")
    if args.to_step < 11:
        print("You stopped before test-set evaluation. When you are satisfied with the shortlist, re-run with --from-step 11 --to-step 13 --include-test.")


if __name__ == "__main__":
    main()
