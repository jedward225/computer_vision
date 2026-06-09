from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready figures.")
    parser.add_argument("--output-dir", default="results/figures", help="Directory for generated figures.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing experiment history CSV files.")
    return parser.parse_args()


def read_history(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                values.setdefault(key, []).append(parsed)
    return values


def plot_histories(logs_dir: Path, output_dir: Path) -> int:
    import matplotlib.pyplot as plt

    history_files = sorted(logs_dir.glob("*/history.csv"))
    if not history_files:
        return 0

    for metric in ("train_loss", "val_loss", "mean_dice", "mean_iou"):
        plt.figure(figsize=(7, 4))
        plotted = False
        for history_file in history_files:
            history = read_history(history_file)
            if metric not in history or "epoch" not in history:
                continue
            plt.plot(history["epoch"], history[metric], label=history_file.parent.name)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=200)
        plt.close()
    return len(history_files)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = plot_histories(Path(args.logs_dir), output_dir)
    if count:
        print(f"Generated training-curve figures from {count} history files into {output_dir}")
    else:
        print(f"No history files found under {args.logs_dir}; qualitative figures will be generated after training.")


if __name__ == "__main__":
    main()
