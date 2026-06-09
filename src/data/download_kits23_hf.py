from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


DEFAULT_REPO = "Angelou0516/kits23"
HF_PREFIX = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a KiTS23 subset from HuggingFace.")
    parser.add_argument("--num-cases", type=int, default=None, help="Download case_00000 ... case_N for quick experiments.")
    parser.add_argument("--case", nargs="*", default=None, help="Explicit case ids, e.g. 00000 00001.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="HuggingFace dataset repo id.")
    parser.add_argument("--output", default="data/kits23/dataset", help="Output dataset directory.")
    return parser.parse_args()


def normalized_cases(cases: list[str] | None, num_cases: int | None) -> list[str]:
    if cases is None:
        if num_cases is None:
            raise ValueError("Pass --num-cases or --case to avoid downloading the full dataset accidentally.")
        cases = [f"{idx:05d}" for idx in range(num_cases)]
    return [case if case.startswith("case_") else f"case_{int(case):05d}" for case in cases]


def copy_file(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for case_id in tqdm(normalized_cases(args.case, args.num_cases), desc="download KiTS23"):
        for filename in ("imaging.nii.gz", "segmentation.nii.gz"):
            remote_path = f"{HF_PREFIX}/{case_id}/{filename}"
            local_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=remote_path)
            copy_file(local_path, output / case_id / filename)


if __name__ == "__main__":
    main()
