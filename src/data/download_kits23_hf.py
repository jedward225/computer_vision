from __future__ import annotations

import argparse
import os
import shutil
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path


TRAINING_CASE_NUMBERS = list(range(300)) + list(range(400, 589))
DEFAULT_REPO_ID = "neheller/KiTS-Challenge-Imaging"
HF_PREFIX = "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KiTS23 imaging volumes from the official Hugging Face dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of training cases to download.")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Explicit case IDs, e.g. case_00000 00001.")
    parser.add_argument("--cache-dir", default="data/kits23_hf_cache", help="Local HF download directory.")
    parser.add_argument("--output-dir", default="data/kits23/kits23_repo/dataset", help="Project dataset directory.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking downloaded files.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint. Use https://huggingface.co or https://hf-mirror.com.",
    )
    return parser.parse_args()


def normalize_case_id(value: str | int) -> str:
    text = str(value)
    if text.startswith("case_"):
        return text
    return f"case_{int(text):05d}"


def selected_cases(args: argparse.Namespace) -> list[str]:
    if args.case_ids:
        return [normalize_case_id(case_id) for case_id in args.case_ids]
    return [normalize_case_id(case_num) for case_num in TRAINING_CASE_NUMBERS[: args.num_cases]]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)


def direct_url(endpoint: str, repo_id: str, filename: str) -> str:
    endpoint = endpoint.rstrip("/")
    return f"{endpoint}/datasets/{repo_id}/resolve/main/{filename}"


def download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".partial.{dst.name}")
    if dst.exists():
        return
    if tmp.exists():
        tmp.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "cv-final-kits23-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
            if tmp.exists():
                tmp.unlink()
            redirected = urllib.parse.urljoin(url, exc.headers["Location"])
            download_url(redirected, dst)
            return
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    shutil.move(str(tmp), str(dst))


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cases = selected_cases(args)
    print(f"Downloading {len(cases)} KiTS23 cases from {args.repo_id}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Output directory: {output_dir}")

    for idx, case_id in enumerate(cases, start=1):
        case_out = output_dir / case_id
        image_out = case_out / "imaging.nii.gz"
        mask_out = case_out / "segmentation.nii.gz"
        if image_out.exists() and mask_out.exists():
            print(f"[{idx}/{len(cases)}] {case_id}: already complete")
            continue

        print(f"[{idx}/{len(cases)}] {case_id}: downloading image")
        if image_out.exists():
            continue
        hf_path = f"{HF_PREFIX}/{case_id}.nii.gz"
        cached_path = cache_dir / args.repo_id / hf_path
        if not cached_path.exists():
            url = direct_url(args.endpoint, args.repo_id, hf_path)
            try:
                download_url(url, cached_path)
            except Exception as exc:
                fallback_endpoint = "https://huggingface.co"
                if args.endpoint.rstrip("/") != fallback_endpoint:
                    print(f"  mirror failed for imaging; falling back to {fallback_endpoint}")
                    download_url(direct_url(fallback_endpoint, args.repo_id, hf_path), cached_path)
                else:
                    raise exc
        link_or_copy(cached_path, image_out, copy=args.copy)

    complete = sum(
        int((output_dir / case_id / "imaging.nii.gz").exists() and (output_dir / case_id / "segmentation.nii.gz").exists())
        for case_id in cases
    )
    print(f"Complete cases in {output_dir}: {complete}/{len(cases)}")


if __name__ == "__main__":
    main()
