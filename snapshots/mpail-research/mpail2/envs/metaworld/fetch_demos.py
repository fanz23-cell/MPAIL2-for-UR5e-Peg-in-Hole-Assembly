"""Download Meta-World demonstration assets from Hugging Face Hub into this stack's demos folder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_local_dir() -> Path:
    # This file lives in mpail2/envs/metaworld/ — demos/ is a sibling directory.
    return Path(__file__).resolve().parent / "demos"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face dataset repo into mpail2/envs/metaworld/demos/ "
            "(same layout produced by data_collection / collect_mw_sb3_demos.sh)."
        ),
        epilog=(
            "Examples:\n"
            "  export MPAIL_METAWORLD_HF_REPO=UWRobotics/mw-metaworld-demos\n"
            "  python -m mpail2.envs.metaworld.fetch_demos\n\n"
            "  python -m mpail2.envs.metaworld.fetch_demos --repo-id UWRobotics/mw-metaworld-demos\n\n"
            "Private repos: set HF_TOKEN or run `hf auth login`.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("MPAIL_METAWORLD_HF_REPO"),
        help="Dataset repo id (org/name). Default: MPAIL_METAWORLD_HF_REPO.",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=f"Destination directory (default: {_default_local_dir()})",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision (branch, tag, or commit).",
    )
    args = p.parse_args()

    if not args.repo_id or not str(args.repo_id).strip():
        p.error(
            "Pass --repo-id or set MPAIL_METAWORLD_HF_REPO (e.g. UWRobotics/mw-metaworld-demos)."
        )

    local_dir = args.local_dir if args.local_dir is not None else _default_local_dir()
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub\n"
            "Or: pip install mpail2[metaworld]"
        ) from e

    print(f"[INFO] repo_id={args.repo_id!r}  local_dir={local_dir}")
    snapshot_download(
        repo_id=args.repo_id.strip(),
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"[INFO] Done. Demos are under {local_dir}")


if __name__ == "__main__":
    main()
    sys.exit(0)
