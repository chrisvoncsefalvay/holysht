#!/usr/bin/env python3
"""Stage and publish the HOLYSHT Hugging Face kernel repository.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tomllib
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = ROOT / "dist" / "hf-kernel-repo"
SYNCED_ROOT_FILES = (
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("build.toml", "build.toml"),
    ("flake.nix", "flake.nix"),
    ("flake.lock", "flake.lock"),
)


def _load_build_config() -> dict:
    with (ROOT / "build.toml").open("rb") as handle:
        return tomllib.load(handle)


def _repo_id_from_config(config: dict) -> str:
    hub = config.get("general", {}).get("hub", {})
    repo_id = hub.get("repo-id")
    if not repo_id:
        raise SystemExit("build.toml is missing [general.hub].repo-id")
    return str(repo_id)


def _version_branch_from_config(config: dict) -> str:
    version = config.get("general", {}).get("version")
    if version is None:
        raise SystemExit("build.toml is missing [general].version")
    return f"v{version}"


def _project_name_from_config(config: dict) -> str:
    name = config.get("general", {}).get("name")
    if not name:
        raise SystemExit("build.toml is missing [general].name")
    return str(name)


def _card_source() -> Path:
    for candidate in (ROOT / "build" / "CARD.md", ROOT / "CARD.md"):
        if candidate.exists():
            return candidate
    raise SystemExit("could not find CARD.md in either build/ or the repository root")


def _variant_dirs(build_dir: Path) -> list[Path]:
    variants = []
    for child in sorted(build_dir.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            variants.append(child)
    return variants


def _validate_variant_dir(repo_name: str, variant_dir: Path) -> None:
    compat_dir = variant_dir / repo_name.replace("-", "_")
    if not (variant_dir / "__init__.py").exists():
        raise SystemExit(f"missing __init__.py in build variant {variant_dir.name}")
    if not compat_dir.is_dir():
        raise SystemExit(
            f"missing compatibility package {compat_dir} in build variant "
            f"{variant_dir.name}"
        )
    if not (compat_dir / "__init__.py").exists():
        raise SystemExit(
            f"missing compatibility __init__.py in build variant {variant_dir.name}"
        )


def stage_kernel_repo(build_dir: Path, stage_dir: Path, project_name: str) -> Path:
    build_dir = build_dir.resolve()
    stage_dir = stage_dir.resolve()
    variant_dirs = _variant_dirs(build_dir)
    if not variant_dirs:
        raise SystemExit(
            "no kernel-builder variants were found under build/. "
            "Run `nix run -L --max-jobs 1 --cores 4 .#build-and-copy` first."
        )

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "build").mkdir(parents=True, exist_ok=True)

    for variant_dir in variant_dirs:
        _validate_variant_dir(project_name, variant_dir)
        shutil.copytree(variant_dir, stage_dir / "build" / variant_dir.name)

    readme_src = _card_source()
    shutil.copy2(readme_src, stage_dir / "README.md")

    for src_name, dst_name in SYNCED_ROOT_FILES[1:]:
        src = ROOT / src_name
        if src.exists():
            shutil.copy2(src, stage_dir / dst_name)

    gitattributes = (
        "*.so binary\n"
        "*.dylib binary\n"
        "*.pyd binary\n"
        "*.dll binary\n"
    )
    (stage_dir / ".gitattributes").write_text(gitattributes, encoding="utf-8")
    return stage_dir


def _ensure_version_branch(api: HfApi, repo_id: str, version_branch: str) -> None:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    existing = {branch.name for branch in refs.branches}
    if version_branch not in existing:
        api.create_branch(
            repo_id=repo_id,
            repo_type="model",
            branch=version_branch,
            revision="main",
        )


def upload_staged_repo(
    stage_dir: Path,
    repo_id: str,
    version_branch: str,
    token: str,
    private: bool,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    commit_suffix = os.environ.get("GITHUB_SHA", "local")[:12]
    commit_message = f"Publish HOLYSHT kernel bundle ({commit_suffix})"
    delete_patterns = [
        "build/*",
        "build/**/*",
        "README.md",
        "LICENSE",
        "build.toml",
        "flake.nix",
        "flake.lock",
        ".gitattributes",
    ]

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        revision="main",
        commit_message=commit_message,
        delete_patterns=delete_patterns,
    )

    _ensure_version_branch(api, repo_id, version_branch)

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        revision=version_branch,
        commit_message=commit_message,
        delete_patterns=delete_patterns,
    )


def parse_args() -> argparse.Namespace:
    config = _load_build_config()
    project_name = _project_name_from_config(config)
    parser = argparse.ArgumentParser(
        description="Stage and publish the HOLYSHT Hugging Face kernel repository."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build",
        help="Kernel-builder output directory.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help="Directory used for the staged Hub repository.",
    )
    parser.add_argument(
        "--repo-id",
        default=_repo_id_from_config(config),
        help="Target Hugging Face repo id.",
    )
    parser.add_argument(
        "--version-branch",
        default=_version_branch_from_config(config),
        help="Version branch to update alongside main.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target Hugging Face repo as private if it does not exist.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Only stage the Hub repository locally; do not upload it.",
    )
    args = parser.parse_args()
    repo_name = args.repo_id.rsplit("/", 1)[-1]
    if repo_name != project_name:
        raise SystemExit(
            f"repo id {args.repo_id!r} is not compliant with this build: "
            f"the repository name must remain {project_name!r}"
        )
    args.project_name = project_name
    return args


def main() -> int:
    args = parse_args()
    stage_dir = stage_kernel_repo(args.build_dir, args.stage_dir, args.project_name)
    print(f"Staged Hugging Face kernel repo at {stage_dir}")

    if args.stage_only:
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to upload the staged kernel repo")

    upload_staged_repo(
        stage_dir=stage_dir,
        repo_id=args.repo_id,
        version_branch=args.version_branch,
        token=token,
        private=args.private,
    )
    print(
        f"Published {args.repo_id} to main and {args.version_branch}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
