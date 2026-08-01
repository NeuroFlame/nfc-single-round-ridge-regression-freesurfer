"""Build and publish a versioned NeuroFLAME computation image."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SEMVER = re.compile(r"\d+\.\d+\.\d+")
REVISION = re.compile(r"[0-9a-f]{7,64}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
PUSH_DIGEST = re.compile(rf"digest:\s+({DIGEST.pattern})")
REQUIRED_CONFIG_KEYS = ("title", "repository", "floatingTag", "tagPrefix", "source")


def _run(command: list[str], repository: Path, *, capture_output: bool = False) -> str:
    """Run a command in the computation repository."""
    completed = subprocess.run(
        command,
        cwd=repository,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return completed.stdout


def _read_version(repository: Path, filename: str) -> str:
    """Read a strict semantic version marker."""
    value = (repository / filename).read_text(encoding="utf-8").strip()
    if not SEMVER.fullmatch(value):
        raise ValueError(f"Invalid semantic version in {filename}: {value!r}")
    return value


def _read_nvflare_version(repository: Path) -> str:
    """Read the exact NVFlare pin from requirements.txt."""
    matches = re.findall(
        r"(?im)^nvflare==([^\s#]+)\s*$",
        (repository / "requirements.txt").read_text(encoding="utf-8"),
    )
    if len(matches) != 1 or not SEMVER.fullmatch(matches[0]):
        raise ValueError("requirements.txt must contain one exact nvflare==X.Y.Z pin")
    return matches[0]


def _read_image_config(repository: Path) -> dict[str, str]:
    """Load and validate repository-specific image publishing configuration."""
    raw = json.loads(
        (repository / ".neuroflame-image.json").read_text(encoding="utf-8")
    )
    missing = [key for key in REQUIRED_CONFIG_KEYS if not isinstance(raw.get(key), str)]
    if missing:
        raise ValueError(f"Missing image configuration keys: {', '.join(missing)}")
    if any(not raw[key].strip() for key in REQUIRED_CONFIG_KEYS if key != "tagPrefix"):
        raise ValueError("Image configuration values must not be empty")
    return {key: raw[key].strip() for key in REQUIRED_CONFIG_KEYS}


def _get_revision(repository: Path, requested_revision: str | None) -> str:
    """Resolve and validate the image source revision."""
    revision = (
        requested_revision
        or _run(["git", "rev-parse", "HEAD"], repository, capture_output=True).strip()
    )
    if not REVISION.fullmatch(revision):
        raise ValueError(f"Invalid Git revision: {revision!r}")
    return revision


def _ensure_tracked_files_clean(repository: Path) -> None:
    """Reject publication when the labeled revision does not describe the build."""
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=repository, check=False
    )
    if dirty.returncode != 0:
        raise ValueError("Refusing to publish with tracked uncommitted changes")


def build_labels(
    repository: Path, revision: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the canonical labels and return them with image configuration."""
    config = _read_image_config(repository)
    labels = {
        "org.opencontainers.image.title": config["title"],
        "org.opencontainers.image.version": _read_version(
            repository, ".neuroflame-computation-version"
        ),
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": config["source"],
        "org.neuroflame.computation-api.version": _read_version(
            repository, ".neuroflame-computation-api-version"
        ),
        "org.neuroflame.boilerplate.version": _read_version(
            repository, ".neuroflame-boilerplate-version"
        ),
        "org.neuroflame.nvflare.version": _read_nvflare_version(repository),
    }
    return labels, config


def _image_tags(config: dict[str, str], version: str, revision: str) -> list[str]:
    """Return floating, release, and revision image references."""
    repository = config["repository"]
    prefix = config["tagPrefix"]
    return [
        f"{repository}:{config['floatingTag']}",
        f"{repository}:{prefix}{version}",
        f"{repository}:{prefix}{revision}",
    ]


def _push_tags(tags: list[str], repository: Path) -> str:
    """Push image tags and return the registry-reported digest."""
    digest: str | None = None
    for tag in tags:
        output = _run(["docker", "push", tag], repository, capture_output=True)
        print(output, end="")
        match = PUSH_DIGEST.search(output)
        if match:
            digest = match.group(1)
    if not digest:
        raise RuntimeError("Docker push completed without reporting an image digest")
    return digest


def publish(
    repository: Path,
    *,
    revision: str | None,
    platform: str,
    push: bool,
    local: bool,
) -> None:
    """Build, inspect, tag, and optionally push a computation image."""
    resolved_revision = _get_revision(repository, revision)
    if push:
        _ensure_tracked_files_clean(repository)
    labels, config = build_labels(repository, resolved_revision)
    tags = _image_tags(
        config, labels["org.opencontainers.image.version"], resolved_revision
    )
    build_command = [
        "docker",
        "build",
        "--platform",
        platform,
        "-f",
        "Dockerfile-prod",
    ]
    for key, value in labels.items():
        build_command.extend(["--label", f"{key}={value}"])
    for tag in tags:
        build_command.extend(["-t", tag])
    build_command.append(".")
    _run(build_command, repository)

    inspected = _run(
        ["docker", "image", "inspect", tags[0], "--format", "{{json .Config.Labels}}"],
        repository,
        capture_output=True,
    )
    actual_labels = json.loads(inspected)
    if any(actual_labels.get(key) != value for key, value in labels.items()):
        raise RuntimeError("Built image does not contain the required metadata labels")

    if local:
        image_id = _run(
            ["docker", "image", "inspect", tags[0], "--format", "{{.Id}}"],
            repository,
            capture_output=True,
        ).strip()
        if not DIGEST.fullmatch(image_id):
            raise RuntimeError(f"Docker returned an invalid local image ID: {image_id}")
        print(f"NeuroFLAME local image: {tags[0]}")
        print(f"Local image ID: {image_id}")
        return

    if not push:
        print(f"Built and validated {tags[0]} without pushing")
        return

    digest = _push_tags(tags, repository)
    print(f"Published immutable image: {config['repository']}@{digest}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the publisher command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", nargs="?", help="explicit Git revision")
    parser.add_argument(
        "--platform", default="linux/amd64", help="target container platform"
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--no-push", action="store_true", help="build and validate without pushing"
    )
    destination.add_argument(
        "--local",
        action="store_true",
        help="build labeled local tags for NeuroFLAME development",
    )
    return parser


def main() -> None:
    """Run the computation image publisher."""
    args = _build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    publish(
        repository,
        revision=args.revision,
        platform=args.platform,
        push=not args.no_push and not args.local,
        local=args.local,
    )


if __name__ == "__main__":
    main()
