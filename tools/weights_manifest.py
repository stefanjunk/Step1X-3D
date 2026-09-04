#!/usr/bin/env python3
"""Pin, verify and mirror the model weights this fork runs on.

The weights are not vendored into git: the geometry checkpoint alone is 8.2 GB
and its largest shard is 4.85 GB, which exceeds GitHub's 2 GB per-file limit for
Git LFS. What is vendored is this manifest — repository, revision, per-file size
and SHA-256 — so that a copy held anywhere (object storage, NAS, a private
mirror) can be proven identical to what the service was validated against.

    python tools/weights_manifest.py build   [--cache DIR] [--output FILE]
    python tools/weights_manifest.py verify  [--cache DIR] [--manifest FILE]
    python tools/weights_manifest.py mirror --destination DIR

Only weights whose licence permits redistribution are listed as mirrorable; see
the `redistributable` flag per repository and NOTICE for the reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / ".cache" / "huggingface" / "hub"
DEFAULT_MANIFEST = REPO_ROOT / "weights" / "MANIFEST.json"

# Repositories the geometry-only service actually needs, and whether their
# licence allows us to hold and hand on a copy.
REPOSITORIES = {
    "stepfun-ai/Step1X-3D": {
        "licence": "Apache-2.0",
        "redistributable": True,
        "note": "Trained weights for geometry, VAE, label and visual encoders.",
        "subfolders": ["Step1X-3D-Geometry-Label-1300m", "Step1X-3D-Geometry-1300m"],
    },
    "facebook/dinov2-with-registers-large": {
        "licence": "Apache-2.0",
        "redistributable": True,
        "note": "Configuration only; the trained weights come from the Step1X-3D checkpoint.",
        "subfolders": None,
    },
    "openai/clip-vit-large-patch14": {
        "licence": "none declared on the model card",
        "redistributable": False,
        "note": (
            "Configuration only, and deliberately not mirrored: redistributing files from a "
            "repository with no licence grant is the one act an undeclared licence makes risky."
        ),
        "subfolders": None,
    },
}


def _cache_dir(repo_id: str, cache: Path) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


def _snapshot(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wanted(relative: str, subfolders: list[str] | None) -> bool:
    return subfolders is None or any(relative.startswith(f"{name}/") for name in subfolders)


def build(cache: Path, output: Path) -> int:
    manifest = {"cache_layout": "huggingface_hub", "repositories": []}
    for repo_id, meta in REPOSITORIES.items():
        repo_dir = _cache_dir(repo_id, cache)
        snapshot = _snapshot(repo_dir)
        if snapshot is None:
            print(f"skip {repo_id}: not present in {cache}", file=sys.stderr)
            continue
        files = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(snapshot).as_posix()
            if not _wanted(relative, meta["subfolders"]):
                continue
            resolved = path.resolve()
            print(f"  hashing {repo_id}/{relative}", file=sys.stderr, flush=True)
            files.append(
                {
                    "path": relative,
                    "bytes": resolved.stat().st_size,
                    "sha256": _sha256(resolved),
                }
            )
        manifest["repositories"].append(
            {
                "repo_id": repo_id,
                "revision": snapshot.name,
                "licence": meta["licence"],
                "redistributable": meta["redistributable"],
                "note": meta["note"],
                "files": files,
                "total_bytes": sum(entry["bytes"] for entry in files),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    total = sum(repo["total_bytes"] for repo in manifest["repositories"])
    print(f"wrote {output} covering {len(manifest['repositories'])} repositories, {total / 2**30:.2f} GiB")
    return 0


def verify(cache: Path, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for repo in manifest["repositories"]:
        snapshot = _cache_dir(repo["repo_id"], cache) / "snapshots" / repo["revision"]
        for entry in repo["files"]:
            path = snapshot / entry["path"]
            if not path.exists():
                failures.append(f"missing {repo['repo_id']}/{entry['path']}")
                continue
            resolved = path.resolve()
            if resolved.stat().st_size != entry["bytes"]:
                failures.append(f"size differs {repo['repo_id']}/{entry['path']}")
                continue
            if _sha256(resolved) != entry["sha256"]:
                failures.append(f"sha256 differs {repo['repo_id']}/{entry['path']}")
    if failures:
        print("FAIL")
        for failure in failures:
            print("  -", failure)
        return 1
    checked = sum(len(repo["files"]) for repo in manifest["repositories"])
    print(f"PASS: {checked} files match the manifest")
    return 0


def mirror(cache: Path, manifest_path: Path, destination: Path) -> int:
    """Copy only the repositories whose licence allows redistribution."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied = 0
    for repo in manifest["repositories"]:
        if not repo["redistributable"]:
            print(f"skip {repo['repo_id']}: {repo['note']}")
            continue
        snapshot = _cache_dir(repo["repo_id"], cache) / "snapshots" / repo["revision"]
        for entry in repo["files"]:
            source = (snapshot / entry["path"]).resolve()
            target = destination / repo["repo_id"] / repo["revision"] / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == entry["bytes"]:
                continue
            shutil.copy2(source, target)
            copied += 1
    print(f"mirrored {copied} files into {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "verify", "mirror"])
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("STEP1X_HF_CACHE", DEFAULT_CACHE)))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()

    cache = args.cache if args.cache.name == "hub" else args.cache / "hub"
    if args.command == "build":
        return build(cache, args.output)
    if args.command == "verify":
        return verify(cache, args.manifest)
    if not args.destination:
        parser.error("mirror requires --destination")
    return mirror(cache, args.manifest, args.destination)


if __name__ == "__main__":
    raise SystemExit(main())
