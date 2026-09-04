"""Acceptance tests for this fork's clean-room volume decoders.

Run standalone:

    python tests/test_volume_decoders.py

or under pytest. The tests drive the decoders with analytic implicit fields, so
the exact iso-surface is known and the hierarchical decoder can be compared
against a dense evaluation of the same field rather than against a snapshot.

Set ``STEP1X_DECODER_FULL=1`` to include the production 384**3 resolution, which
is skipped by default because a dense reference at that size costs 57 million CPU
evaluations per field.

Optionally set ``STEP1X_UPSTREAM_VOLUME_DECODERS`` to a copy of the upstream
module to additionally assert that the dense grid is bit-for-bit identical to
upstream's sampling; the check is skipped when the variable is unset.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
from skimage import measure

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _load_decoders():
    """Load the module directly so the test does not need the training stack.

    ``step1x3d_geometry/__init__.py`` imports PyTorch Lightning, which is not
    required by the decoders themselves; loading the file keeps this test
    runnable wherever torch, scikit-image and tqdm are available.
    """
    path = os.path.join(
        REPO_ROOT, "step1x3d_geometry", "models", "autoencoders", "volume_decoders.py"
    )
    spec = importlib.util.spec_from_file_location("step1x_volume_decoders", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VanillaVolumeDecoder, module.HierarchicalVolumeDecoder


VanillaVolumeDecoder, HierarchicalVolumeDecoder = _load_decoders()

BOUNDS = 1.05
LATENTS = torch.zeros(1, 8, 16)


def _sphere(points):
    return 0.6 - points.norm(dim=-1)


def _torus(points):
    offset = torch.stack(
        [points[..., 0:2].norm(dim=-1) - 0.55, points[..., 2]], dim=-1
    )
    return 0.2 - offset.norm(dim=-1)


def _wobbly(points):
    displacement = 0.06 * torch.sin(9 * points[..., 0]) * torch.cos(9 * points[..., 1])
    return 0.55 + displacement - points.norm(dim=-1)


def _spike(points):
    """A body, a thin antenna and a small detached bead.

    The antenna radius and the bead are deliberately close to one coarse voxel,
    which is the case a sparse refinement scheme can lose.
    """
    body = 0.45 - points.norm(dim=-1)
    axis_distance = torch.stack([points[..., 0], points[..., 1]], dim=-1).norm(dim=-1)
    antenna = torch.minimum(0.022 - axis_distance, 0.95 - points[..., 2].abs())
    bead = 0.035 - (points - torch.tensor([0.85, 0.0, 0.0])).norm(dim=-1)
    return torch.maximum(torch.maximum(body, antenna), bead)


FIELDS = {"sphere": _sphere, "torus": _torus, "wobbly": _wobbly, "spike": _spike}


def _run(decoder, field, resolution):
    calls = [0]

    def query_fn(queries, latents):
        calls[0] += queries.shape[1]
        return field(queries.float()).unsqueeze(-1)

    grid = decoder(
        LATENTS,
        query_fn,
        bounds=BOUNDS,
        octree_resolution=resolution,
        mc_level=0.0,
        verbose=False,
    )
    if isinstance(grid, tuple):  # tolerate a tuple-returning implementation
        grid = grid[0]
    return grid[0], calls[0]


def _surface(grid, resolution):
    values = np.nan_to_num(grid.numpy(), nan=-1e3)
    verts, faces, _, _ = measure.marching_cubes(values, 0.0, method="lewiner")
    return verts / resolution * (2 * BOUNDS) - BOUNDS, faces


def _component_count(faces, vertex_count):
    parent = list(range(vertex_count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for triangle in faces:
        first = find(int(triangle[0]))
        for other in triangle[1:]:
            root = find(int(other))
            if root != first:
                parent[root] = first
                first = find(first)
    return len({find(int(vertex)) for triangle in faces for vertex in triangle})


def test_grid_shape_and_extent():
    """The grid spans bounds inclusively with resolution + 1 samples per axis."""
    seen = []

    def query_fn(queries, latents):
        seen.append(queries.reshape(-1, 3).clone())
        return _sphere(queries.float()).unsqueeze(-1)

    grid = VanillaVolumeDecoder()(
        LATENTS, query_fn, bounds=BOUNDS, octree_resolution=16, verbose=False
    )
    assert tuple(grid.shape) == (1, 17, 17, 17)
    points = torch.cat(seen, dim=0)
    assert points.shape[0] == 17 ** 3
    ticks = torch.unique(points[:, 0])
    assert ticks.numel() == 17
    assert torch.allclose(ticks.min(), torch.tensor(-BOUNDS))
    assert torch.allclose(ticks.max(), torch.tensor(BOUNDS))


def test_bounds_forms_agree():
    """A scalar bound and its explicit six-value form produce the same grid."""
    scalar = VanillaVolumeDecoder()(
        LATENTS,
        lambda q, l: _sphere(q.float()).unsqueeze(-1),
        bounds=BOUNDS,
        octree_resolution=24,
        verbose=False,
    )
    explicit = VanillaVolumeDecoder()(
        LATENTS,
        lambda q, l: _sphere(q.float()).unsqueeze(-1),
        bounds=[-BOUNDS, -BOUNDS, -BOUNDS, BOUNDS, BOUNDS, BOUNDS],
        octree_resolution=24,
        verbose=False,
    )
    assert torch.equal(scalar, explicit)


def test_hierarchical_matches_dense_surface():
    """Same iso-surface as a dense evaluation, at a fraction of the queries."""
    # A dense 384**3 reference costs 57M CPU evaluations per field, so the
    # production resolution is opt-in via STEP1X_DECODER_FULL=1.
    resolutions = (96, 192) if not os.environ.get("STEP1X_DECODER_FULL") else (96, 192, 384)
    for name, field in FIELDS.items():
        for resolution in resolutions:
            dense, dense_queries = _run(VanillaVolumeDecoder(), field, resolution)
            sparse, sparse_queries = _run(HierarchicalVolumeDecoder(), field, resolution)

            dense_verts, dense_faces = _surface(dense, resolution)
            sparse_verts, sparse_faces = _surface(sparse, resolution)

            assert len(sparse_verts) == len(dense_verts), (
                f"{name}@{resolution}: {len(sparse_verts)} vertices "
                f"versus {len(dense_verts)} from a dense evaluation"
            )
            assert _component_count(sparse_faces, len(sparse_verts)) == _component_count(
                dense_faces, len(dense_verts)
            ), f"{name}@{resolution}: a connected component was lost"

            error = field(torch.from_numpy(sparse_verts).float()).abs().max().item()
            voxel = 2 * BOUNDS / resolution
            assert error < 0.25 * voxel, f"{name}@{resolution}: surface error {error:.2e}"
            if resolution > HierarchicalVolumeDecoder().min_resolution:
                # Below the coarsest ladder level the decoder is dense by design.
                assert sparse_queries < 0.35 * dense_queries, (
                    f"{name}@{resolution}: {sparse_queries} queries against "
                    f"{dense_queries} for a dense evaluation"
                )


def test_dense_grid_matches_upstream_when_available():
    """Optional: the sampling grid is unchanged from upstream."""
    reference_path = os.environ.get("STEP1X_UPSTREAM_VOLUME_DECODERS")
    if not reference_path or not os.path.isfile(reference_path):
        print("skipped: STEP1X_UPSTREAM_VOLUME_DECODERS is not set")
        return
    spec = importlib.util.spec_from_file_location("upstream_volume_decoders", reference_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    for name, field in FIELDS.items():
        ours, _ = _run(VanillaVolumeDecoder(), field, 64)
        theirs, _ = _run(upstream.VanillaVolumeDecoder(), field, 64)
        delta = (ours - theirs).abs().max().item()
        assert delta <= 1e-6, f"{name}: dense grid differs from upstream by {delta:.2e}"


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
