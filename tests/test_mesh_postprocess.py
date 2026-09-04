"""Tests for the mesh post-processing helpers.

These replaced a pymeshlab (GPL-3.0) implementation with trimesh and Open3D
(both MIT), so the behaviour that the geometry pipeline relies on is asserted
here: small disconnected parts are dropped, degenerate and duplicated faces are
removed, decimation respects the target face count, and the material the
pipeline assigns survives.

Run standalone:  python tests/test_mesh_postprocess.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from step1x3d_geometry.models.autoencoders.surface_extractors import (  # noqa: E402
    MeshExtractResult,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import (  # noqa: E402
    as_trimesh,
    reduce_face,
    remove_degenerate_face,
    remove_floater,
)


def _body_with_part(part: trimesh.Trimesh) -> trimesh.Trimesh:
    body = trimesh.creation.icosphere(subdivisions=4, radius=1.0)  # 5120 faces
    part = part.copy()
    part.apply_translation([3.0, 0.0, 0.0])
    return trimesh.util.concatenate([body, part])


def test_as_trimesh_accepts_mesh_extract_result():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    result = MeshExtractResult(
        verts=torch.tensor(np.asarray(sphere.vertices), dtype=torch.float32),
        faces=torch.tensor(np.asarray(sphere.faces), dtype=torch.long),
    )
    converted = as_trimesh(result)
    assert isinstance(converted, trimesh.Trimesh)
    assert len(converted.faces) == len(sphere.faces)


def test_remove_floater_drops_small_part():
    """Below 0.1% of all faces (5 faces here) a component is a floater."""
    combined = _body_with_part(trimesh.creation.box(extents=[0.05, 0.05, 0.05]).convex_hull)
    tetra = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
        faces=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
    combined = _body_with_part(tetra)
    assert len(combined.split(only_watertight=False)) == 2
    cleaned = remove_floater(combined)
    assert len(cleaned.split(only_watertight=False)) == 1
    assert len(cleaned.faces) == len(trimesh.creation.icosphere(subdivisions=4).faces)


def test_remove_floater_keeps_part_above_the_ratio():
    """A 20-face part is above the threshold and must survive, as upstream did."""
    combined = _body_with_part(trimesh.creation.icosphere(subdivisions=0, radius=0.05))
    cleaned = remove_floater(combined)
    assert len(cleaned.split(only_watertight=False)) == 2
    assert len(cleaned.faces) == len(combined.faces)


def test_remove_floater_never_empties_a_single_part_mesh():
    sphere = trimesh.creation.icosphere(subdivisions=1)
    assert len(remove_floater(sphere).faces) == len(sphere.faces)


def test_remove_degenerate_face():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    faces = np.vstack(
        [
            np.asarray(sphere.faces),
            np.asarray(sphere.faces[:1]),          # duplicate
            [[0, 0, 1]],                           # zero-area
        ]
    )
    dirty = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    cleaned = remove_degenerate_face(dirty)
    assert len(cleaned.faces) == len(sphere.faces)
    assert len(cleaned.nondegenerate_faces()) == len(cleaned.faces)


def test_reduce_face_respects_target_and_keeps_material():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=(255, 255, 255), metallicFactor=0.05, roughnessFactor=1.0
        )
    )
    target = 800
    reduced = reduce_face(sphere, max_facenum=target)
    assert len(reduced.faces) <= target
    assert len(reduced.faces) > target * 0.5
    assert reduced.visual.material.metallicFactor == 0.05
    assert reduced.is_watertight


def test_reduce_face_is_a_noop_below_target():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    same = reduce_face(sphere, max_facenum=len(sphere.faces) + 10)
    assert len(same.faces) == len(sphere.faces)
    assert len(reduce_face(sphere, max_facenum=0).faces) == len(sphere.faces)


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
