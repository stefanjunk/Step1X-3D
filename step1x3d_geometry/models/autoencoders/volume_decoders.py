"""Volume decoders that turn a shape latent into a sampled implicit field.

Fork note
---------
The file that upstream ships at this path carries a verbatim
``TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT`` header, and it is imported
and instantiated by :mod:`michelangelo_autoencoder`, so every geometry run
executed third-party code under a licence that does not permit commercial use
in the European Union. This module is an independent re-implementation written
for this fork against the two behaviours the geometry VAE actually needs:

* evaluate the decoder's implicit field on a regular grid that spans
  ``bounds`` with ``octree_resolution + 1`` samples per axis, so that
  :class:`~step1x3d_geometry.models.autoencoders.surface_extractors.MCSurfaceExtractor`
  can run marching cubes on it, and
* do that cheaply at high resolution by refining coarse-to-fine and evaluating
  the network only where the iso-surface can actually pass.

The sampling convention matches the upstream behaviour exactly (samples are
``torch.linspace(-bounds, +bounds, resolution + 1)`` per axis, ``ij`` indexing),
which was established by probing the upstream implementation as a black box, so
meshes stay dimensionally identical to earlier runs.

Both decoders return a single ``[1, G, G, G]`` float32 tensor. Upstream's
vanilla decoder returned a 4-tuple, which ``extract_geometry`` could not
concatenate; returning the grid from both classes makes the non-default
``volume_decoder_type="vanilla"`` path usable.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

__all__ = ["VanillaVolumeDecoder", "HierarchicalVolumeDecoder"]

BoundsLike = Union[float, int, Sequence[float]]
QueryFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_PROGRESS_DESC = "Volume Decoding"
_DEFAULT_CHUNK = 65536


def _normalise_bounds(bounds: BoundsLike) -> Tuple[List[float], List[float]]:
    """Return ``(bbox_min, bbox_max)`` for a scalar, a 2-tuple or a 6-tuple."""
    if isinstance(bounds, (int, float)):
        extent = float(bounds)
        return [-extent] * 3, [extent] * 3
    values = [float(value) for value in bounds]
    if len(values) == 2:
        return [values[0]] * 3, [values[1]] * 3
    if len(values) == 6:
        return values[0:3], values[3:6]
    raise ValueError(f"bounds must hold 1, 2 or 6 values, received {len(values)}")


def _axis_ticks(
    bbox_min: Sequence[float], bbox_max: Sequence[float], grid_size: int, device: torch.device
) -> List[torch.Tensor]:
    return [
        torch.linspace(float(low), float(high), grid_size, device=device, dtype=torch.float32)
        for low, high in zip(bbox_min, bbox_max)
    ]


def _dense_points(ticks: Sequence[torch.Tensor]) -> torch.Tensor:
    """All grid points in ``ij`` order, shaped ``[G**3, 3]``."""
    grid = torch.meshgrid(*ticks, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def _evaluate_field(
    query_fn: QueryFn,
    latents: torch.Tensor,
    points: torch.Tensor,
    chunk_size: int,
    verbose: bool,
) -> torch.Tensor:
    """Evaluate ``query_fn`` at ``points`` in chunks and return ``[N]`` logits."""
    if points.numel() == 0:
        return points.new_zeros((0,), dtype=torch.float32)
    chunk_size = max(1, int(chunk_size))
    starts = range(0, points.shape[0], chunk_size)
    iterator = tqdm(starts, desc=_PROGRESS_DESC, disable=not verbose, leave=False)
    logits: List[torch.Tensor] = []
    for start in iterator:
        chunk = points[start : start + chunk_size].to(dtype=latents.dtype).unsqueeze(0)
        values = query_fn(chunk, latents)
        if values.dim() == 3:
            if values.shape[-1] != 1:
                raise ValueError(
                    "The geometry decoder must return one channel per query point, "
                    f"received {values.shape[-1]}"
                )
            values = values.squeeze(-1)
        logits.append(values.reshape(-1).float())
    return torch.cat(logits, dim=0)


def _refinement_mask(
    grid: torch.Tensor, level: float, dilation: int, safety: float
) -> torch.Tensor:
    """Mark every sample that may sit near the iso-surface.

    Two criteria are combined:

    * the 3x3x3 neighbourhood straddles ``level``, i.e. the surface provably
      passes through it, and
    * the value is within ``safety`` local steps of ``level``, where the local
      step is the largest absolute difference to a neighbour. This second term
      is what protects thin features: a spike or a small detached component
      that is still too small to flip a sign at the coarse resolution is
      approached in value first, so it is refined before it can be lost.

    Marching cubes only reads values around the level set, so everything outside
    the mask can stay interpolated. Edges are replicated rather than zero-padded
    so a surface touching the volume boundary is still refined.
    """
    values = grid[None, None]
    padded = F.pad(values, (1, 1, 1, 1, 1, 1), mode="replicate")
    neighbourhood_max = F.max_pool3d(padded, kernel_size=3, stride=1)
    neighbourhood_min = -F.max_pool3d(-padded, kernel_size=3, stride=1)
    mask = (neighbourhood_max >= level) & (neighbourhood_min <= level)
    if safety > 0:
        local_step = torch.maximum(neighbourhood_max - values, values - neighbourhood_min)
        mask = mask | ((values - level).abs() <= safety * local_step)
    for _ in range(max(0, int(dilation))):
        mask = F.max_pool3d(mask.float(), kernel_size=3, stride=1, padding=1) > 0
    return mask[0, 0]


def _resolve_chunk_size(num_chunks: Optional[int], chunk_size: Optional[int]) -> int:
    """Accept upstream's ``num_chunks`` (points per chunk) and an explicit alias."""
    for candidate in (chunk_size, num_chunks):
        if candidate:
            return int(candidate)
    return _DEFAULT_CHUNK


def _resolve_progress(verbose: bool, enable_pbar: Optional[bool]) -> bool:
    """The geometry pipeline switches progress output with ``enable_pbar``."""
    if enable_pbar is None:
        return bool(verbose)
    return bool(enable_pbar)


class VanillaVolumeDecoder:
    """Evaluate the field at every point of the target grid."""

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor,
        query_fn: QueryFn,
        bounds: BoundsLike = 1.05,
        octree_resolution: int = 256,
        num_chunks: Optional[int] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
        enable_pbar: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        verbose = _resolve_progress(verbose, enable_pbar)
        resolution = int(octree_resolution)
        if resolution < 1:
            raise ValueError(f"octree_resolution must be positive, received {resolution}")
        bbox_min, bbox_max = _normalise_bounds(bounds)
        grid_size = resolution + 1
        ticks = _axis_ticks(bbox_min, bbox_max, grid_size, latents.device)
        points = _dense_points(ticks)
        logits = _evaluate_field(
            query_fn, latents, points, _resolve_chunk_size(num_chunks, chunk_size), verbose
        )
        return logits.reshape(1, grid_size, grid_size, grid_size)


class HierarchicalVolumeDecoder:
    """Refine the field coarse-to-fine and only evaluate near the iso-surface.

    The coarsest level is evaluated densely. Each further level doubles the
    resolution, carries the previous level over by trilinear interpolation and
    then replaces the interpolated values with exact network evaluations wherever
    the iso-surface may pass. Values far from the surface stay interpolated:
    marching cubes never reads them, and the saving is what makes a 384**3 grid
    affordable.
    """

    def __init__(
        self, min_resolution: int = 96, dilation: int = 1, safety: float = 1.5
    ):
        self.min_resolution = int(min_resolution)
        self.dilation = int(dilation)
        self.safety = float(safety)

    def _resolution_ladder(self, target: int) -> List[int]:
        if target < 1:
            raise ValueError(f"octree_resolution must be positive, received {target}")
        ladder = [target]
        while ladder[-1] // 2 >= self.min_resolution:
            ladder.append(ladder[-1] // 2)
        return list(reversed(ladder))

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor,
        query_fn: QueryFn,
        bounds: BoundsLike = 1.05,
        octree_resolution: int = 256,
        mc_level: float = 0.0,
        num_chunks: Optional[int] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
        enable_pbar: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        verbose = _resolve_progress(verbose, enable_pbar)
        bbox_min, bbox_max = _normalise_bounds(bounds)
        resolved_chunk = _resolve_chunk_size(num_chunks, chunk_size)
        ladder = self._resolution_ladder(int(octree_resolution))

        coarse_size = ladder[0] + 1
        ticks = _axis_ticks(bbox_min, bbox_max, coarse_size, latents.device)
        grid = _evaluate_field(
            query_fn, latents, _dense_points(ticks), resolved_chunk, verbose
        ).reshape(coarse_size, coarse_size, coarse_size)

        for resolution in ladder[1:]:
            grid_size = resolution + 1
            grid = F.interpolate(
                grid[None, None],
                size=(grid_size, grid_size, grid_size),
                mode="trilinear",
                align_corners=True,
            )[0, 0]
            mask = _refinement_mask(
                grid, float(mc_level), self.dilation, self.safety
            )
            indices = mask.nonzero(as_tuple=False)
            if indices.numel() == 0:
                continue
            ticks = _axis_ticks(bbox_min, bbox_max, grid_size, latents.device)
            points = torch.stack(
                [ticks[axis][indices[:, axis]] for axis in range(3)], dim=-1
            )
            values = _evaluate_field(query_fn, latents, points, resolved_chunk, verbose)
            grid[indices[:, 0], indices[:, 1], indices[:, 2]] = values.to(grid.dtype)

        return grid.unsqueeze(0).float()
