# Fork notes

This fork of [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D) runs
**geometry only** on a local two-GPU host, and it removes the third-party code
whose licence does not permit commercial use in the European Union.

Forked at upstream commit `cb5ac94`. Upstream remains available as the
`upstream` remote.

## What changed and why

### 1. Two-GPU, containerised runtime

Per-stage device selection (`--geometry-device`, `GEOMETRY_DEVICE`), device
validation before weights load, explicit CUDA device contexts, VRAM cleanup
between stages, plus a CUDA 12.4 / Python 3.10 `Dockerfile`,
`docker-compose.yml` and pinned `requirements*.txt`.

### 2. The texture pipeline is gone

The product line prints in filament colours assigned downstream in CAD or in
the slicer, so generated textures were never used. Removing them also removes
eleven of the twelve files that carried a Tencent Hunyuan non-commercial
licence header, both vendored CUDA extensions (`custom_rasterizer`,
`mesh_processor`, which shipped with no declared licence at all), the
Stable Diffusion XL dependency and its CreativeML Open RAIL++-M flow-down
obligation, the `madebyollin/sdxl-vae-fp16-fix` and `ZhengPeng7/BiRefNet`
weights, and a reference to the withdrawn `runwayml/stable-diffusion-v1-5`
repository.

Deleted: `step1x3d_texture/`, `configs/train-texture-ig2mv/`, `data/ig2mv/`,
`train_ig2mv.py`, `train_ig2mv.sh`. Dropped from the runtime: `xatlas`,
`pybind11`, `pygltflib`, `cupy-cuda12x`, `pytorch3d`, `kaolin`, and the two
compiled extensions. `nvdiffrast` moved to `requirements-dataprep.txt` because
only the optional geometry data-preparation script imports it and it ships under
the Nvidia Source Code License rather than a plain permissive licence.

### 3. `volume_decoders.py` is an independent implementation

Upstream's file at that path carried the Tencent Hunyuan header, and the
geometry VAE imports and instantiates it on **every** geometry run
(`michelangelo_autoencoder.py` line 23, instantiated in both configuration
branches). It was replaced by an implementation written for this fork against
the interface the VAE needs, not by transcribing the original: the upstream
module was only exercised as a black box to confirm the sampling convention.

Verified in `tests/test_volume_decoders.py`:

- the dense decoder reproduces upstream's sampling grid to within float32
  rounding (`<= 1e-6`) on four analytic fields, so meshes stay dimensionally
  identical to earlier runs;
- the hierarchical decoder produces the **same** iso-surface as a dense
  evaluation — identical vertex counts and connected-component counts at
  192³ and 384³ for a sphere, a torus, a high-frequency wobbly surface and a
  body with a thin antenna plus a small detached bead;
- it needs far fewer network evaluations: 2.3M instead of 57M at 384³
  (96% fewer). Upstream evaluated the field densely at every ladder level, i.e.
  about 19.1M queries at 256³ against 17.0M for a plain dense pass, so this is
  also a substantial speed-up.

Two safeguards protect thin geometry, which matters for printed parts: the
coarsest level is dense at 96³, and refinement selects cells by sign change
**and** by a gradient-aware value band, so a feature that is still too small to
flip a sign at a coarse level is refined before it can be lost.

Upstream's vanilla decoder returned a 4-tuple that `extract_geometry` could not
concatenate; both decoders here return the grid, which makes
`volume_decoder_type="vanilla"` usable.

### 4. Mesh post-processing no longer uses pymeshlab

`remove_floater`, `remove_degenerate_face` and `reduce_face` were reimplemented
with trimesh and Open3D (both MIT) instead of pymeshlab (GPL-3.0), removing the
only copyleft component from the inference path. `plyfile` (GPL-3.0-or-later)
and `easydict` (LGPL-3.0) were unused and are dropped. Decimation keeps the
boundary weighting the previous filter used, the PLY temp-file round-trips are
gone, and the material the pipeline assigns now survives post-processing.
Covered by `tests/test_mesh_postprocess.py`. Both implementations were also run
side by side on the same fixtures inside the runtime image: `remove_floater`,
`remove_degenerate_face` and `reduce_face` return identical face, vertex and
component counts on a dense sphere and on spheres with attached parts of 4, 20
and 5120 faces, including decimation to exactly 800 faces. The only difference
is on a mesh deliberately seeded with a duplicated and a zero-area face, where
the new `remove_degenerate_face` removes both (320 faces, one component) while
the pymeshlab PLY round-trip removed only the zero-area one (321 faces, three
components) — the new behaviour is what the function name promises.

## Licence posture

`NOTICE` records the retained third-party components and the licences of the
weights downloaded at runtime.

### Checked and closed: the DiT reference

`facebookresearch/DiT` is CC BY-NC 4.0 and is named in the `Reference:` comment
of `models/conditional_encoders/clip/modeling_conditional_clip.py` and
`models/conditional_encoders/dinov2/modeling_conditional_dinov2.py`, which
looked like a second non-commercial exposure. It is not one:

- both files carry the `Copyright 2023 Meta AI and The HuggingFace Inc. team`
  Apache-2.0 header of the Transformers model files they adapt, and the DiT link
  sits in a reference list beside `transformers/models/dinov2/modeling_dinov2.py`
  and `3DTopia/OpenLRM`;
- the cited region of DiT (`models.py` around line 101) is the adaLN-Zero
  `DiTBlock`, and `models/conditional_encoders/` contains no `modulate()`,
  `adaLN`, `shift_msa` or `scale_msa` code at all.

The reference is therefore architectural, not copied expression. This was a
targeted check of the cited construct, not a line-by-line diff of all of DiT.

## Open items

1. **`openai/clip-vit-large-patch14` declares no licence on its model card.**
   Establish the terms for the weights rather than relying on the MIT licence of
   the CLIP code repository.
2. **Ask StepFun** whether the Hunyuan headers were stale. It no longer blocks
   this fork, but it determines whether upstream can be merged again.
3. **Slim the base image.** No source extension is compiled any more, so the
   CUDA `devel` base image could become a `runtime` one.
4. **Training data provenance.** The published weights derive from Objaverse and
   Objaverse-XL with heterogeneous per-asset licences; unresolved upstream.

## Tests

```bash
python tests/test_volume_decoders.py     # add STEP1X_DECODER_FULL=1 for 384**3
python tests/test_mesh_postprocess.py    # needs open3d, i.e. the container image
```
