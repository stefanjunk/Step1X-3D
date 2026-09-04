# Weights: independence from Hugging Face and from the author

The service needs 8.2 GB of trained weights. They are **not** committed here. What is committed is
`MANIFEST.json`: repository, revision, and per-file size and SHA-256, so any copy you hold can be
proven identical to the weights this fork was validated against.

```bash
python tools/weights_manifest.py build                 # pin what is in the local cache
python tools/weights_manifest.py verify                # prove a cache matches the manifest
python tools/weights_manifest.py mirror --destination /mnt/backup/step1x-weights
```

## What may be copied, and what may not

| Repository | Licence | Mirror? |
| --- | --- | --- |
| `stepfun-ai/Step1X-3D` | Apache-2.0 | **Yes.** Redistribution is permitted with attribution and the NOTICE; this is the 8.2 GB that matters. |
| `facebook/dinov2-with-registers-large` | Apache-2.0 | Yes, but it is 20 KB of configuration only. |
| `openai/clip-vit-large-patch14` | none declared | **No.** Only its 16 KB `config.json` is used and no weight file is ever fetched. Republishing files from a repository that grants no licence is exactly the act an undeclared licence makes risky, so `mirror` skips it by design. |

Losing the CLIP `config.json` would not be fatal — it is architecture metadata that can be
reconstructed — but keep a private copy rather than a public mirror.

## Why this is not in Git LFS

Two hard blockers, one cost:

- **File size.** GitHub caps a single Git LFS file at 2 GB. The geometry transformer shard is
  4.85 GB and the visual encoder is 2.79 GB. They would have to be split (`split -b 1900M`) and
  rejoined on checkout — a real mechanism, but one that hides the artifact behind a build step and
  breaks `verify` against the upstream hashes.
- **Quota.** A GitHub account includes 1 GB of LFS storage and 1 GB/month of bandwidth. 8.2 GB needs
  paid data packs, and every clone or CI run that fetches the weights spends bandwidth from that
  allowance.
- **Fit.** Git is a poor fit for immutable multi-gigabyte binaries that never diff.

## What to do instead

Any of these gives independence from Hugging Face and from the author, and all of them are verifiable
against `MANIFEST.json`:

1. **Own object storage or NAS** — `mirror --destination` to S3/R2/Backblaze or a local disk. Cheapest
   and no file-size limit. Record the location in the product provenance.
2. **A private Hugging Face mirror repository** — free and purpose-built for weights, but it keeps a
   dependency on the platform, which was the thing to avoid.
3. **Git LFS anyway**, if a single origin is worth more than the drawbacks: split the two large
   shards, add `weights/mirror/** filter=lfs diff=lfs merge=lfs -text` to `.gitattributes`, and buy
   the data packs. The rejoin step must run before `verify`.

Option 1 is the recommendation. The manifest is what makes the copy trustworthy, whichever you pick.
