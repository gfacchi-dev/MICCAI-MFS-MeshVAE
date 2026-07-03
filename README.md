# MICCAI-MFS-MeshVAE

**Marfan facial screening from 3D meshes — a conditional mesh-VAE encoder and linear screener.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/deps-uv-purple.svg)](https://docs.astral.sh/uv/)

Official code and model release for the paper

> **Learning the Marfan Face: A Conditional Mesh VAE for Screening and Explainable 3D Facial Analysis**
> *Workshop on Shape in Medical Imaging (ShapeMI), MICCAI 2026.*

Given a registered 3D facial mesh, this package predicts the probability that the
subject presents Marfan-syndrome (MFS)–like craniofacial morphology.

## Abstract

Marfan syndrome (MFS) is a rare connective-tissue disorder whose diagnosis can be
challenging when systemic manifestations are mild. Although craniofacial
morphology contributes to the phenotype, facial characteristics are often subtle
and spatially distributed. We propose a conditional mesh variational autoencoder
(cVAE) framework that unifies three-dimensional facial screening and anatomical
explanation. An age- and sex-conditioned representation model learns latent
embeddings of registered facial meshes for disease screening through a linear
classifier and decoder-based counterfactual explanations for result
interpretability. A second instance of the same architecture, additionally
conditioned on disease status, generates synthetic facial meshes. These synthetic
data are evaluated as an alternative training source through train-on-synthetic,
test-on-real experiments, providing a quantitative assessment of generative
fidelity. Experiments on 1548 registered facial meshes (1423 controls and 125
individuals with MFS) demonstrate accurate screening, anatomically meaningful
explanations, and high agreement between real- and synthetic-trained models in
terms of latent organization, facial phenotype, and decision-boundary
counterfactuals. These results show that a single conditional generative
framework can simultaneously support explainable disease screening and
quantitative validation of synthetic-data fidelity for rare-disease applications.

## This release

This repository releases the **deployable screening model** from the paper: an
age- and sex-conditioned spiral-convolutional encoder that maps a registered mesh
to a 64-dimensional latent code, followed by a logistic-regression screener. It
was trained on 1548 quality-controlled meshes (1423 controls, 125 MFS) with
class- and sex-specific PCA statistical-shape-model augmentation. The full
framework additionally includes a disease-conditioned generator and
decoder-based anatomical explanation; see the paper for those components.

## What is (and is not) included

| Component | Shared? | Why |
|---|---|---|
| **Encoder** (mesh → 64-d latent), trained on the full cohort | ✅ | needed to embed a face |
| **Linear screener** (logistic regression) | ✅ | produces the MFS probability |
| Mesh topology: spiral indices, **down**-sampling operators, template, normalization | ✅ | needed to run the encoder |
| **Decoder** (latent → mesh) weights | ❌ | **withheld by design** |
| Decoder **up**-sampling operators | ❌ | withheld — decode cannot run without them |
| Training meshes / patient data | ❌ | not distributable |

> **Privacy by design.** This bundle screens faces but cannot *generate* or
> *reconstruct* them: the generative decoder is not part of the release, so a
> face cannot be recovered from its latent code. The 64-d embedding is a
> compressed, non-invertible representation. The two bundled example faces are
> **class averages** (means of the real MFS / control meshes), not individuals.

## Install

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`uv.lock`:

```bash
uv sync
```

No GPU required (CPU inference is fast). There is no `torch_scatter` dependency —
the mesh pooling is reimplemented with `index_add_`.

## Quick start

```bash
uv run python demo/demo.py                                   # scores the two example faces
uv run python demo/demo.py --mesh face.obj --age 22 --sex M  # scores your own mesh
```

```python
import numpy as np, trimesh
from marfan_face import MarfanScreener

verts = np.asarray(trimesh.load("face.obj", process=False).vertices)  # (7160, 3)
out = MarfanScreener().screen(verts, age_years=22, sex="M")
print(out["marfan_probability"])
```

Expected demo output:

```
example_marfan.obj    age  22/M  |z|=1.62  P(Marfan)=0.999  -> MARFAN-like
example_control.obj   age  28/F  |z|=1.30  P(Marfan)=0.000  -> control-like
```

## Input requirements

The encoder accepts **only** meshes in the shared template topology
(**7160 vertices / 14050 faces**). Register your raw scans to
`weights/template.ply` before screening, using
[AutoFaceMonker](https://github.com/gfacchi-dev/AutoFaceMonker) (landmark-guided
non-rigid registration) followed by Generalized Procrustes Analysis (GPA), as in
the paper's preprocessing.

- `age_years`: subject age in years (standardized internally).
- `sex`: `"F"` or `"M"` (encoded `F=0`, `M=1`).
- Age is standardized in months using the model's training statistics
  (see `weights/metadata.json`).

> **Single deployment model.** This release ships one encoder and one screener
> trained on all 1548 subjects (no held-out fold). `weights/template.ply` defines
> the Procrustes frame; register new meshes to it before screening.

## Operating point

`screen()` returns a calibrated MFS probability. Thresholding at `0.50` is the
default operating point; for high-sensitivity screening use the ≈`0.28` threshold
(≥ 95 % recall) reported in the paper.

## Repository structure

```
marfan_face/            encoder + inference code (no decoder)
  encoder.py              spiral-convolutional encoder (en_layers only)
  inference.py            MarfanScreener: mesh -> latent -> probability
weights/
  encoder.pt              encoder weights
  screener.pkl            logistic-regression screener
  norm.pt                 vertex normalization (mean/std)
  age_stats.pt            age standardization (months)
  template.ply            shared topology reference (deployment frame)
  metadata.json           config (dims, conditioning, encodings)
  topology/               spiral indices + down-sampling operators
demo/
  demo.py                 runnable example
  example_marfan.obj      mean of the real MFS meshes (class average)
  example_control.obj     mean of the real control meshes (class average)
pyproject.toml, uv.lock   uv project definition and pinned dependencies
```

## Citation

If you use this code or model, please cite:

[TBD]

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
