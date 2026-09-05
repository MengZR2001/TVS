# TVS CPU Scorer

Code for our ICML 2026 paper **Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty**.

An author-requested, independent cleaned implementation of the supplied ICML
prototype for **Torque Variation Score (TVS)**, with numerical parity unverified.
This is usable core scoring code, **not a verified
reproduction of the paper, its scores, or its policy evaluations**. No imports
or implicit paths refer to `icml_code` or `aaai_code`. Those folders are untouched.

## License

This prepared release is licensed under GNU GPL version 3 only
(`GPL-3.0-only`); see [LICENSE](LICENSE) for the complete terms and
[NOTICE.md](NOTICE.md) for PIP attribution and source provenance. The author
explicitly authorized GPL-3.0-compatible publication on 2026-09-05; author
licensing approval is no longer pending. This does not establish numerical
parity or native integration correctness.

The cloned TVS repository's original MIT license was inspected on 2026-09-05
at commit `5139865445ed2e1921a50d1ca775dbe28dc0ecc9`. Its full text, including
`Copyright (c) 2026 MengZR2001`, is preserved in [LICENSE-MIT](LICENSE-MIT)
as a historical license notice. Previous MIT grants remain unaffected.
MIT-licensed third-party material can be included under GPL-compatible terms
with its required notices preserved; those MIT grants are not invalidated.
The PIP-derived adaptation remains GPL-covered; `LICENSE-MIT` does not
dual-license this release. External assets retain their separate terms.
This local preparation does not commit, push or modify the remote project.

## Environment

- Python 3.8+ with NumPy and SciPy: `python -m pip install -r requirements.txt`.
- CPU only; no PyTorch, CUDA, SMPL pickle, Chumpy, OpenCV, PyBullet, rendering,
  policy network, or global model loading is needed.
- Separately build a compatible RBDL binding with its URDF reader addon. Select
  the API explicitly; **do not assume `pip install rbdl` or `pip install pyrbdl`
  installs the correct project**. No native binary is redistributed here.
- Run commands below from this directory, or put this directory on `PYTHONPATH`.
  `python -B -S -m tvs --help` works even without numerical packages installed.

### RBDL Interfaces

The local `icml_code/articulate/utils/rbdl/model.py` imports `pyrbdl` and uses
`Model()`, `URDFReadFromFile(bytes, model, False, False)`, `set_gravity`, a
return-value `CompositeRigidBodyAlgorithm(model,q,update)`, and return-value
`NonlinearEffects(model,q,qdot)`. This is the `--binding pyrbdl` contract.
The exact source revision/build of that local binding is **not identified**.
It is not installed on the verification machine.

[Upstream PIP](https://github.com/Xinyu-Yi/PIP) instead imports `rbdl`, using
`loadModel(bytes)`, `model.gravity`, and output-array arguments for CRBA and
NonlinearEffects. `--binding rbdl` implements that concrete alternative, not a
silent fallback. PIP points to [official RBDL](https://github.com/rbdl/rbdl),
including its Python wrapper and URDF reader. Follow that project's build
instructions for your chosen revision and Python/NumPy ABI. Typical CMake
options are `RBDL_BUILD_PYTHON_WRAPPER=ON` and
`RBDL_BUILD_ADDON_URDFREADER=ON`; consult that revision's documentation.
Linux/WSL is usually easier; PIP explicitly notes Windows needs source/CMake
adjustments. No tested native Windows build recipe is claimed here.

Use the matching external `physics.urdf`. PIP's [official model
download](https://xinyu-yi.github.io/PIP/files/urdfmodels.zip) is linked in its
[readme](https://github.com/Xinyu-Yi/PIP/blob/main/readme.md), with PhysCap
provenance. The supplied ICML model may differ; identical assets must not be
assumed. This scorer requires **q_size = qdot_size = 75** and PIP's exact
coordinate ordering. A model with 75 dimensions but different joint ordering
is still incompatible. Do not add another floating base: the URDF already has
one. Quaternion-root models are rejected, not converted by guesswork.

The CLI records the URDF SHA-256, binding name/version (if exposed), and package
versions. Also retain your RBDL commit, compiler, build options and model source
with experiment records; they cannot be inferred from an unversioned extension.

## Usage

Input is one numeric NPZ file containing exactly the arrays used by the scorer:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pose` | `(T,24,3)` | SMPL joint-local axis-angle, radians, including global root |
| `pose` alternative | `(T,24,3,3)` | proper rotation matrices; choose `--representation matrix` |
| `tran` | `(T,3)` | root translation in meters, in the same world coordinates as gravity |

Choose the matching representation. At least two frames are required. Matrix
validation rejects non-rotations/reflections, NaNs and shape mismatches. NPZ is
loaded with `allow_pickle=False`; `.pt`/pickle loading is deliberately excluded.
Poses and translations must have real integer or floating dtypes. Complex
arrays (even with zero imaginary parts), strings, objects and booleans are
rejected before float64 conversion, not silently coerced or truncated.
No SMPL-H/SMPL-X joint truncation or hand mapping is guessed.

Example for an **already prepared 60 FPS, Y-up** clip. These values explicitly
select the corrected prototype's settings, not a calibrated paper experiment.
All paths are examples; replace them with your own assets. The output parent
must exist, and existing output files are never overwritten.

```powershell
python -B -m tvs --input "D:\motions\clip.npz" --output "D:\results\clip.json" --urdf "D:\assets\physics.urdf" --binding pyrbdl --fps 60 --epsilon 1e-6 --sampling all --weight-mode prototype-adaptive --formulation prototype-slice --gravity 0 -9.81 0 --representation axis-angle
```

Use `--binding rbdl` only with the upstream output-array API. Use
`--weight-mode paper-fixed` to select the paper's fixed `[0.4,0.3,0.3]` weights;
this changes **only the weights**, not the unresolved formulation. FPS,
epsilon, sampling, gravity and formulation acknowledgement are all required.
Errors return exit status 1 and include perturbation/frame context where relevant.
No failure is converted to a zero score. JSON includes the score components,
actual frame indices, weights, fixed prototype constants, paths and hashes.

For an already prepared pair of numeric arrays:

```python
import numpy as np
np.savez("clip.npz", pose=pose_axis_angle, tran=root_translation)
```

Programmatic use accepts rotation matrices and an explicit dynamics instance:

```python
from tvs.core import Settings, score_motion
from tvs.dynamics import RBDLDynamics

settings = Settings(fps=60, epsilon=1e-6, sampling="all",
                    weight_mode="prototype-adaptive", formulation="prototype-slice")
backend = RBDLDynamics("/assets/physics.urdf", "rbdl", [0, -9.81, 0])
result = score_motion(pose_matrices, translations, backend, settings)
```

Each sampled frame costs 144 dynamics evaluations. Full-clip angular kinematics
are recomputed on both perturbation sides; sampling only reduces Jacobian
evaluation, not the sequence used for derivatives. This favors clarity over
maximum throughput. No speed claim is made for this release.

## Dataset Preparation

The paper describes 100-frame clips from [AMASS](https://amass.is.tue.mpg.de/).
Acquire data from the official site, accept its license and the constituent
datasets' conditions, and retain sequence/split provenance. See AMASS's
[license](https://amass.is.tue.mpg.de/license.html) and the official
[SMPL model site](https://smpl.is.tue.mpg.de/) for model access/terms.
No data, fitted body parameters, body-model files or derived skeleton assets
are distributed in this directory.

AMASS source releases can use SMPL-H poses, different frame rates, coordinate
frames and body shapes. This release intentionally does **not** invent the
missing preprocessing protocol. Prepare 24-joint SMPL poses explicitly; record
the joint mapping, coordinate transform, resampling, FPS, segmentation boundaries
and dropped tails. Merely passing `--fps 60` does not resample a 120 FPS clip.
Do not treat the first 72 SMPL-H parameters as a validated SMPL conversion.

The original batch script consumes preprocessed `pose.pt` and `tran.pt`, splits
at 100 frames, skips tails shorter than 10 frames and silently truncates mismatched
lengths. Here each invocation scores exactly one supplied clip: no hidden
truncation, skipping, resampling, multiprocessing or dataset split is supplied.
For a comparable annotation workflow, prepare and record those choices yourself
before invoking the CLI. Scores on sampled frames or different clip lengths are
not directly interchangeable. No annotated AMASS download or policy checkpoint
is claimed to be included.

## Numerical Contract

The implemented formulation is named `prototype-slice` because several choices
are unresolved. It follows `cal_seq_debug.py`, not an invented physical fix:

1. Convert poses to the PIP 75-coordinate Euler configuration. The root conversion
   and the 69-component non-root permutation are taken from the shared PIP utility.
2. Compute translation velocity/acceleration by centered differences inside the
   clip and one-sided differences at its endpoints. Compute local angular
   velocity from `R[t+1] @ R[t].T`, repeat the penultimate velocity at the last
   frame, and differentiate it similarly.
3. Assign those angular vectors directly into `qdot[3:]` and `qddot[3:]` **in
   SMPL order**, even though `q` uses reordered Euler coordinates. Angular
   velocity is not generally an Euler-coordinate derivative. This mismatch is
   retained and not presented as validated inverse dynamics.
4. Clip acceleration to `[-1000,1000]`, calculate `M(q) @ qddot + h(q,qdot)`,
   clip all torque components to `[-10000,10000]`, then retain **`tau[6:30]`**.
   These are 24 scalar generalized-force entries, **not 24 three-axis joint
   torques**. No norm, aggregation, reordering or 69/72-axis replacement is invented.
5. Left-multiply each of 24 joint rotations at a target frame by positive and
   negative axis perturbations, recompute kinematics and dynamics, and use
   central differences. Each Jacobian has shape `(24,72)`.
6. Flatten sampled Jacobians to `(sample_count,1728)`. Spectral diversity is
   `sum(log(s + 1e-6))` over singular values `s > 1e-10`. Variance diversity is
   `sum(log(var + 1e-6))`, using sample variance (`ddof=1`) over time and input
   directions for each output row. Segment diversity averages spectral diversity
   across `torch.chunk(...,4)`-style contiguous chunks; this can yield fewer than
   four chunks (e.g. five sampled frames yield three). With fewer than four
   sampled frames, segment diversity equals spectral diversity. Dynamic range
   is `log(max(J)-min(J)+1e-6)` and does not enter the final score.
7. `all` uses all frames; `sparse` uses step `max(1,T//10)`; `uniform` uses
   step `max(1,T//min(20,T))` and takes at most 20 frames. This preserves the
   prototype's sometimes uneven coverage rather than substituting linspace.
8. Prototype motion activity is `min(1,10*(mean_rotation_step +
   10*mean_translation_step))`, without FPS normalization. Adaptive weights are
   `[0.4,0.3,0.3]` above 0.5 activity and `[0.3,0.4,0.3]` otherwise.

### Corrections And Differences

- **Aliasing fixed:** a `.copy()` preserves the unperturbed rotation for both
  signs. Originally, assignment on the positive side mutated a view reused for
  the negative side, corrupting the central difference.
- **Stale RBDL state fixed:** every mass-matrix evaluation explicitly requests
  kinematic updates. Originally manual updates were enabled but not performed.
- **Float64 CPU numerics:** NumPy/SciPy replace mixed Torch float32 and OpenCV
  Rodrigues conversions. Small-angle behavior, singular thresholds, clipping
  boundaries and resulting scores can differ materially. Epsilon sensitivity
  should be studied on real data; no stable epsilon range is asserted.
- **SMPL dependency removed algebraically:** the original zero-shape FK centers
  the rest root at zero, so its root position is exactly `tran`. Only joint 0's
  linear derivatives are consumed by `pose_to_rbdl_params`; all other joint
  positions are unused in scoring. Root translation derivatives therefore avoid
  loading a licensed pickle without substituting a different skeleton/method.
- **Failures are fatal:** invalid input, incompatible models and dynamics errors
  cannot silently create zero torques or plausible scores. T<2 is rejected.
- **Paper discrepancies remain visible:** Sec. 4.2.4 / Appendix B.2 specify fixed
  weights; the prototype switches them. Eq. 6 describes population variance but
  source Torch uses sample variance. The appendix's full-state/per-frame
  Jacobian/volume discussion is not identical to this pose-only perturbation and
  flattened-sequence SVD computation. No proof-equivalent replacement is claimed.
- Ground contacts are omitted as in the source/paper calculation, not simulated
  or estimated by another method. This release does not verify the paper's
  contact-neglect argument or applicability to impacts.

Because of these differences, **do not reuse the paper's TVS cutoffs (200/300/350),
MID values, rankings, correlations, or difficulty labels as calibrated results**.
Resolve coordinate derivatives, torque selection and mathematical aggregation
with the authors before claiming paper reproduction.

## Verification

```powershell
python -B -m unittest discover -s tests -v
python -B -S -m tvs --help
```

Tests cover conversion ordering, nontrivial root conversion, perturbation
symmetry/input immutability, derivative/FPS scaling, sampling, prototype metric
parity against optional Torch float64, weight selection, fake-backend end-to-end
scoring, invalid input, failure propagation, and both RBDL call contracts with
update flags. Torch is only an optional reference-test dependency.

Verification here does **not** include real RBDL loading/torques, an SMPL/URDF
coordinate equivalence test, real AMASS clips, policy training/evaluation, paper
score parity, or runtime benchmarking. Mock binding tests cannot establish
binary compatibility or physical correctness. The exact local `pyrbdl` build
and native integration are remaining verification blockers. See NOTICE.md for
the authorized release license, attribution, and external-asset conditions.

Local verification on 2026-09-05 used Python 3.8, isolated NumPy 1.24.4 and
SciPy 1.10.1 wheels, plus the installed Torch reference. The machine's base
Anaconda NumPy 1.20.1 failed to import its native DLLs; it was not modified.
Use a clean environment rather than interpreting that installation failure as
a scorer result. The CLI NPZ-to-JSON test also uses a fake dynamics backend and
checks provenance and refusal to overwrite existing output.

The cloned release was checked on 2026-09-05 using the existing
`sk-Perlin/.venv` Python 3.8.8 environment with NumPy 1.24.4 and SciPy 1.10.1:
14 tests passed and the optional Torch reference test was skipped because
Torch was unavailable in that environment. The dependency-free CLI help
check passed. No native RBDL or external assets were used.

## Paper And Policies

Paper inspected: supplied `ICML.pdf`, text identifies
[arXiv:2512.07248v2](https://arxiv.org/abs/2512.07248), 8 June 2026.
Its title page lists ICML 2026, PMLR 306. This release implements no imitation
policy. For the paper's policy experiments consult the official repositories:
[UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl) and
[PHC / PHC+](https://github.com/ZhengyiLuo/PHC), including their dataset,
checkpoint, simulator and licensing instructions. No alternate method from the
AAAI folder is included. Utility provenance is not method provenance.

```bibtex
@article{meng2026tvs,
  title={Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty},
  author={Meng, Zhaorui and Yin, Lu and Chen, Xinrui and Zuo, Chengxu and
          Chen, Anjun and Guo, Shihui and Qin, Yipeng},
  journal={arXiv preprint arXiv:2512.07248},
  year={2026},
  note={Version 2; supplied manuscript lists ICML 2026, PMLR 306}
}

@inproceedings{yi2022pip,
  title={Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors},
  author={Yi, Xinyu and Zhou, Yuxiao and Habermann, Marc and Shimada, Soshi and
          Golyanik, Vladislav and Theobalt, Christian and Xu, Feng},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{mahmood2019amass,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and
          Pons-Moll, Gerard and Black, Michael J.},
  booktitle={ICCV},
  year={2019}
}
```
