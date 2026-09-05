# Torque Variation Score (TVS)

Code for **[Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty](https://arxiv.org/abs/2512.07248)**, ICML 2026.

TVS scores human motion clips using torque sensitivity to pose perturbations.
This CPU-only scorer takes SMPL joint rotations and root translations, uses an
external RBDL dynamics model, and writes a JSON report with score components,
settings, and input provenance. Each invocation scores one prepared clip.

## Installation

Use Python 3.8 or later. From the repository directory, install the numerical
dependencies:

```sh
python -m pip install -r requirements.txt
```

Scoring uses NumPy and SciPy; it does not require CUDA, PyTorch, rendering, or an
SMPL body-model file. RBDL and the URDF model must be obtained separately.
Run the commands below from the repository directory, or add it to `PYTHONPATH`.

### RBDL Binding

Build a compatible Python binding with the URDF reader addon and select it
explicitly with `--binding`. The adapters require these APIs:

| Binding | Required API |
| --- | --- |
| `rbdl` | `loadModel(path_bytes, floating_base=False)`, writable `model.gravity`, `CompositeRigidBodyAlgorithm(model, q, matrix, update_kinematics=True)`, and `NonlinearEffects(model, q, qdot, bias)` using output arrays |
| `pyrbdl` | `Model()`, `URDFReadFromFile(path_bytes, model, False, False)`, `model.set_gravity(gravity)`, and return-value `CompositeRigidBodyAlgorithm(model, q, True)` and `NonlinearEffects(model, q, qdot)` |

For the `rbdl` API, follow the [official RBDL build instructions](https://github.com/rbdl/rbdl)
for your revision and Python/NumPy environment. Relevant CMake options are
`RBDL_BUILD_PYTHON_WRAPPER=ON` and `RBDL_BUILD_ADDON_URDFREADER=ON`.
The [PIP repository](https://github.com/Xinyu-Yi/PIP) also provides dependency
guidance. Package names alone do not guarantee API compatibility:
`pip install rbdl` or `pip install pyrbdl` is not a substitute for checking these
requirements. There is no automatic binding fallback or bundled native binary.

### URDF Model

Use a PIP-compatible `physics.urdf`. PIP provides an
[official model download](https://xinyu-yi.github.io/PIP/files/urdfmodels.zip)
and asset instructions in its [README](https://github.com/Xinyu-Yi/PIP/blob/main/readme.md).

The model must have **`q_size = qdot_size = 75`** and the exact PIP
Euler-coordinate ordering used by [the pose conversion](tvs/conversions.py).
Matching dimensions alone is not sufficient. The URDF already includes the
floating base; do not add another. Quaternion-root models are incompatible.
Retain the model source and RBDL revision/build settings with your experiments.

## Input Data

Provide a numeric NPZ archive containing these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pose` | `(T, 24, 3)` | SMPL joint-local axis-angle rotations in radians, including global root orientation; use `--representation axis-angle` |
| `pose` alternative | `(T, 24, 3, 3)` | Proper rotation matrices in the same joint order; use `--representation matrix` |
| `tran` | `(T, 3)` | Root translations in meters, in the same world coordinate system as gravity |

Both arrays must have the same frame count, with **at least two frames**, finite
values, and real integer or floating-point dtypes. Rotation matrices must be
orthonormal with determinant +1. Boolean, complex, string, and object arrays are
not accepted. NPZ loading uses `allow_pickle=False`; `.pt` and pickle inputs
are not supported.

After preparing your arrays, write the input archive with NumPy:

```python
import numpy as np

# pose_axis_angle: (T, 24, 3), root_translation: (T, 3)
np.savez("clip.npz", pose=pose_axis_angle, tran=root_translation)
```

### Data Preparation

Obtain motion data from [AMASS](https://amass.is.tue.mpg.de/) under its
[license](https://amass.is.tue.mpg.de/license.html) and the constituent datasets'
terms. Consult the [official SMPL site](https://smpl.is.tue.mpg.de/) for body-model
access and licensing. Motion data, body-model files, and URDF assets are not
included in this repository.

1. Convert the source poses to the 24-joint SMPL convention. SMPL-H and SMPL-X inputs require an explicit joint mapping, not simply the first 72 pose parameters.
2. Express root orientation, translation, and gravity in a consistent coordinate system and use meters for translation.
3. Resample as needed before scoring, then set `--fps` to the actual input frame rate. The CLI does not resample motion.
4. Split sequences into clips before invocation. The paper uses 100-frame clips; the CLI scores the entire supplied clip without truncating or skipping tails.
5. Record sequence sources, splits, joint mappings, coordinate transforms, resampling, and clip boundaries. Keep clip lengths and scoring settings consistent when comparing scores.

For imitation-policy workflows, see the official
[UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl) and
[PHC / PHC+](https://github.com/ZhengyiLuo/PHC) repositories for their data,
checkpoint, and simulator instructions.

## Usage

With `clip.npz` prepared at **60 FPS in Y-up coordinates**, and a compatible
`physics.urdf` in the current directory:

```sh
python -B -m tvs --input clip.npz --output clip.json --urdf physics.urdf --binding rbdl --fps 60 --epsilon 1e-6 --sampling all --weight-mode prototype-adaptive --formulation prototype-slice --gravity 0 -9.81 0 --representation axis-angle
```

Adjust paths, binding, frame rate, gravity, and representation to match your
assets. All flags shown are required; CLI help is available with
`python -B -m tvs --help`.

The output parent directory must already exist; existing output files
are never overwritten. Input or scoring failures print an error and return
exit status 1 rather than a zero score.

### Method Notes

The scorer uses torque components `tau[6:30]`. Each sampled frame requires 144
dynamics evaluations; temporal derivatives use the full clip.

The state uses PIP Euler coordinates for `q`. Angular velocity components are
rotation-vector differences between consecutive poses, in SMPL joint order;
angular accelerations are their temporal gradients. These components are
assigned directly to `qdot` and `qddot`, without applying `q`'s Euler permutation
or converting them to Euler-coordinate derivatives.

## Output

The CLI prints the final score and writes a JSON report:

| Field | Description |
| --- | --- |
| `final_score` | Weighted sum of spectral, variance, and segment diversity |
| `enhanced_metrics` | `spectral_diversity`, `variance_diversity`, `segment_diversity`, and `dynamic_range`; dynamic range is not included in the final score |
| `motion_dynamics`, `weights` | Motion activity value and the three weights applied |
| `settings` | FPS, perturbation size, sampling, weight mode, and formulation |
| `frames`, `total_frames` | Zero-based sampled frame indices and input clip length |
| `jacobian_shape` | `[sample_count, 24, 72]` |
| `numerics` | Numeric precision, clipping limits, torque slice, and aggregation constants |
| `provenance` | Input and URDF paths and SHA-256 hashes, binding/version, gravity, representation, and Python/package versions |
| `status` | `success` when scoring completes |

## Tests

```sh
python -B -m unittest discover -s tests -v
python -B -S -m tvs --help
```

Tests cover conversions, perturbations, derivatives, sampling, metrics, input
validation, CLI output, and binding call contracts using synthetic data and
mock dynamics backends.
PyTorch is an optional reference-test dependency. CLI help requires only the
Python standard library.

## Citation

If you use TVS, cite the paper. For PIP utilities or AMASS data, also cite the
corresponding work:

```bibtex
@article{meng2026tvs,
  title={Distinguishing Imitation Error from Intrinsic Motion Learning Difficulty},
  author={Meng, Zhaorui and Yin, Lu and Chen, Xinrui and Zuo, Chengxu and
          Chen, Anjun and Guo, Shihui and Qin, Yipeng},
  journal={arXiv preprint arXiv:2512.07248},
  year={2026}
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

## License

This release is licensed under **GNU GPL version 3 only (`GPL-3.0-only`)**;
see [LICENSE](LICENSE). [NOTICE.md](NOTICE.md) contains attribution and source
provenance, including PIP-derived utilities.

[LICENSE-MIT](LICENSE-MIT) preserves the original repository's historical MIT
notice, including `Copyright (c) 2026 MengZR2001`. It does not dual-license this
release. External datasets, models, and other assets retain their own terms.
