# Provenance And Redistribution Notice

Prepared 2026-09-05. This is an author-requested cleaned implementation with
numerical parity unverified. No warranty of numerical or physical correctness
is made. Existing source folders were inspected read-only.

## Local Sources

- `icml_code/cal_seq_debug.py`: scoring flow, differentiation, clipping, torque
  slice, sampling, metrics and adaptive weights. Reimplemented using NumPy/SciPy;
  aliasing, stale-state and silent-failure bugs corrected.
- `icml_code/process_amass.py`: inspected for dataset segmentation/entrypoint
  behavior; no parallel runner or policy code copied.
- `icml_code/articulate/utils/rbdl/model.py`: inspected for the `pyrbdl` contract.
- `icml_code/articulate/model.py`: inspected to establish that zero-shape root
  FK equals translation. No mesh/body assets or full articulate module copied.
- `icml_code/articulate/math/angular.py`: inspected conversion conventions.
- `aaai_code/utils.py`: candidate missing utility. Its forward permutation,
  forward/inverse SMPL conversions and Body enum match the corresponding
  upstream PIP definitions retrieved on 2026-09-05. The local file additionally
  contains N/T-pose helpers. Only the necessary forward permutation and forward
  conversion are adapted in `tvs/conversions.py`. No AAAI scoring method,
  optimizer, network, visualization, inverse conversion or Body enum is included.
- Supplied `ICML.pdf`: arXiv:2512.07248v2, text extracted for method/citation
  inspection; no PDF is copied into the release.

## Upstream Utility

PIP, Xinyu Yi and collaborators:

- https://github.com/Xinyu-Yi/PIP
- https://github.com/Xinyu-Yi/PIP/blob/main/utils.py
- https://github.com/Xinyu-Yi/PIP/blob/main/articulate/utils/rbdl/model.py
- https://github.com/Xinyu-Yi/PIP/blob/main/readme.md
- https://github.com/Xinyu-Yi/PIP/blob/main/LICENSE

The retrieved upstream LICENSE is GNU GPL version 3. The adaptation in
`tvs/conversions.py` retains that provenance and is subject to GPL-3.0 terms;
it is not relicensed as MIT/BSD. The modification date and changes are recorded
above and in the module. Upstream sources were retrieved from mutable `main`,
not a verified historical revision used for ICML. Matching utility definitions
establish a shared implementation, not a verified chain of authorship or a
license grant for the separate ICML code.

## Authorized Release License

On 2026-09-05, the author explicitly authorized GPL-3.0-compatible publication
of this prepared TVS source directory. This prepared release is licensed as a
whole under GNU GPL version 3 only (`GPL-3.0-only`), including the PIP-derived
adaptation under GPL v3 terms. The pending author licensing approval gate is
resolved. The complete, unmodified license text is supplied in `LICENSE`, from
https://www.gnu.org/licenses/gpl-3.0.txt. Retain this provenance notice and the
source modification notices when redistributing; comply with the GPL's
applicable source and other distribution requirements. Citation alone is not
a substitute for license compliance.

On 2026-09-05, the cloned https://github.com/MengZR2001/TVS repository's
README, LICENSE and .gitignore were inspected at commit
`5139865445ed2e1921a50d1ca775dbe28dc0ecc9`. Its LICENSE is MIT and carries:

Copyright (c) 2026 MengZR2001

The complete original MIT copyright, permission and warranty notice is
preserved in `LICENSE-MIT` as a historical license notice. The local release
replaces `LICENSE` with GPL v3 under the author's explicit authorization.
This authorization applies to this prepared release, not retroactively to
earlier releases: previous MIT grants remain unaffected. Third-party
MIT-licensed material may be included in a GPL-covered work while preserving
its required copyright and permission notices; this does not invalidate its
MIT grants. `LICENSE-MIT` does not offer this release as a whole or the
PIP-derived GPL adaptation under MIT. Author authorization does not replace
third-party permissions or change external assets' terms. These are local
release-preparation changes only; no commit, push or remote publication is
performed by this update.

## External Assets

RBDL, PIP's physics assets (credited by PIP to PhysCap), SMPL, AMASS and its
constituent datasets, and UHC/PHC policies each have their own access and license
conditions. None of their binaries, models, data, checkpoints or derived body
assets are redistributed here. Obtain them from their official providers and
review current terms before use or redistribution. See README.md for links.
