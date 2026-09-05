# Attribution And License Notice

## TVS Sources

The scoring flow, differentiation, clipping, torque slice, sampling, metrics,
and adaptive weights in `tvs/core.py` are based on
`icml_code/cal_seq_debug.py`, implemented here using NumPy/SciPy.
The `pyrbdl` adapter follows the API used in
`icml_code/articulate/utils/rbdl/model.py`.

## PIP Utilities

The forward SMPL-to-RBDL conversion and coordinate permutation in
`tvs/conversions.py` are adapted from PIP's `utils.py`, also present in the
local source `aaai_code/utils.py`.

PIP authors: Xinyu Yi and collaborators.

- Project: https://github.com/Xinyu-Yi/PIP
- Conversion source: https://github.com/Xinyu-Yi/PIP/blob/main/utils.py
- RBDL API reference: https://github.com/Xinyu-Yi/PIP/blob/main/articulate/utils/rbdl/model.py
- License: https://github.com/Xinyu-Yi/PIP/blob/main/LICENSE

Modified 2026-09-05: retained the forward conversion and its permutation,
using NumPy/SciPy in `tvs/conversions.py`. This adaptation is covered by
GNU GPL version 3 terms. Retain this attribution and the module's modification
notice when redistributing.

## Release License

This release as a whole is licensed under GNU GPL version 3 only
(`GPL-3.0-only`), including the PIP-derived adaptation. The complete license
text is supplied in `LICENSE`. Redistribution is subject to the GPL's
applicable source and other distribution requirements. The software is
provided without warranty; see `LICENSE`.

The original https://github.com/MengZR2001/TVS repository's MIT notice,
including `Copyright (c) 2026 MengZR2001`, is preserved in full in
`LICENSE-MIT`. Earlier MIT grants remain unaffected. `LICENSE-MIT` does not
dual-license this release as a whole or the PIP-derived adaptation under MIT.

## External Assets

RBDL, PIP's physics assets (credited by PIP to PhysCap), SMPL, AMASS and its
constituent datasets, and UHC/PHC policies each have their own access and license
conditions. None of their binaries, models, data, checkpoints or derived body
assets are redistributed here. Obtain them from their official providers and
review current terms before use or redistribution. See README.md for links.
