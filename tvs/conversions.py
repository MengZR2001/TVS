# SPDX-License-Identifier: GPL-3.0-only
"""SMPL-to-PIP coordinates, adapted 2026-09-05 from PIP utils.py.

PIP authors: Xinyu Yi et al.; upstream GPL-3.0. See NOTICE.md.
Only the forward conversion and its permutation are retained.
"""

import numpy as np
from scipy.spatial.transform import Rotation


SMPL_TO_RBDL = [
    0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29,
    3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
    15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47,
    51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
    48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68,
    33, 34, 35, 42, 43, 44,
]


def smpl_to_rbdl(poses, trans):
    """Convert validated (T,24,3,3)/(T,3) inputs to (T,75) q."""
    local = Rotation.from_matrix(poses[:, 1:].reshape(-1, 3, 3))
    local = local.as_euler("XYZ").reshape(-1, 69)
    root = Rotation.from_matrix(poses[:, 0]).as_euler("xyz")
    root = Rotation.from_euler("xyz", root[:, [2, 1, 0]]).as_euler("zyx")
    q = np.concatenate((trans, root, local[:, SMPL_TO_RBDL]), axis=1)
    angles = q[:, 3:] % (2 * np.pi)
    angles[angles >= np.pi] -= 2 * np.pi
    q[:, 3:] = angles
    return np.ascontiguousarray(q, dtype=np.float64)
