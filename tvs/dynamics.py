# SPDX-License-Identifier: GPL-3.0-only
"""Explicit adapters for the ICML and upstream PIP RBDL APIs."""

import importlib
from pathlib import Path

import numpy as np


class RBDLDynamics:
    def __init__(self, urdf, binding, gravity):
        path = Path(urdf).resolve(strict=True)
        if binding not in ("pyrbdl", "rbdl"):
            raise ValueError("binding must be pyrbdl or rbdl; no auto-detection")
        gravity = np.asarray(gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity must contain three finite values")
        try:
            self.api = importlib.import_module(binding)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import {binding}. Compile the matching RBDL binding with "
                "the URDF reader; see README.md. No substitute backend is used."
            ) from exc
        self.binding = binding
        if binding == "pyrbdl":
            self.model = self.api.Model()
            loaded = self.api.URDFReadFromFile(
                str(path).encode(), self.model, False, False
            )
            if loaded is not None and not loaded:
                raise RuntimeError(f"URDFReadFromFile failed: {path}")
            self.model.set_gravity(gravity)
        else:
            self.model = self.api.loadModel(str(path).encode(), floating_base=False)
            self.model.gravity = gravity
        if (self.model.q_size, self.model.qdot_size) != (75, 75):
            raise ValueError(
                "Expected PIP Euler-coordinate physics model with q_size=qdot_size=75; "
                f"got {self.model.q_size}/{self.model.qdot_size}. "
                "Quaternion floating-base models are not interchangeable."
            )

    def torque(self, q, qdot, qddot):
        """Unclipped M(q) qddot + h(q,qdot); refresh M for every state."""
        if self.binding == "pyrbdl":
            matrix = self.api.CompositeRigidBodyAlgorithm(self.model, q, True)
            bias = self.api.NonlinearEffects(self.model, q, qdot)
        else:
            matrix = np.zeros((75, 75), dtype=np.float64)
            bias = np.zeros(75, dtype=np.float64)
            self.api.CompositeRigidBodyAlgorithm(
                self.model, q, matrix, update_kinematics=True
            )
            self.api.NonlinearEffects(self.model, q, qdot, bias)
        matrix, bias = np.asarray(matrix), np.asarray(bias)
        if matrix.shape != (75, 75) or bias.shape != (75,):
            raise ValueError("RBDL returned an incompatible matrix/bias shape")
        if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
            raise FloatingPointError("RBDL returned non-finite dynamics")
        return matrix @ qddot + bias
