# SPDX-License-Identifier: GPL-3.0-only
"""Torque sensitivity and diversity metrics for motion clips.

Equations/settings trace to icml_code/cal_seq_debug.py; see README.md.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .conversions import smpl_to_rbdl


@dataclass(frozen=True)
class Settings:
    fps: float
    epsilon: float
    sampling: str
    weight_mode: str
    formulation: str

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < np.pi:
            raise ValueError("epsilon must be finite and in (0, pi) radians")
        if self.sampling not in ("all", "uniform", "sparse"):
            raise ValueError("unknown sampling strategy")
        if self.weight_mode not in ("prototype-adaptive", "paper-fixed"):
            raise ValueError("choose prototype-adaptive or paper-fixed weights")
        if self.formulation != "prototype-slice":
            raise ValueError("formulation must be prototype-slice")


def real_numeric_array(values, name):
    """Reject lossy/coercive input types before converting to float64."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values (integer or floating dtype)")
    return values.astype(np.float64, copy=False)


def validate_motion(poses, trans):
    poses = real_numeric_array(poses, "poses")
    trans = real_numeric_array(trans, "translations")
    if poses.ndim != 4 or poses.shape[1:] != (24, 3, 3):
        raise ValueError("poses must have shape (T,24,3,3)")
    if len(poses) < 2 or trans.shape != (len(poses), 3):
        raise ValueError("need at least two frames and matching (T,3) translations")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("motion contains NaN or infinity")
    if (not np.allclose(poses @ poses.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(poses), 1, atol=1e-5, rtol=0)):
        raise ValueError("poses must be proper rotation matrices, not reflections")
    return poses, trans


def frame_indices(total, sampling):
    if sampling == "all":
        return list(range(total))
    if sampling == "sparse":
        return list(range(0, total, max(1, total // 10)))
    if sampling == "uniform":
        return list(range(0, total, max(1, total // min(20, total))))[:20]
    raise ValueError("unknown sampling strategy")


def prototype_state(poses, trans, fps, frame):
    """Build q and assign angular rotation-vector differences to qdot/qddot."""
    velocity = np.gradient(trans, 1 / fps, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, 1 / fps, axis=0, edge_order=1)
    delta = poses[1:] @ poses[:-1].swapaxes(-1, -2)
    angular = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec()
    angular = angular.reshape(-1, 24, 3) * fps
    angular = np.concatenate((angular, angular[-1:]), axis=0)
    angular_acc = np.gradient(angular, 1 / fps, axis=0, edge_order=1)
    q = smpl_to_rbdl(poses[frame:frame + 1], trans[frame:frame + 1])[0]
    # Angular components stay in SMPL joint order, unlike q's Euler permutation.
    qdot = np.concatenate((velocity[frame], angular[frame].ravel()))
    qddot = np.concatenate((acceleration[frame], angular_acc[frame].ravel()))
    if not all(np.isfinite(x).all() for x in (q, qdot, qddot)):
        raise FloatingPointError("non-finite state; check motion scale and FPS")
    return q, qdot, np.clip(qddot, -1e3, 1e3)


def torque_jacobian(poses, trans, dynamics, settings, frame):
    jacobian = np.empty((24, 72), dtype=np.float64)
    for joint in range(24):
        for axis in range(3):
            # A copy is essential: the positive assignment must not change R0.
            original_rotation = poses[frame, joint].copy()
            torques = []
            for sign in (1, -1):
                perturbed = poses.copy()
                rotvec = np.zeros(3)
                rotvec[axis] = sign * settings.epsilon
                perturbed[frame, joint] = Rotation.from_rotvec(rotvec).as_matrix() @ original_rotation
                try:
                    state = prototype_state(perturbed, trans, settings.fps, frame)
                    torque = np.asarray(dynamics.torque(*state), dtype=np.float64)
                    if torque.shape != (75,) or not np.isfinite(torque).all():
                        raise ValueError("dynamics must return 75 finite torque components")
                    torques.append(np.clip(torque, -1e4, 1e4)[6:30])
                except Exception as exc:
                    raise RuntimeError(
                        f"Dynamics failed at frame={frame}, joint={joint}, axis={axis}, sign={sign}: {exc}"
                    ) from exc
            jacobian[:, joint * 3 + axis] = (torques[0] - torques[1]) / (2 * settings.epsilon)
    if not np.isfinite(jacobian).all():
        raise FloatingPointError("non-finite Jacobian; check epsilon")
    return jacobian


def diversity_metrics(jacobians):
    def spectral(values):
        singular = np.linalg.svd(values.reshape(len(values), -1), compute_uv=False)
        return float(np.log(singular[singular > 1e-10] + 1e-6).sum())

    spectral_value = spectral(jacobians)
    variance = np.var(jacobians, axis=(0, 2), ddof=1)
    segment = spectral_value
    if len(jacobians) >= 4:
        # Match torch.chunk(..., 4), which can yield fewer than four chunks.
        size = math.ceil(len(jacobians) / 4)
        segment = float(np.mean([
            spectral(jacobians[i:i + size]) for i in range(0, len(jacobians), size)
        ]))
    return {
        "spectral_diversity": spectral_value,
        "variance_diversity": float(np.log(variance + 1e-6).sum()),
        "segment_diversity": segment,
        "dynamic_range": float(np.log(np.ptp(jacobians) + 1e-6)),
    }


def score_motion(poses, trans, dynamics, settings):
    """Score one whole clip. dynamics.torque must implement the 75-DOF model."""
    poses, trans = validate_motion(poses, trans)
    frames = frame_indices(len(poses), settings.sampling)
    jacobians = np.stack([
        torque_jacobian(poses, trans, dynamics, settings, frame) for frame in frames
    ])
    metrics = diversity_metrics(jacobians)
    delta = poses[1:] @ poses[:-1].swapaxes(-1, -2)
    angles = Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude()
    movement = min(1.0, float((angles.mean() + 10 * np.linalg.norm(np.diff(trans, axis=0), axis=1).mean()) * 10))
    weights = [0.4, 0.3, 0.3]
    if settings.weight_mode == "prototype-adaptive" and movement <= 0.5:
        weights = [0.3, 0.4, 0.3]
    final = sum(w * metrics[k] for w, k in zip(weights, (
        "spectral_diversity", "variance_diversity", "segment_diversity"
    )))
    return {
        "status": "success",
        "final_score": final, "enhanced_metrics": metrics,
        "motion_dynamics": movement, "weights": weights,
        "settings": asdict(settings), "frames": frames,
        "total_frames": len(poses), "jacobian_shape": list(jacobians.shape),
        "numerics": {"dtype": "float64", "acceleration_clip": 1e3,
                     "torque_clip": 1e4, "torque_slice": [6, 30],
                     "variance_ddof": 1, "log_offset": 1e-6,
                     "singular_threshold": 1e-10, "segment_chunks": 4},
    }
