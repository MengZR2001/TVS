import subprocess
import sys
import types
import unittest
import tempfile
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from tvs.conversions import SMPL_TO_RBDL, smpl_to_rbdl
from tvs.core import (Settings, diversity_metrics, frame_indices, prototype_state,
                      score_motion, torque_jacobian, validate_motion)
from tvs.dynamics import RBDLDynamics


def settings(**changes):
    values = dict(fps=60, epsilon=1e-6, sampling="all",
                  weight_mode="prototype-adaptive", formulation="prototype-slice")
    values.update(changes)
    return Settings(**values)


class FakeDynamics:
    def torque(self, q, qdot, qddot):
        return q + qdot + qddot


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.tile(np.eye(3), (2, 24, 1, 1))
        self.tran = np.zeros((2, 3))

    def test_conversion_permutation_and_nontrivial_root(self):
        self.assertEqual(sorted(SMPL_TO_RBDL), list(range(69)))
        euler = np.arange(69).reshape(23, 3) * 0.001
        self.pose[:, 1:] = Rotation.from_euler("XYZ", euler).as_matrix()
        self.pose[:, 0] = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
        q = smpl_to_rbdl(self.pose, self.tran)
        np.testing.assert_allclose(q[:, 6:], np.tile(euler.ravel()[SMPL_TO_RBDL], (2, 1)), atol=1e-14)
        root = Rotation.from_euler("xyz", [0.3, 0.2, 0.1]).as_euler("zyx")
        np.testing.assert_allclose(q[:, 3:6], np.tile(root, (2, 1)), atol=1e-14)

    def test_perturbations_are_symmetric_and_inputs_unchanged(self):
        recorded = []

        class Record:
            def torque(self, q, qdot, qddot):
                recorded.append(q.copy())
                return q

        before = self.pose.copy()
        jac = torque_jacobian(self.pose, self.tran, Record(), settings(), 0)
        np.testing.assert_array_equal(before, self.pose)
        self.assertEqual(len(recorded), 144)
        # joint 1 X is q[6]; catches the original minus-side alias/half derivative.
        self.assertAlmostEqual(recorded[6][6], 1e-6, places=12)
        self.assertAlmostEqual(recorded[7][6], -1e-6, places=12)
        self.assertAlmostEqual(jac[0, 3], 1, places=8)

    def test_state_endpoints_and_fps(self):
        trans = np.array([[0., 0, 0], [1., 0, 0], [4., 0, 0]])
        pose = np.tile(np.eye(3), (3, 24, 1, 1))
        _, vel, acc = prototype_state(pose, trans, 2, 0)
        self.assertEqual(vel[0], 2)
        self.assertEqual(acc[0], 4)
        _, vel2, acc2 = prototype_state(pose, trans, 4, 0)
        self.assertEqual(vel2[0], 2 * vel[0])
        self.assertEqual(acc2[0], 4 * acc[0])

    def test_sampling_matches_source(self):
        self.assertEqual(frame_indices(21, "uniform"), list(range(20)))
        self.assertEqual(frame_indices(101, "sparse"), list(range(0, 101, 10)))
        self.assertEqual(frame_indices(2, "all"), [0, 1])

    def test_metrics_match_torch_reference_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch reference unavailable")
        rng = np.random.RandomState(3)
        for count in (2, 4, 5, 9):
            array = rng.randn(count, 24, 72)
            tensor = torch.from_numpy(array)
            def spectral(x):
                s = torch.svd(x.reshape(len(x), -1))[1]
                return torch.log(s[s > 1e-10] + 1e-6).sum().item()
            expected_segment = spectral(tensor)
            if count >= 4:
                expected_segment = np.mean([spectral(x) for x in torch.chunk(tensor, 4)])
            actual = diversity_metrics(array)
            self.assertAlmostEqual(actual["spectral_diversity"], spectral(tensor), places=9)
            self.assertAlmostEqual(actual["segment_diversity"], expected_segment, places=9)
            self.assertAlmostEqual(actual["variance_diversity"], torch.log(torch.var(tensor, dim=[0, 2]) + 1e-6).sum().item(), places=9)

    def test_complete_fake_score_and_weight_choice(self):
        result = score_motion(self.pose, self.tran, FakeDynamics(), settings())
        self.assertTrue(np.isfinite(result["final_score"]))
        self.assertEqual(result["jacobian_shape"], [2, 24, 72])
        self.assertEqual(result["weights"], [0.3, 0.4, 0.3])
        fixed = score_motion(self.pose, self.tran, FakeDynamics(), settings(weight_mode="paper-fixed"))
        self.assertEqual(fixed["weights"], [0.4, 0.3, 0.3])

    def test_invalid_inputs(self):
        for changes in ({"fps": 0}, {"fps": float("nan")}, {"epsilon": 0},
                        {"sampling": "guess"}, {"formulation": "paper"}):
            with self.assertRaises(ValueError):
                settings(**changes)
        for pose, tran in ((self.pose[:1], self.tran[:1]), (self.pose, self.tran[:1]),
                           (-self.pose, self.tran), (self.pose * np.nan, self.tran)):
            with self.assertRaises(ValueError):
                validate_motion(pose, tran)

    def test_failure_has_frame_context(self):
        class Broken:
            def torque(self, *args):
                raise ValueError("backend broke")
        with self.assertRaisesRegex(RuntimeError, "frame=0, joint=0, axis=0, sign=1.*backend broke"):
            score_motion(self.pose, self.tran, Broken(), settings())

    def test_nonreal_and_nonnumeric_motion_rejected_before_dynamics(self):
        for field in ("pose", "tran"):
            original = self.pose if field == "pose" else self.tran
            imaginary_nan = original.astype(complex)
            imaginary_nan.imag.flat[0] = np.nan
            invalid_arrays = (
                original.astype(complex), original + 1j, imaginary_nan,
                original.astype(str), original.astype(object), original.astype(bool),
            )
            for invalid in invalid_arrays:
                with self.subTest(field=field, dtype=invalid.dtype, value=invalid.flat[0]):
                    pose = invalid if field == "pose" else self.pose
                    tran = invalid if field == "tran" else self.tran
                    backend = Mock()
                    with self.assertRaisesRegex(ValueError, "real numeric"):
                        score_motion(pose, tran, backend, settings())
                    backend.torque.assert_not_called()

    def test_real_integer_and_float_motion_still_accepted(self):
        for dtype in (np.int64, np.uint8, np.float32, np.float64):
            pose, tran = validate_motion(self.pose.astype(dtype), self.tran.astype(dtype))
            self.assertEqual(pose.dtype, np.float64)
            self.assertEqual(tran.dtype, np.float64)

    def test_cli_rejects_nonreal_and_nonnumeric_before_model_load(self):
        from tvs.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "motion.npz", Path(directory) / "score.json"
            for representation in ("axis-angle", "matrix"):
                pose = np.zeros((2, 24, 3)) if representation == "axis-angle" else self.pose
                for field in ("pose", "tran"):
                    original = pose if field == "pose" else self.tran
                    imaginary_nan = original.astype(complex)
                    imaginary_nan.imag.flat[0] = np.nan
                    for invalid in (original.astype(complex), original + 1j, imaginary_nan,
                                    original.astype(str), original.astype(bool)):
                        with self.subTest(representation=representation, field=field, dtype=invalid.dtype,
                                          value=invalid.flat[0]):
                            np.savez(source, pose=invalid if field == "pose" else pose,
                                     tran=invalid if field == "tran" else self.tran)
                            argv = ["--input", str(source), "--output", str(output),
                                    "--urdf", __file__, "--binding", "pyrbdl", "--fps", "60",
                                    "--epsilon", "1e-6", "--sampling", "all",
                                    "--weight-mode", "prototype-adaptive", "--formulation", "prototype-slice",
                                    "--gravity", "0", "-9.81", "0", "--representation", representation]
                            errors = io.StringIO()
                            with patch("tvs.dynamics.RBDLDynamics") as backend, redirect_stderr(errors):
                                self.assertEqual(main(argv), 1)
                                backend.assert_not_called()
                            self.assertIn("real numeric", errors.getvalue())
                            self.assertFalse(output.exists())

    def test_nonfinite_torque_is_not_hidden_by_clipping(self):
        class Broken:
            def torque(self, *args):
                return np.full(75, np.inf)
        with self.assertRaisesRegex(RuntimeError, "finite torque"):
            score_motion(self.pose, self.tran, Broken(), settings())

    def test_both_binding_contracts_update_every_mass_matrix(self):
        for binding in ("pyrbdl", "rbdl"):
            calls = []
            model = types.SimpleNamespace(q_size=75, qdot_size=75, set_gravity=lambda g: None)
            def matrix(m, q, *args, **kwargs):
                if binding == "pyrbdl":
                    calls.append(args[0])
                    return np.eye(75) * (1 + q[0])
                calls.append(kwargs["update_kinematics"])
                args[0][:] = np.eye(75) * (1 + q[0])
            def bias(m, q, qdot, *args):
                if binding == "pyrbdl":
                    return np.zeros(75)
                args[0][:] = 0
            api = types.SimpleNamespace(Model=lambda: model, URDFReadFromFile=lambda *a: True,
                                        loadModel=lambda *a, **k: model,
                                        CompositeRigidBodyAlgorithm=matrix, NonlinearEffects=bias)
            with patch("tvs.dynamics.importlib.import_module", return_value=api):
                backend = RBDLDynamics(__file__, binding, [0, -9.81, 0])
                for value in (0., 1.):
                    q = np.full(75, value)
                    np.testing.assert_allclose(backend.torque(q, q, np.ones(75)), 1 + value)
            self.assertEqual(calls, [True, True])

    def test_help_without_site_packages_or_models(self):
        result = subprocess.run([sys.executable, "-B", "-S", "-m", "tvs", "--help"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--formulation", result.stdout)

    def test_cli_npz_json_and_no_overwrite(self):
        from tvs.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "motion.npz", root / "score.json"
            np.savez(source, pose=np.zeros((2, 24, 3)), tran=self.tran)
            fake = FakeDynamics()
            fake.api = types.SimpleNamespace(__version__="test-only")
            argv = ["--input", str(source), "--output", str(output),
                    "--urdf", __file__, "--binding", "pyrbdl", "--fps", "60",
                    "--epsilon", "1e-6", "--sampling", "all",
                    "--weight-mode", "prototype-adaptive", "--formulation", "prototype-slice",
                    "--gravity", "0", "-9.81", "0", "--representation", "axis-angle"]
            with patch("tvs.dynamics.RBDLDynamics", return_value=fake), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                original = output.read_bytes()
                result = json.loads(original)
                self.assertEqual(result["provenance"]["binding_version"], "test-only")
                self.assertEqual(len(result["provenance"]["input_sha256"]), 64)
                self.assertEqual(main(argv), 1)
                self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
