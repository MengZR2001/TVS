# SPDX-License-Identifier: GPL-3.0-only
"""CLI: argparse and --help require only Python's standard library."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU Torque Variation Score (TVS) for one prepared motion clip")
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing pose and tran")
    parser.add_argument("--output", required=True, type=Path, help="new JSON file; parent must exist")
    parser.add_argument("--urdf", required=True, type=Path, help="external PIP-compatible physics.urdf")
    parser.add_argument("--binding", required=True, choices=("pyrbdl", "rbdl"))
    parser.add_argument("--fps", required=True, type=float, help="actual input FPS; no resampling")
    parser.add_argument("--epsilon", required=True, type=float, help="central perturbation in radians")
    parser.add_argument("--sampling", required=True, choices=("all", "uniform", "sparse"))
    parser.add_argument("--weight-mode", required=True, choices=("prototype-adaptive", "paper-fixed"))
    parser.add_argument("--formulation", required=True, choices=("prototype-slice",), help="score torque components tau[6:30] using the angular-vector state assignment")
    parser.add_argument("--gravity", required=True, nargs=3, type=float, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--representation", required=True, choices=("axis-angle", "matrix"))
    args = parser.parse_args(argv)
    try:
        import numpy as np
        import scipy
        from scipy.spatial.transform import Rotation
        from . import __version__
        from .core import Settings, real_numeric_array, score_motion, validate_motion
        from .dynamics import RBDLDynamics

        settings = Settings(args.fps, args.epsilon, args.sampling, args.weight_mode, args.formulation)
        source = args.input.resolve(strict=True)
        urdf = args.urdf.resolve(strict=True)
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"refusing to overwrite {output}")
        if not output.parent.is_dir():
            raise ValueError(f"output parent does not exist: {output.parent}")
        with np.load(source, allow_pickle=False) as data:
            pose = real_numeric_array(data["pose"], "poses")
            tran = real_numeric_array(data["tran"], "translations")
        if args.representation == "axis-angle":
            if pose.ndim != 3 or pose.shape[1:] != (24, 3) or not np.isfinite(pose).all():
                raise ValueError("axis-angle pose must be finite (T,24,3) in radians")
            pose = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        pose, tran = validate_motion(pose, tran)
        dynamics = RBDLDynamics(urdf, args.binding, args.gravity)
        print("Scoring motion clip with TVS.", file=sys.stderr)
        result = score_motion(pose, tran, dynamics, settings)
        result["provenance"] = {
            "input": str(source), "urdf": str(urdf), "binding": args.binding,
            "gravity": args.gravity, "representation": args.representation,
            "tvs_version": __version__, "python": sys.version,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "binding_version": str(getattr(dynamics.api, "__version__", "unknown")),
        }
        for name, path in (("input", source), ("urdf", urdf)):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result["provenance"][name + "_sha256"] = digest.hexdigest()
        text = json.dumps(result, indent=2, allow_nan=False)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        print(f"Score: {result['final_score']:.8g}; output: {output}")
        return 0
    except Exception as exc:
        print(f"TVS error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
