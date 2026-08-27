#!/usr/bin/env python3
"""
vgn_mujoco_bridge.py
====================
Bridge between the MuJoCo MMO-700 simulation (Python 3.13) and VGN
(Python 3.10 in vgn_env conda, torch + open3d).

The Problem
-----------
vgn_env packages (torch, open3d) are compiled for Python 3.10. The MuJoCo
environment runs Python 3.13. Mixing them in the same interpreter causes ABI errors.

The Solution
------------
VGN inference runs in a subprocess using vgn_env's Python 3.10 interpreter
via the companion script `vgn_worker.py`. Depth images + TSDF params are
written to a temporary .npz file; results come back as a JSON line on stdout.

Public API
----------
VGNBridge(model_path, tsdf_size, tsdf_resolution)
    .init_tsdf(box_approx_pos)
    .collect_view(model, data, renderer, cam_name, box_approx_pos) -> int
    .run_inference() -> bool
    .best_grasp_world() -> (pos_world, rot_mat_3x3, width_m, score) | None

Helpers
-------
read_lidar(model, data, sensor_name) -> float   # metres (+inf = no obstacle)
scan_arm_configs(n_views) -> list[float]         # shoulder_pan angles for scan
VGN_OK : bool                                    # True if vgn_env Python found
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_DIR = Path(__file__).resolve().parent.parent
_VGN_SRC = _REPO_DIR / "vgn" / "src"
_WORKER_PATH = Path(__file__).resolve().parent / "vgn_worker.py"


def _resolve_vgn_python() -> Path:
    """Dynamically locate the vgn_env Python binary across macOS and Linux."""
    search_candidates = [
        Path(sys.executable),
        Path("/Users/shrikar/anaconda3/envs/vgn_env/bin/python3"),
        Path("/Users/shrikar/miniconda3/envs/vgn_env/bin/python3"),
        Path.home() / "anaconda3" / "envs" / "vgn_env" / "bin" / "python3",
        Path.home() / "miniconda3" / "envs" / "vgn_env" / "bin" / "python3",
        Path.home() / "anaconda3" / "envs" / "vgn_env" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "vgn_env" / "bin" / "python",
        Path("/home/shrikar/anaconda3/envs/vgn_env/bin/python3"),
        Path("/home/shrikar/anaconda3/envs/vgn_env/bin/python"),
    ]

    for candidate in search_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    which_py = shutil.which("python3") or shutil.which("python")
    return Path(which_py) if which_py else search_candidates[1]


_VGN_ENV_PYTHON = _resolve_vgn_python()

DEFAULT_MODEL_PATH = _REPO_DIR / "vgn" / "data" / "models" / "vgn_conv.pth"
DEFAULT_TSDF_SIZE = 0.30        # metres
DEFAULT_TSDF_RESOLUTION = 40    # voxels per side -> ~7.5 mm voxel
MAX_DEPTH = 2.0                 # clip depth beyond this value

# ── Availability check ────────────────────────────────────────────────────────
VGN_OK = _VGN_ENV_PYTHON.exists() and _WORKER_PATH.exists()
if not VGN_OK:
    log.warning(
        f"VGN unavailable: "
        f"vgn_env Python exists={_VGN_ENV_PYTHON.exists()}, "
        f"vgn_worker.py exists={_WORKER_PATH.exists()}. "
        "Falling back to depth-unproject."
    )


# ==============================================================================
class VGNBridge:
    """
    Manages TSDF depth-view collection and VGN inference via subprocess.

    Parameters
    ----------
    model_path       : Path to vgn_conv.pth weights.
    tsdf_size        : Cube side length (m) for the TSDF volume.
    tsdf_resolution  : Voxels per side (40 -> 7.5 mm voxels).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        tsdf_size: float = DEFAULT_TSDF_SIZE,
        tsdf_resolution: int = DEFAULT_TSDF_RESOLUTION,
    ):
        self.model_path = Path(model_path)
        self.tsdf_size = tsdf_size
        self.tsdf_resolution = tsdf_resolution
        self.available = VGN_OK and self.model_path.exists()

        self._views: list = []
        self._tsdf_origin: Optional[np.ndarray] = None
        self._grasp_pos_tsdf: Optional[np.ndarray] = None
        self._grasp_rot: Optional[np.ndarray] = None
        self._grasp_width: float = 0.0
        self._grasp_score: float = 0.0
        self._n_candidates: int = 0

        if not VGN_OK:
            print("  [VGN] vgn_env Python not found — fallback: depth-unproject")
        elif not self.model_path.exists():
            print(f"  [VGN] Model not found: {self.model_path} — fallback active")
            self.available = False
        else:
            print(
                f"  [VGN] Ready  model={self.model_path.name}  "
                f"Python={_VGN_ENV_PYTHON}  worker={_WORKER_PATH.name}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def init_tsdf(self, box_approx_pos: np.ndarray) -> None:
        """Initialise a fresh TSDF centred on box_approx_pos."""
        half = self.tsdf_size / 2.0
        self._tsdf_origin = box_approx_pos.copy() - np.array([half, half, half])
        self._views = []
        self._grasp_pos_tsdf = None
        self._grasp_rot = None
        self._grasp_width = 0.0
        self._grasp_score = 0.0
        self._n_candidates = 0

    def collect_view(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        renderer: mujoco.Renderer,
        cam_name: str,
        box_approx_pos: np.ndarray,
    ) -> int:
        """
        Render one depth frame from the current arm pose and store it.
        Initialises the TSDF on the first call.
        Returns the number of views collected so far (0 if unavailable).
        """
        if not self.available:
            return 0

        if self._tsdf_origin is None:
            self.init_tsdf(box_approx_pos)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = float(model.cam_fovy[cam_id])
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()  # world<-cam

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render().copy().astype(np.float32)
        depth[depth > MAX_DEPTH] = 0.0
        depth[depth < 1e-6] = 0.0

        self._views.append(
            {
                "depth": depth,
                "cam_pos": cam_pos,
                "cam_mat": cam_mat.flatten(),  # flat 9-elem array
                "img_h": renderer.height,
                "img_w": renderer.width,
                "fovy": fovy,
            }
        )
        return len(self._views)

    def run_inference(self) -> bool:
        """
        Run VGN inference on accumulated views via vgn_env subprocess.
        Returns True if >= 1 grasp candidate found.
        """
        if not self.available or not self._views or self._tsdf_origin is None:
            return False

        # ── Serialise views to temp npz ───────────────────────────────────
        npz_data: dict = {
            "n_views": np.array(len(self._views)),
            "tsdf_origin": self._tsdf_origin,
        }
        for i, v in enumerate(self._views):
            npz_data[f"depth_{i}"] = v["depth"]
            npz_data[f"cam_pos_{i}"] = v["cam_pos"]
            npz_data[f"cam_mat_{i}"] = v["cam_mat"]
            npz_data[f"fovy_{i}"] = np.array(v["fovy"])
            npz_data[f"img_h_{i}"] = np.array(v["img_h"])
            npz_data[f"img_w_{i}"] = np.array(v["img_w"])

        with tempfile.NamedTemporaryFile(
            suffix=".npz", delete=False, prefix="vgn_views_"
        ) as nf:
            npz_path = nf.name
        np.savez_compressed(npz_path, **npz_data)

        # ── Run subprocess ─────────────────────────────────────────────────
        try:
            result = subprocess.run(
                [
                    str(_VGN_ENV_PYTHON),
                    str(_WORKER_PATH),
                    npz_path,
                    str(self.model_path),
                    str(self.tsdf_size),
                    str(self.tsdf_resolution),
                    str(_VGN_SRC),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"  [VGN worker stderr]:\n{result.stderr[-800:]}")

            out = result.stdout.strip()
            if not out:
                print("  [VGN] Worker produced no output")
                if result.stderr:
                    print(f"  [VGN worker stderr]: {result.stderr[-500:]}")
                return False

            res = json.loads(out)
            if res.get("error"):
                print(f"  [VGN worker] Error:\n{res['error']}")
                return False

            self._n_candidates = res.get("n_candidates", 0)
            if not res.get("found"):
                print("  [VGN] No grasp candidates above quality threshold")
                return False

            self._grasp_pos_tsdf = np.array(res["pos_tsdf"])
            self._grasp_rot = np.array(res["rot_flat"]).reshape(3, 3)
            self._grasp_width = float(res["width_m"])
            self._grasp_score = float(res["score"])
            print(
                f"  [VGN] {self._n_candidates} candidate(s)  "
                f"best score={self._grasp_score:.3f}"
            )
            return True

        except subprocess.TimeoutExpired:
            print("  [VGN] Subprocess timed out (>120 s)")
            return False
        except json.JSONDecodeError as exc:
            print(f"  [VGN] JSON parse error: {exc}  raw: {out[:200]}")
            return False
        except Exception as exc:
            print(f"  [VGN] Subprocess error: {exc}")
            return False
        finally:
            try:
                os.unlink(npz_path)
            except OSError:
                pass

    def best_grasp_world(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Return (pos_world, rot_mat_3x3, width_m, score) or None.

        pos_world : grasp centre in world frame (m)
        rot_mat   : 6-DOF grasp orientation from VGN (3x3 rotation matrix)
        width_m   : predicted finger opening width (m)
        score     : grasp quality score [0, 1]
        """
        if self._grasp_pos_tsdf is None or self._tsdf_origin is None:
            return None
        pos_world = self._tsdf_origin + self._grasp_pos_tsdf
        return (
            pos_world,
            self._grasp_rot.copy(),
            self._grasp_width,
            self._grasp_score,
        )

    @property
    def n_views(self) -> int:
        return len(self._views)


# ==============================================================================
# Standalone helpers
# ==============================================================================

def scan_arm_configs(n_views: int = 4) -> list:
    """
    Return n_views shoulder_pan angles (rad) evenly spanning [-0.35, +0.35].
    Focuses directly on object for fast TSDF volume capture.
    """
    return list(np.linspace(-0.35, 0.35, n_views))


def read_lidar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> float:
    """
    Read a MuJoCo rangefinder sensor by name.
    Returns distance in metres, or +inf when no intersection detected
    (MuJoCo reports -1 for out-of-range).
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        return float("inf")
    adr = model.sensor_adr[sid]
    val = float(data.sensordata[adr])
    return val if val > 0.0 else float("inf")
