"""Convert top-level HDF5 actions between joint and IK pose formats.

This is the cloth-task counterpart of LeIsaac's MimicGen eef_action_process.py.
It rewrites only /data/demo_*/actions and updates /data attrs["actions_mode"].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _to_2d_array(arr: np.ndarray) -> np.ndarray:
    """Normalize action-like arrays to [T, D]."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Expected 2D or [1,T,D] array, got shape={tuple(arr.shape)}")


def _build_ik_actions_from_demo(src_demo: h5py.Group) -> np.ndarray:
    """Build IK actions [T,16] (bimanual) or [T,8] (single-arm) from demo fields."""
    if "actions" not in src_demo:
        raise KeyError("Demo is missing top-level 'actions' dataset.")
    if "obs" not in src_demo:
        raise KeyError("Demo is missing 'obs' group.")

    joint_actions = _to_2d_array(np.asarray(src_demo["actions"]))
    obs_group = src_demo["obs"]

    # Bimanual cloth format.
    if "left_ee_frame_state" in obs_group and "right_ee_frame_state" in obs_group:
        left = _to_2d_array(np.asarray(obs_group["left_ee_frame_state"])).astype(np.float32, copy=False)
        right = _to_2d_array(np.asarray(obs_group["right_ee_frame_state"])).astype(np.float32, copy=False)
        if left.shape[1] != 7 or right.shape[1] != 7:
            raise ValueError(
                "Expected left/right ee_frame_state to be [T,7], got "
                f"{tuple(left.shape)} and {tuple(right.shape)}."
            )
        if joint_actions.shape[1] < 12:
            raise ValueError(
                "Expected joint actions with >=12 dims for bimanual conversion, got "
                f"{joint_actions.shape[1]}."
            )
        if left.shape[0] != joint_actions.shape[0] or right.shape[0] != joint_actions.shape[0]:
            raise ValueError(
                "Action and ee_frame_state horizon mismatch: "
                f"actions={joint_actions.shape[0]}, left={left.shape[0]}, right={right.shape[0]}."
            )

        out = np.zeros((joint_actions.shape[0], 16), dtype=np.float32)
        out[:, :7] = left[:, :7]
        out[:, 7] = joint_actions[:, 5]
        out[:, 8:15] = right[:, :7]
        out[:, 15] = joint_actions[:, 11]
        return out

    # Single-arm convention (kept for compatibility).
    if "ee_frame_state" in obs_group:
        eef = _to_2d_array(np.asarray(obs_group["ee_frame_state"])).astype(np.float32, copy=False)
        if eef.shape[1] != 7:
            raise ValueError(f"Expected ee_frame_state to be [T,7], got {tuple(eef.shape)}.")
        if eef.shape[0] != joint_actions.shape[0]:
            raise ValueError(
                "Action and ee_frame_state horizon mismatch: "
                f"actions={joint_actions.shape[0]}, eef={eef.shape[0]}."
            )
        out = np.zeros((joint_actions.shape[0], 8), dtype=np.float32)
        out[:, :7] = eef[:, :7]
        out[:, 7] = joint_actions[:, -1]
        return out

    # Fallback: observation.ee_pose already contains [pos+quat+gripper] blocks.
    if "ee_pose" in obs_group:
        ee_pose = _to_2d_array(np.asarray(obs_group["ee_pose"])).astype(np.float32, copy=False)
        if ee_pose.shape[0] != joint_actions.shape[0]:
            raise ValueError(
                "Action and ee_pose horizon mismatch: "
                f"actions={joint_actions.shape[0]}, ee_pose={ee_pose.shape[0]}."
            )
        if ee_pose.shape[1] not in (8, 16):
            raise ValueError(f"Unsupported obs/ee_pose dim {ee_pose.shape[1]}, expected 8 or 16.")
        return ee_pose

    raise ValueError(
        "Cannot build IK actions: expected obs/left_ee_frame_state+right_ee_frame_state "
        "or obs/ee_frame_state or obs/ee_pose."
    )


def _infer_output_ik_quat_order(
    src_demo: h5py.Group,
    preserve_existing: bool,
    source_quat_order: str | None,
) -> str:
    """Infer quaternion order for rewritten IK actions."""
    if preserve_existing:
        return source_quat_order if source_quat_order in {"xyzw", "wxyz"} else "xyzw"

    obs_group = src_demo["obs"]
    if "left_ee_frame_state" in obs_group and "right_ee_frame_state" in obs_group:
        return "wxyz"
    if "ee_frame_state" in obs_group:
        return "wxyz"
    if "ee_pose" in obs_group:
        return source_quat_order if source_quat_order in {"xyzw", "wxyz"} else "xyzw"
    return "xyzw"


def _build_joint_actions_from_demo(src_demo: h5py.Group) -> np.ndarray:
    """Recover joint actions from obs/actions."""
    if "obs" not in src_demo or "actions" not in src_demo["obs"]:
        raise KeyError("Cannot convert to joint: missing obs/actions.")
    obs_actions = _to_2d_array(np.asarray(src_demo["obs"]["actions"])).astype(np.float32, copy=False)
    return obs_actions


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Replace dataset in-place while preserving gzip compression."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert top-level HDF5 actions to IK or joint format.")
    parser.add_argument("--input_file", type=str, required=True, help="Input HDF5 dataset.")
    parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 dataset.")
    parser.add_argument("--to_ik", action="store_true", help="Convert top-level actions to IK format.")
    parser.add_argument("--to_joint", action="store_true", help="Convert top-level actions to joint format.")
    parser.add_argument(
        "--force_rebuild_from_obs",
        action="store_true",
        help=(
            "When --to_ik is used and source actions are already ee_pose, force rebuilding actions "
            "from obs/*_ee_frame_state instead of preserving existing top-level actions."
        ),
    )
    args = parser.parse_args()

    if args.to_ik == args.to_joint:
        raise ValueError("Specify exactly one of --to_ik or --to_joint.")

    input_path = Path(args.input_file).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src_file, h5py.File(output_path, "w") as dst_file:
        # Copy full file first (including metadata), then rewrite per-demo actions.
        for attr_name, attr_value in src_file.attrs.items():
            dst_file.attrs[attr_name] = attr_value
        for key in src_file.keys():
            src_file.copy(key, dst_file)

        if "data" not in dst_file:
            raise KeyError("Input file has no /data group.")
        dst_data = dst_file["data"]
        src_data = src_file["data"]

        demo_names = sorted(
            [name for name in src_data.keys() if name.startswith("demo_")],
            key=lambda name: int(name.split("_", maxsplit=1)[1]) if name.split("_", maxsplit=1)[1].isdigit() else 10**9,
        )
        if not demo_names:
            raise ValueError("No demo_* groups found under /data.")

        source_actions_mode = str(dst_data.attrs.get("actions_mode", "")).strip().lower()
        raw_source_quat_order = dst_data.attrs.get("ik_quat_order")
        source_ik_quat_order = str(raw_source_quat_order).strip().lower() if raw_source_quat_order is not None else None
        output_ik_quat_order: str | None = None

        for demo_name in demo_names:
            src_demo = src_data[demo_name]
            dst_demo = dst_data[demo_name]

            if args.to_ik:
                existing_actions = _to_2d_array(np.asarray(src_demo["actions"])).astype(np.float32, copy=False)
                preserve_existing = (
                    not args.force_rebuild_from_obs
                    and source_actions_mode == "ee_pose"
                    and int(existing_actions.shape[1]) in (8, 16)
                )
                if preserve_existing:
                    new_actions = existing_actions
                    print(
                        f"[{demo_name}] Preserving existing ee_pose actions with shape "
                        f"{tuple(new_actions.shape)} (use --force_rebuild_from_obs to rebuild from observations)."
                    )
                else:
                    new_actions = _build_ik_actions_from_demo(src_demo)
                demo_quat_order = _infer_output_ik_quat_order(
                    src_demo,
                    preserve_existing=preserve_existing,
                    source_quat_order=source_ik_quat_order,
                )
                if output_ik_quat_order is None:
                    output_ik_quat_order = demo_quat_order
                elif output_ik_quat_order != demo_quat_order:
                    raise ValueError(
                        "Inconsistent IK quaternion order inferred across demos: "
                        f"{output_ik_quat_order} vs {demo_quat_order}."
                    )
            else:
                new_actions = _build_joint_actions_from_demo(src_demo)

            _replace_dataset(dst_demo, "actions", new_actions.astype(np.float32, copy=False))
            dst_demo.attrs["num_samples"] = int(new_actions.shape[0])

        dst_data.attrs["actions_mode"] = "ee_pose" if args.to_ik else "joint"
        if args.to_ik:
            # IK actions built from ee_frame_state / observation.ee_pose are base-frame by construction.
            dst_data.attrs["actions_frame"] = "base"
            dst_data.attrs["ik_quat_order"] = output_ik_quat_order or "xyzw"
        else:
            if "actions_frame" in dst_data.attrs:
                del dst_data.attrs["actions_frame"]
            if "ik_quat_order" in dst_data.attrs:
                del dst_data.attrs["ik_quat_order"]

    mode = "IK" if args.to_ik else "joint"
    print(f"Converted actions to {mode} and wrote: {output_path}")


if __name__ == "__main__":
    main()
