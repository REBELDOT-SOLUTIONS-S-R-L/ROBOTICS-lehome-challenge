#!/usr/bin/env python3
"""Comprehensive HDF5 dataset inspector for Halton vs Random MimicGen comparison."""

import json
import os
import sys

import h5py
import numpy as np

BASE = "/workspace/IsaacTools/ROBOTICS-lehome-challenge/Datasets/hdf5_mimicgen_pipeline"

FILES = {
    "Recorded Halton": f"{BASE}/1_annotated_teleop/Top_Long_Seen_0-HALTON_25.hdf5",
    "Recorded Random": f"{BASE}/1_annotated_teleop/Top_Long_Seen_0_TEST-1.hdf5",
    "Generated Halton (Success)": f"{BASE}/2_generated/Top_Long_Seen_0-generated-HALTON_24.hdf5",
    "Generated Halton (Failed)": f"{BASE}/2_generated/Top_Long_Seen_0-generated-HALTON_24_failed.hdf5",
    "Generated Random (Success)": f"{BASE}/2_generated/Top_Long_Unseen_0_-generated-TEST-1.hdf5",
    "Generated Random (Failed)": f"{BASE}/2_generated/Top_Long_Unseen_0_-generated-TEST-1_failed.hdf5",
}


def separator(title, char="="):
    width = 100
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def get_demos(f):
    """Return sorted list of demo keys."""
    demos = sorted(
        [k for k in f["data"].keys() if k.startswith("demo_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return demos


def print_file_overview(name, path):
    """Print basic file info."""
    separator(f"FILE: {name}", "=")
    fsize = os.path.getsize(path)
    print(f"  Path: {path}")
    print(f"  Size: {fsize / 1024 / 1024:.1f} MB ({fsize / 1024 / 1024 / 1024:.2f} GB)")

    f = h5py.File(path, "r")

    # Root attrs
    print("\n  Root attributes:")
    for k, v in f.attrs.items():
        print(f"    {k}: {v}")

    # Data group attrs
    print("\n  data/ attributes:")
    for k, v in f["data"].attrs.items():
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:120] + "..."
        print(f"    {k}: {val_str}")

    demos = get_demos(f)
    print(f"\n  Number of episodes: {len(demos)}")

    # Episode lengths
    lengths = []
    successes = []
    for d in demos:
        n = int(f[f"data/{d}"].attrs.get("num_samples", 0))
        s = bool(f[f"data/{d}"].attrs.get("success", False))
        lengths.append(n)
        successes.append(s)

    print(f"  Episode lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    print(f"  Success count: {sum(successes)}/{len(successes)} "
          f"({100 * sum(successes) / len(successes):.1f}%)")

    print(f"\n  Per-episode breakdown:")
    for i, d in enumerate(demos):
        print(f"    {d}: length={lengths[i]}, success={successes[i]}")

    f.close()
    return lengths, successes


def analyze_recorded_dataset(name, path):
    """Deep analysis of recorded (annotated teleop) datasets."""
    separator(f"DETAILED ANALYSIS: {name}", "-")
    f = h5py.File(path, "r")
    demos = get_demos(f)

    # --- Initial garment poses ---
    print("\n  [Garment Initial Poses]")
    print(f"  {'Demo':<12} {'X':>8} {'Y':>8} {'Z':>8} {'Rx':>8} {'Ry':>8} {'Rz':>8}")
    print(f"  {'-'*60}")

    all_poses = []
    for d in demos:
        # Find the garment name
        garment_grp = f[f"data/{d}/initial_state/garment"]
        garment_name = list(garment_grp.keys())[0]
        pose = garment_grp[garment_name]["initial_pose"][:]
        all_poses.append(pose)
        print(f"  {d:<12} {pose[0]:8.4f} {pose[1]:8.4f} {pose[2]:8.4f} "
              f"{pose[3]:8.4f} {pose[4]:8.4f} {pose[5]:8.4f}")

    all_poses = np.array(all_poses)
    print(f"\n  Pose statistics (position x,y,z):")
    for i, label in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
        vals = all_poses[:, i]
        print(f"    {label}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, range={np.max(vals) - np.min(vals):.4f}")

    # --- Garment scale ---
    print("\n  [Garment Scales]")
    for d in demos:
        garment_grp = f[f"data/{d}/initial_state/garment"]
        garment_name = list(garment_grp.keys())[0]
        scale = garment_grp[garment_name]["scale"][:]
        print(f"    {d}: scale={scale}")

    # --- Action distributions ---
    print("\n  [Action Distributions]")
    # Collect all actions
    all_actions = []
    for d in demos:
        acts = f[f"data/{d}/actions"][:]
        all_actions.append(acts)
    all_actions_cat = np.concatenate(all_actions, axis=0)

    action_dim = all_actions_cat.shape[1]
    print(f"  Action shape: {all_actions_cat.shape} (total_steps x {action_dim})")

    # Check if action dimensions include EE pose (16-dim) or joint (12-dim)
    # 16-dim = 7 (pos+quat) * 2 arms + 2 grippers? or similar
    print(f"\n  {'Dim':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"  {'-'*55}")
    for dim in range(action_dim):
        vals = all_actions_cat[:, dim]
        print(f"  {dim:<5} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
              f"{np.min(vals):10.4f} {np.max(vals):10.4f} {np.max(vals) - np.min(vals):10.4f}")

    # Also look at obs/actions (12-dim joint actions)
    obs_actions = []
    for d in demos:
        if f"data/{d}/obs/actions" in f:
            oa = f[f"data/{d}/obs/actions"][:]
            obs_actions.append(oa)
    if obs_actions:
        obs_actions_cat = np.concatenate(obs_actions, axis=0)
        print(f"\n  obs/actions shape: {obs_actions_cat.shape}")
        print(f"  {'Dim':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*45}")
        for dim in range(obs_actions_cat.shape[1]):
            vals = obs_actions_cat[:, dim]
            print(f"  {dim:<5} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
                  f"{np.min(vals):10.4f} {np.max(vals):10.4f}")

    # --- Subtask termination signals ---
    print("\n  [Subtask Termination Signals]")
    signal_names = None
    for d in demos:
        sig_path = f"data/{d}/obs/datagen_info/subtask_term_signals"
        if sig_path in f:
            signal_names = sorted(f[sig_path].keys())
            break

    if signal_names:
        print(f"  Signals: {signal_names}")
        print(f"\n  {'Demo':<12}", end="")
        for sn in signal_names:
            short = sn[:18]
            print(f" {short:>18}", end="")
        print()

        for d in demos:
            sig_path = f"data/{d}/obs/datagen_info/subtask_term_signals"
            if sig_path not in f:
                continue
            print(f"  {d:<12}", end="")
            length = int(f[f"data/{d}"].attrs["num_samples"])
            for sn in signal_names:
                sig = f[f"{sig_path}/{sn}"][:]
                # Find first True index
                true_indices = np.where(sig.flatten())[0]
                if len(true_indices) > 0:
                    first_true = true_indices[0]
                    pct = 100 * first_true / length
                    print(f" {first_true:>8}({pct:4.0f}%)", end="")
                else:
                    print(f" {'NEVER':>18}", end="")
            print()
    else:
        print("  No subtask termination signals found.")

    # --- EEF Pose analysis (object-relative grasping) ---
    print("\n  [Object Keypoint Positions at t=0 (garment layout)]")
    kp_names = None
    for d in demos:
        op_path = f"data/{d}/obs/datagen_info/object_pose"
        if op_path in f:
            kp_names = sorted(f[op_path].keys())
            break

    if kp_names:
        print(f"  Keypoints: {kp_names}")
        for d in demos:
            op_path = f"data/{d}/obs/datagen_info/object_pose"
            if op_path not in f:
                continue
            print(f"\n  {d}:")
            for kn in kp_names:
                pose_mat = f[f"{op_path}/{kn}"][0]  # 4x4 at t=0
                pos = pose_mat[:3, 3]
                print(f"    {kn}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # --- EEF pose trajectory analysis ---
    print("\n  [EEF Pose at first timestep for each demo]")
    for d in demos:
        for arm in ["left_arm", "right_arm"]:
            eef_path = f"data/{d}/obs/datagen_info/eef_pose/{arm}"
            if eef_path in f:
                pose = f[eef_path][0]  # 4x4 at t=0
                pos = pose[:3, 3]
                print(f"    {d} {arm}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    f.close()


def analyze_generated_dataset(name, path):
    """Analyze generated (MimicGen output) datasets."""
    separator(f"DETAILED ANALYSIS: {name}", "-")
    f = h5py.File(path, "r")
    demos = get_demos(f)

    # Success/failure
    successes = []
    lengths = []
    for d in demos:
        s = bool(f[f"data/{d}"].attrs.get("success", False))
        n = int(f[f"data/{d}"].attrs.get("num_samples", 0))
        successes.append(s)
        lengths.append(n)

    print(f"\n  Total demos: {len(demos)}")
    print(f"  Successes: {sum(successes)} / {len(successes)} "
          f"({100 * sum(successes) / max(len(successes), 1):.1f}%)")
    print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # --- Action distributions ---
    print("\n  [Action Distributions]")
    all_actions = []
    for d in demos:
        acts = f[f"data/{d}/actions"][:]
        all_actions.append(acts)
    all_actions_cat = np.concatenate(all_actions, axis=0)

    action_dim = all_actions_cat.shape[1]
    print(f"  Total action steps: {all_actions_cat.shape[0]}, dims: {action_dim}")
    print(f"  {'Dim':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*45}")
    for dim in range(action_dim):
        vals = all_actions_cat[:, dim]
        print(f"  {dim:<5} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
              f"{np.min(vals):10.4f} {np.max(vals):10.4f}")

    # --- Object keypoints at initial timestep (to check cloth position diversity) ---
    # Generated datasets have obs/object_pose/garment_* keys
    print("\n  [Garment Keypoint Positions at t=0]")
    kp_names = None
    for d in demos[:1]:
        op_path = f"data/{d}/obs/object_pose"
        if op_path in f:
            kp_names = sorted(f[op_path].keys())
            break

    if kp_names:
        print(f"  Keypoints: {kp_names}")

        # Collect initial positions for all demos
        kp_positions = {kn: [] for kn in kp_names}
        for d in demos:
            op_path = f"data/{d}/obs/object_pose"
            if op_path not in f:
                continue
            for kn in kp_names:
                pose_mat = f[f"{op_path}/{kn}"][0]  # 4x4 at t=0
                pos = pose_mat[:3, 3]
                kp_positions[kn].append(pos)

        print(f"\n  Keypoint position statistics at t=0:")
        for kn in kp_names:
            positions = np.array(kp_positions[kn])
            if len(positions) == 0:
                continue
            print(f"    {kn}:")
            for i, axis in enumerate(["X", "Y", "Z"]):
                vals = positions[:, i]
                print(f"      {axis}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    # --- Check for correlations between initial positions and success ---
    if kp_names and "garment_center" in kp_names:
        print("\n  [Garment Center Position vs Success/Failure]")
        for d_idx, d in enumerate(demos):
            op_path = f"data/{d}/obs/object_pose/garment_center"
            if op_path in f:
                pos = f[op_path][0][:3, 3]
                status = "OK" if successes[d_idx] else "FAIL"
                print(f"    {d}: center=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
                      f"len={lengths[d_idx]} [{status}]")

    f.close()


def compare_recorded_datasets():
    """Direct comparison between Halton and Random recorded datasets."""
    separator("COMPARISON: Recorded Halton vs Random", "#")

    h_path = FILES["Recorded Halton"]
    r_path = FILES["Recorded Random"]
    fh = h5py.File(h_path, "r")
    fr = h5py.File(r_path, "r")

    demos_h = get_demos(fh)
    demos_r = get_demos(fr)

    print(f"\n  Halton: {len(demos_h)} episodes, Random: {len(demos_r)} episodes")

    # Compare initial poses
    print("\n  [Initial Pose Spread Comparison]")
    poses_h = []
    poses_r = []
    for d in demos_h:
        garment_grp = fh[f"data/{d}/initial_state/garment"]
        gn = list(garment_grp.keys())[0]
        poses_h.append(garment_grp[gn]["initial_pose"][:])
    for d in demos_r:
        garment_grp = fr[f"data/{d}/initial_state/garment"]
        gn = list(garment_grp.keys())[0]
        poses_r.append(garment_grp[gn]["initial_pose"][:])

    poses_h = np.array(poses_h)
    poses_r = np.array(poses_r)

    print(f"\n  {'Dim':<4} {'H_mean':>8} {'H_std':>8} {'H_range':>8}  |  {'R_mean':>8} {'R_std':>8} {'R_range':>8}")
    print(f"  {'-'*70}")
    for i, label in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
        hv = poses_h[:, i]
        rv = poses_r[:, i]
        print(f"  {label:<4} {np.mean(hv):8.4f} {np.std(hv):8.4f} {np.ptp(hv):8.4f}  |  "
              f"{np.mean(rv):8.4f} {np.std(rv):8.4f} {np.ptp(rv):8.4f}")

    # Compare episode lengths
    lens_h = [int(fh[f"data/{d}"].attrs["num_samples"]) for d in demos_h]
    lens_r = [int(fr[f"data/{d}"].attrs["num_samples"]) for d in demos_r]
    print(f"\n  [Episode Length Comparison]")
    print(f"    Halton: mean={np.mean(lens_h):.0f}, std={np.std(lens_h):.0f}, "
          f"range=[{min(lens_h)}, {max(lens_h)}]")
    print(f"    Random: mean={np.mean(lens_r):.0f}, std={np.std(lens_r):.0f}, "
          f"range=[{min(lens_r)}, {max(lens_r)}]")

    # Compare action distributions
    print(f"\n  [Action Distribution Comparison (all dims aggregated)]")
    acts_h = np.concatenate([fh[f"data/{d}/actions"][:] for d in demos_h])
    acts_r = np.concatenate([fr[f"data/{d}/actions"][:] for d in demos_r])

    print(f"    Halton: shape={acts_h.shape}, mean={np.mean(acts_h):.4f}, std={np.std(acts_h):.4f}")
    print(f"    Random: shape={acts_r.shape}, mean={np.mean(acts_r):.4f}, std={np.std(acts_r):.4f}")

    print(f"\n  [Per-Dimension Action Comparison]")
    print(f"  {'Dim':<5} {'H_mean':>8} {'H_std':>8}  |  {'R_mean':>8} {'R_std':>8}  | {'Diff_mean':>10}")
    print(f"  {'-'*60}")
    for dim in range(acts_h.shape[1]):
        hm, hs = np.mean(acts_h[:, dim]), np.std(acts_h[:, dim])
        rm, rs = np.mean(acts_r[:, dim]), np.std(acts_r[:, dim])
        print(f"  {dim:<5} {hm:8.4f} {hs:8.4f}  |  {rm:8.4f} {rs:8.4f}  | {hm - rm:10.4f}")

    # Compare subtask timing
    print(f"\n  [Subtask Timing Comparison (first True step / episode_length)]")
    signal_names = None
    for d in demos_h:
        sig_path = f"data/{d}/obs/datagen_info/subtask_term_signals"
        if sig_path in fh:
            signal_names = sorted(fh[sig_path].keys())
            break

    if signal_names:
        for sn in signal_names:
            h_pcts = []
            r_pcts = []

            for d in demos_h:
                sig = fh[f"data/{d}/obs/datagen_info/subtask_term_signals/{sn}"][:]
                length = int(fh[f"data/{d}"].attrs["num_samples"])
                true_idx = np.where(sig.flatten())[0]
                if len(true_idx) > 0:
                    h_pcts.append(true_idx[0] / length)

            for d in demos_r:
                sig = fr[f"data/{d}/obs/datagen_info/subtask_term_signals/{sn}"][:]
                length = int(fr[f"data/{d}"].attrs["num_samples"])
                true_idx = np.where(sig.flatten())[0]
                if len(true_idx) > 0:
                    r_pcts.append(true_idx[0] / length)

            h_str = f"mean={np.mean(h_pcts):.3f}, std={np.std(h_pcts):.3f}" if h_pcts else "NEVER"
            r_str = f"mean={np.mean(r_pcts):.3f}, std={np.std(r_pcts):.3f}" if r_pcts else "NEVER"
            print(f"    {sn}:")
            print(f"      Halton ({len(h_pcts)}/{len(demos_h)} triggered): {h_str}")
            print(f"      Random ({len(r_pcts)}/{len(demos_r)} triggered): {r_str}")

    # Compare object keypoint spread at t=0
    print(f"\n  [Object Keypoint Spread at t=0]")
    kp_names_h = None
    for d in demos_h:
        op_path = f"data/{d}/obs/datagen_info/object_pose"
        if op_path in fh:
            kp_names_h = sorted(fh[op_path].keys())
            break

    if kp_names_h:
        for kn in kp_names_h:
            h_pos = []
            r_pos = []
            for d in demos_h:
                pose = fh[f"data/{d}/obs/datagen_info/object_pose/{kn}"][0]
                h_pos.append(pose[:3, 3])
            for d in demos_r:
                pose = fr[f"data/{d}/obs/datagen_info/object_pose/{kn}"][0]
                r_pos.append(pose[:3, 3])

            h_pos = np.array(h_pos)
            r_pos = np.array(r_pos)

            print(f"    {kn}:")
            for i, axis in enumerate(["X", "Y", "Z"]):
                print(f"      {axis}: Halton std={np.std(h_pos[:, i]):.4f} range={np.ptp(h_pos[:, i]):.4f}  |  "
                      f"Random std={np.std(r_pos[:, i]):.4f} range={np.ptp(r_pos[:, i]):.4f}")

    fh.close()
    fr.close()


def compare_generated_datasets():
    """Compare generated success/fail datasets."""
    separator("COMPARISON: Generated Halton vs Random", "#")

    pairs = [
        ("Halton", FILES["Generated Halton (Success)"], FILES["Generated Halton (Failed)"]),
        ("Random", FILES["Generated Random (Success)"], FILES["Generated Random (Failed)"]),
    ]

    for label, succ_path, fail_path in pairs:
        print(f"\n  --- {label} ---")
        fs = h5py.File(succ_path, "r")
        ff = h5py.File(fail_path, "r")

        demos_s = get_demos(fs)
        demos_f = get_demos(ff)

        n_s = len(demos_s)
        n_f = len(demos_f)
        total = n_s + n_f

        print(f"    Success: {n_s}, Failed: {n_f}, Total attempts: {total}")
        print(f"    Success rate: {100 * n_s / max(total, 1):.1f}%")

        # Lengths comparison
        lens_s = [int(fs[f"data/{d}"].attrs["num_samples"]) for d in demos_s]
        lens_f = [int(ff[f"data/{d}"].attrs["num_samples"]) for d in demos_f]

        if lens_s:
            print(f"    Success lengths: mean={np.mean(lens_s):.0f}, "
                  f"std={np.std(lens_s):.0f}, range=[{min(lens_s)}, {max(lens_s)}]")
        if lens_f:
            print(f"    Failed lengths: mean={np.mean(lens_f):.0f}, "
                  f"std={np.std(lens_f):.0f}, range=[{min(lens_f)}, {max(lens_f)}]")

        # Check garment center positions at t=0 for success vs failure
        print(f"\n    [Garment Center at t=0: Success vs Failure]")

        def get_center_positions(f_handle, demos_list):
            positions = []
            for d in demos_list:
                op_path = f"data/{d}/obs/object_pose/garment_center"
                if op_path in f_handle:
                    pos = f_handle[op_path][0][:3, 3]
                    positions.append(pos)
            return np.array(positions) if positions else None

        pos_s = get_center_positions(fs, demos_s)
        pos_f = get_center_positions(ff, demos_f)

        if pos_s is not None:
            for i, axis in enumerate(["X", "Y", "Z"]):
                s_str = f"mean={np.mean(pos_s[:, i]):.4f}, std={np.std(pos_s[:, i]):.4f}, range=[{np.min(pos_s[:, i]):.4f}, {np.max(pos_s[:, i]):.4f}]"
                f_str = "N/A"
                if pos_f is not None and len(pos_f) > 0:
                    f_str = f"mean={np.mean(pos_f[:, i]):.4f}, std={np.std(pos_f[:, i]):.4f}, range=[{np.min(pos_f[:, i]):.4f}, {np.max(pos_f[:, i]):.4f}]"
                print(f"      {axis} Success: {s_str}")
                print(f"      {axis} Failed:  {f_str}")

        # Check action distributions for success vs failure
        print(f"\n    [Action Distribution: Success vs Failure]")
        if demos_s:
            acts_s = np.concatenate([fs[f"data/{d}/actions"][:] for d in demos_s])
            print(f"      Success actions: shape={acts_s.shape}, "
                  f"mean={np.mean(acts_s):.4f}, std={np.std(acts_s):.4f}")
        if demos_f:
            acts_f = np.concatenate([ff[f"data/{d}/actions"][:] for d in demos_f])
            print(f"      Failed actions: shape={acts_f.shape}, "
                  f"mean={np.mean(acts_f):.4f}, std={np.std(acts_f):.4f}")

        # Check all keypoint initial positions to see if failed demos have cloth in extreme positions
        print(f"\n    [All Keypoint Spread at t=0: Success vs Failure]")
        kp_names = None
        for d in demos_s[:1]:
            op_path = f"data/{d}/obs/object_pose"
            if op_path in fs:
                kp_names = sorted(fs[op_path].keys())
                break

        if kp_names:
            for kn in ["garment_left_lower", "garment_right_lower", "garment_left_upper", "garment_right_upper"]:
                if kn not in kp_names:
                    continue
                s_pos = []
                f_pos = []
                for d in demos_s:
                    p = fs[f"data/{d}/obs/object_pose/{kn}"][0][:3, 3]
                    s_pos.append(p)
                for d in demos_f:
                    p = ff[f"data/{d}/obs/object_pose/{kn}"][0][:3, 3]
                    f_pos.append(p)
                s_pos = np.array(s_pos)
                f_pos = np.array(f_pos)

                print(f"      {kn}:")
                for i, axis in enumerate(["X", "Y", "Z"]):
                    if len(s_pos) > 0 and len(f_pos) > 0:
                        print(f"        {axis}: Succ mean={np.mean(s_pos[:, i]):.4f} std={np.std(s_pos[:, i]):.4f} | "
                              f"Fail mean={np.mean(f_pos[:, i]):.4f} std={np.std(f_pos[:, i]):.4f}")

        # Per-episode details for failed dataset
        print(f"\n    [Failed Episodes Detail (first 20)]")
        for d in demos_f[:20]:
            length = int(ff[f"data/{d}"].attrs["num_samples"])
            op_path = f"data/{d}/obs/object_pose/garment_center"
            pos_str = "N/A"
            if op_path in ff:
                pos = ff[op_path][0][:3, 3]
                pos_str = f"center=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
            print(f"      {d}: len={length}, {pos_str}")

        fs.close()
        ff.close()


def analyze_subtask_consistency():
    """Check if subtask annotations are consistent and plausible."""
    separator("SUBTASK ANNOTATION QUALITY CHECK", "#")

    for name, path in [("Recorded Halton", FILES["Recorded Halton"]),
                        ("Recorded Random", FILES["Recorded Random"])]:
        print(f"\n  --- {name} ---")
        f = h5py.File(path, "r")
        demos = get_demos(f)

        signal_order = [
            "grasp_left_lower", "grasp_right_lower",
            "left_middle_to_lower", "right_middle_to_lower",
            "grasp_left_middle", "grasp_right_middle",
            "left_lower_to_upper", "right_lower_to_upper",
            "left_return_home", "right_return_home",
        ]

        for d in demos:
            sig_path = f"data/{d}/obs/datagen_info/subtask_term_signals"
            if sig_path not in f:
                continue
            length = int(f[f"data/{d}"].attrs["num_samples"])

            # Get activation timesteps for each signal
            activation = {}
            for sn in signal_order:
                if sn in f[sig_path]:
                    sig = f[f"{sig_path}/{sn}"][:]
                    true_idx = np.where(sig.flatten())[0]
                    activation[sn] = int(true_idx[0]) if len(true_idx) > 0 else None
                else:
                    activation[sn] = None

            # Check ordering: signals should activate in roughly the expected sequence
            prev_step = -1
            issues = []
            for sn in signal_order:
                step = activation.get(sn)
                if step is not None:
                    if step < prev_step:
                        issues.append(f"{sn} at {step} < prev {prev_step}")
                    prev_step = step

            # Compute subtask segment lengths
            activated_signals = [(sn, activation[sn]) for sn in signal_order if activation[sn] is not None]
            segments = []
            for i in range(len(activated_signals)):
                start = activated_signals[i - 1][1] if i > 0 else 0
                end = activated_signals[i][1]
                segments.append((activated_signals[i][0], end - start))

            status = "OK" if not issues else f"ISSUES: {issues}"
            print(f"\n    {d} (len={length}) [{status}]")
            for sn, seg_len in segments:
                pct = 100 * seg_len / length
                bar = "#" * int(pct / 2)
                print(f"      {sn:<25} step_delta={seg_len:>4} ({pct:5.1f}%) {bar}")

        f.close()


def analyze_eef_workspace():
    """Check if EEF reaches extreme positions in Halton demos."""
    separator("EEF WORKSPACE ANALYSIS (Recorded Datasets)", "#")

    for name, path in [("Recorded Halton", FILES["Recorded Halton"]),
                        ("Recorded Random", FILES["Recorded Random"])]:
        print(f"\n  --- {name} ---")
        f = h5py.File(path, "r")
        demos = get_demos(f)

        all_left_pos = []
        all_right_pos = []

        for d in demos:
            for arm, collector in [("left_arm", all_left_pos), ("right_arm", all_right_pos)]:
                eef_path = f"data/{d}/obs/datagen_info/eef_pose/{arm}"
                if eef_path in f:
                    poses = f[eef_path][:]  # (T, 4, 4)
                    positions = poses[:, :3, 3]  # (T, 3)
                    collector.append(positions)

        all_left_pos = np.concatenate(all_left_pos) if all_left_pos else np.array([])
        all_right_pos = np.concatenate(all_right_pos) if all_right_pos else np.array([])

        if len(all_left_pos) > 0:
            print(f"    Left arm EEF positions (all timesteps):")
            for i, axis in enumerate(["X", "Y", "Z"]):
                vals = all_left_pos[:, i]
                print(f"      {axis}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

        if len(all_right_pos) > 0:
            print(f"    Right arm EEF positions (all timesteps):")
            for i, axis in enumerate(["X", "Y", "Z"]):
                vals = all_right_pos[:, i]
                print(f"      {axis}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

        f.close()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Verify all files exist
    print("Checking file existence...")
    for name, path in FILES.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / 1024 / 1024 if exists else 0
        print(f"  {'OK' if exists else 'MISSING'} {name}: {size:.1f} MB")

    # 1. File overviews
    for name, path in FILES.items():
        if os.path.exists(path):
            print_file_overview(name, path)

    # 2. Detailed recorded dataset analysis
    for name in ["Recorded Halton", "Recorded Random"]:
        if os.path.exists(FILES[name]):
            analyze_recorded_dataset(name, FILES[name])

    # 3. Detailed generated dataset analysis
    for name in ["Generated Halton (Success)", "Generated Halton (Failed)",
                  "Generated Random (Success)", "Generated Random (Failed)"]:
        if os.path.exists(FILES[name]):
            analyze_generated_dataset(name, FILES[name])

    # 4. Direct comparisons
    compare_recorded_datasets()
    compare_generated_datasets()

    # 5. Subtask annotation quality
    analyze_subtask_consistency()

    # 6. EEF workspace
    analyze_eef_workspace()

    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)
