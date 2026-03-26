"""Offline visualization for generation pose traces.

Reads the CSV emitted by scripts/mimicgen/dataset_generate_dataset.py and plots one
episode at a time after generation, including a 3D trajectory view for EEFs and
garment keypoints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TRACE_EEF_KEYPOINT_COLUMNS = (
    "dist_left_arm_to_garment_left_middle_m",
    "dist_left_arm_to_garment_left_lower_m",
    "dist_left_arm_to_garment_left_upper_m",
    "dist_right_arm_to_garment_right_middle_m",
    "dist_right_arm_to_garment_right_lower_m",
    "dist_right_arm_to_garment_right_upper_m",
)
TRACE_TERM_COLUMNS = (
    ("dist_left_middle_to_lower_m", "threshold_left_middle_to_lower_m"),
    ("dist_right_middle_to_lower_m", "threshold_right_middle_to_lower_m"),
    ("dist_left_lower_to_upper_m", "threshold_left_lower_to_upper_m"),
    ("dist_right_lower_to_upper_m", "threshold_right_lower_to_upper_m"),
)
TRACE_Z_COLUMNS = (
    "eef_left_arm_z",
    "eef_right_arm_z",
    "keypoint_garment_left_middle_z",
    "keypoint_garment_left_lower_z",
    "keypoint_garment_left_upper_z",
    "keypoint_garment_right_middle_z",
    "keypoint_garment_right_lower_z",
    "keypoint_garment_right_upper_z",
)
TRACE_3D_SPECS = (
    ("left eef", "eef_left_arm", "#1f77b4", "-", 2.2),
    ("right eef", "eef_right_arm", "#d62728", "-", 2.2),
    ("left upper", "keypoint_garment_left_upper", "#9467bd", "--", 1.6),
    ("left middle", "keypoint_garment_left_middle", "#17becf", "--", 1.6),
    ("left lower", "keypoint_garment_left_lower", "#2ca02c", "--", 1.6),
    ("right upper", "keypoint_garment_right_upper", "#e377c2", "--", 1.6),
    ("right middle", "keypoint_garment_right_middle", "#ff7f0e", "--", 1.6),
    ("right lower", "keypoint_garment_right_lower", "#8c564b", "--", 1.6),
)
INT_COLUMNS = {
    "step",
    "env_id",
    "episode_index",
    "episode_step",
    "completed_attempts",
    "completed_successes",
}


def _parse_scalar(column: str, value: str | None) -> Any:
    if value is None or value == "":
        return None
    if column in INT_COLUMNS or column.startswith("pass_"):
        try:
            return int(float(value))
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return value


def _load_rows(trace_file: Path) -> list[dict[str, Any]]:
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    rows: list[dict[str, Any]] = []
    with trace_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {key: _parse_scalar(key, value) for key, value in raw_row.items()}
            if row.get("step") is None:
                continue
            rows.append(row)
    return rows


def _select_episode_rows(rows: list[dict[str, Any]], env_id: int, episode_arg: str) -> tuple[list[dict[str, Any]], int | None]:
    env_rows = [row for row in rows if row.get("env_id") == env_id]
    if not env_rows:
        return [], None

    episode_ids = sorted(
        {int(row["episode_index"]) for row in env_rows if row.get("episode_index") is not None}
    )
    if not episode_ids:
        return env_rows, None

    episode_index = episode_ids[-1] if episode_arg == "latest" else int(episode_arg)
    episode_rows = [row for row in env_rows if row.get("episode_index") == episode_index]
    return episode_rows, episode_index


def _available_episodes(rows: list[dict[str, Any]], env_id: int) -> list[int]:
    return sorted(
        {int(row["episode_index"]) for row in rows if row.get("env_id") == env_id and row.get("episode_index") is not None}
    )


def _series(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [row.get(key) for row in rows]


def _x_axis(rows: list[dict[str, Any]]) -> list[float]:
    if rows and all(row.get("episode_step") is not None for row in rows):
        return [float(row["episode_step"]) for row in rows]
    return [float(i) for i in range(len(rows))]


def _xyz_series(rows: list[dict[str, Any]], prefix: str) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for row in rows:
        x = row.get(f"{prefix}_x")
        y = row.get(f"{prefix}_y")
        z = row.get(f"{prefix}_z")
        if x is None or y is None or z is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    return xs, ys, zs


def _set_equal_3d_limits(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_3d_trajectories(ax, rows: list[dict[str, Any]]) -> None:
    all_points: list[np.ndarray] = []
    for label, prefix, color, linestyle, linewidth in TRACE_3D_SPECS:
        xs, ys, zs = _xyz_series(rows, prefix)
        if not xs:
            continue
        xyz = np.column_stack((xs, ys, zs))
        all_points.append(xyz)
        ax.plot(xs, ys, zs, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        ax.scatter(xs[0], ys[0], zs[0], color=color, marker="o", s=28, alpha=0.9)
        ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker="x", s=42, alpha=0.9)

    if all_points:
        _set_equal_3d_limits(ax, np.concatenate(all_points, axis=0))
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax.set_title("3D EEF And Garment Keypoint Trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.3)


def _plot_episode(fig, axes, ax3d, trace_file: Path, rows: list[dict[str, Any]], env_id: int, episode_index: int | None) -> None:
    if ax3d is not None:
        ax3d.clear()
    for ax in axes:
        ax.clear()
        ax.set_axis_on()

    if not rows:
        if ax3d is not None:
            ax3d.text(0.5, 0.5, 0.5, "No rows available yet.", ha="center", va="center")
            ax3d.set_axis_off()
        axes[0].text(0.5, 0.5, "No rows available yet.", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()
        fig.canvas.draw_idle()
        return

    x = _x_axis(rows)
    last_row = rows[-1]
    title_episode = "unknown" if episode_index is None else str(episode_index)
    fig.suptitle(
        f"{trace_file.name} | env={env_id} | episode={title_episode} | "
        f"steps={len(rows)} | completed={last_row.get('completed_attempts', 0)} | "
        f"successes={last_row.get('completed_successes', 0)}"
    )
    if ax3d is not None:
        _plot_3d_trajectories(ax3d, rows)

    for column in TRACE_EEF_KEYPOINT_COLUMNS:
        values = _series(rows, column)
        if any(value is not None for value in values):
            axes[0].plot(x, values, label=column.replace("dist_", "").replace("_m", ""))
    axes[0].set_ylabel("distance [m]")
    axes[0].set_title("EEF To Keypoint Distances")
    axes[0].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", fontsize=8)

    for dist_column, threshold_column in TRACE_TERM_COLUMNS:
        dist_values = _series(rows, dist_column)
        if any(value is not None for value in dist_values):
            line = axes[1].plot(x, dist_values, label=dist_column.replace("dist_", "").replace("_m", ""))[0]
            threshold_value = next((row.get(threshold_column) for row in rows if row.get(threshold_column) is not None), None)
            if threshold_value is not None:
                axes[1].axhline(
                    float(threshold_value),
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.5,
                )
    axes[1].set_ylabel("distance [m]")
    axes[1].set_title("Garment Fold-Term Distances")
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="upper right", fontsize=8)

    for column in TRACE_Z_COLUMNS:
        values = _series(rows, column)
        if any(value is not None for value in values):
            axes[2].plot(x, values, label=column.replace("keypoint_", "").replace("eef_", ""))
    axes[2].set_xlabel("episode_step")
    axes[2].set_ylabel("z [m]")
    axes[2].set_title("EEF And Garment Keypoint Heights")
    axes[2].grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(loc="upper right", fontsize=8, ncol=2)

    fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="[Support Tool] Offline viewer for garment generation pose trace CSV.")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to pose trace CSV.")
    parser.add_argument(
        "--episode",
        type=str,
        default="latest",
        help='Episode index to visualize, or "latest" for the newest recorded episode.',
    )
    parser.add_argument("--env_id", type=int, default=0, help="Environment id to visualize.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--list_episodes",
        action="store_true",
        help="List episode indices available in the trace file for the selected env and exit.",
    )
    parser.add_argument(
        "--show_3d",
        action="store_true",
        help="Show only the 3D trajectory plot for EEFs and garment keypoints.",
    )
    args = parser.parse_args()

    trace_file = Path(args.trace_file).expanduser()
    rows = _load_rows(trace_file)

    if args.list_episodes:
        episodes = _available_episodes(rows, env_id=int(args.env_id))
        if episodes:
            print("Available episodes:", ", ".join(str(ep) for ep in episodes))
        else:
            print("No episode_index values found in the trace for env", args.env_id)
        return

    episode_rows, episode_index = _select_episode_rows(rows, env_id=int(args.env_id), episode_arg=str(args.episode))
    if not episode_rows:
        raise ValueError(
            f"No rows found for env_id={args.env_id} and episode={args.episode}. "
            f"Available episodes: {_available_episodes(rows, env_id=int(args.env_id))}"
        )

    if args.show_3d:
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        ax3d = fig.add_subplot(1, 1, 1, projection="3d")
        ax0 = fig.add_subplot(3, 1, 1)
        ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)
        ax2 = fig.add_subplot(3, 1, 3, sharex=ax0)
        ax0.set_visible(False)
        ax1.set_visible(False)
        ax2.set_visible(False)
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
        ax3d = None
    axes = [ax0, ax1, ax2]
    _plot_episode(
        fig,
        axes,
        ax3d,
        trace_file=trace_file,
        rows=episode_rows,
        env_id=int(args.env_id),
        episode_index=episode_index,
    )

    if args.save_path:
        save_path = Path(args.save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    main()
