"""
Convert ManiSkill Data to ACT Format

This script converts data recorded from ManiSkill environments
(with tactile data) into the format expected by ACT training.

Usage:
    python scripts/convert_maniskill_to_act.py \
        --input_dir data/maniskill_pickcube \
        --output_dir data/act_pickcube \
        --num_episodes 50
"""

import os
import sys
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_episode(
    input_path: str,
    output_path: str,
    camera_names: list,
    target_image_size: tuple = (480, 640),
):
    """
    Convert a single episode from ManiSkill format to ACT format.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        camera_names: List of camera names to use
        target_image_size: Target image size (H, W)
    """
    with h5py.File(input_path, "r") as src:
        # Read source data
        qpos = src["observations/qpos"][:]
        qvel = src["observations/qvel"][:]
        action = src["action"][:]

        # Read tactile data if available
        has_tactile = "observations/tactile" in src
        if has_tactile:
            tactile = src["observations/tactile"][:]
        else:
            # Create dummy tactile data
            tactile = np.zeros((len(qpos), 6), dtype=np.float32)

        # Read images
        images = {}
        for cam_name in camera_names:
            key = f"observations/images/{cam_name}"
            if key in src:
                images[cam_name] = src[key][:]

        # Get metadata
        is_sim = src.attrs.get("sim", True)
        tactile_dim = src.attrs.get("tactile_dim", tactile.shape[-1])

        max_timesteps = len(action)

    # Write to ACT format
    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as dst:
        dst.attrs["sim"] = is_sim
        dst.attrs["tactile_dim"] = tactile_dim

        obs = dst.create_group("observations")

        # Save images
        image_group = obs.create_group("images")
        for cam_name, img_data in images.items():
            h, w, c = img_data.shape[1:]
            image_group.create_dataset(
                cam_name,
                data=img_data,
                dtype="uint8",
                chunks=(1, h, w, c),
            )

        # Save qpos
        obs.create_dataset("qpos", data=qpos.astype(np.float64))

        # Save qvel
        obs.create_dataset("qvel", data=qvel.astype(np.float64))

        # Save tactile
        obs.create_dataset("tactile", data=tactile.astype(np.float64))

        # Save action
        dst.create_dataset("action", data=action.astype(np.float64))


def compute_norm_stats(
    dataset_dir: str,
    num_episodes: int,
    has_tactile: bool = True,
) -> dict:
    """
    Compute normalization statistics for the dataset.

    Args:
        dataset_dir: Directory containing the dataset
        num_episodes: Number of episodes
        has_tactile: Whether dataset has tactile data

    Returns:
        stats: Dictionary containing normalization statistics
    """
    all_qpos = []
    all_action = []
    all_tactile = []

    for episode_idx in range(num_episodes):
        episode_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(episode_path):
            continue

        with h5py.File(episode_path, "r") as f:
            qpos = f["observations/qpos"][:]
            action = f["action"][:]
            all_qpos.append(qpos)
            all_action.append(action)

            if has_tactile and "observations/tactile" in f:
                tactile = f["observations/tactile"][:]
                all_tactile.append(tactile)

    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)

    stats = {
        "qpos_mean": all_qpos.mean(axis=0),
        "qpos_std": np.clip(all_qpos.std(axis=0), 1e-2, np.inf),
        "action_mean": all_action.mean(axis=0),
        "action_std": np.clip(all_action.std(axis=0), 1e-2, np.inf),
    }

    if all_tactile:
        all_tactile = np.concatenate(all_tactile, axis=0)
        stats["tactile_mean"] = all_tactile.mean(axis=0)
        stats["tactile_std"] = np.clip(all_tactile.std(axis=0), 1e-2, np.inf)
    else:
        stats["tactile_mean"] = np.zeros(6)
        stats["tactile_std"] = np.ones(6)

    return stats


def save_norm_stats(stats: dict, output_path: str):
    """Save normalization statistics to file."""
    np.savez(
        output_path,
        qpos_mean=stats["qpos_mean"],
        qpos_std=stats["qpos_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        tactile_mean=stats["tactile_mean"],
        tactile_std=stats["tactile_std"],
    )
    print(f"Saved normalization stats to {output_path}")


def main(args):
    """Main conversion function."""
    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    num_episodes = args["num_episodes"]
    camera_names = args.get("camera_names", ["base_camera"])
    file_nums = len(os.listdir(input_dir))
    if num_episodes < file_nums:
        num_episodes = file_nums
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {num_episodes} episodes")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Camera names: {camera_names}")

    # Convert each episode
    converted_count = 0
    for episode_idx in tqdm(range(num_episodes), desc="Converting episodes"):
        input_path = os.path.join(input_dir, f"episode_{episode_idx}.hdf5")
        output_path = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping")
            continue

        convert_episode(input_path, output_path, camera_names)
        converted_count += 1

    print(f"\nConverted {converted_count} episodes")

    # Compute and save normalization statistics
    if converted_count > 0:
        print("\nComputing normalization statistics...")
        stats = compute_norm_stats(output_dir, num_episodes, has_tactile=True)
        save_norm_stats(stats, os.path.join(output_dir, "norm_stats.npz"))

        print("\nStatistics:")
        print(f"  qpos_mean shape: {stats['qpos_mean'].shape}")
        print(f"  action_mean shape: {stats['action_mean'].shape}")
        print(f"  tactile_mean shape: {stats['tactile_mean'].shape}")

    print("\nConversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ManiSkill data to ACT format")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing ManiSkill recordings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for ACT format data",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes to convert"
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["base_camera"],
        help="Camera names to include",
    )

    args = parser.parse_args()
    main(vars(args))
