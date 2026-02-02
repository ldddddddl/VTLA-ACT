"""
VTLA Dataset Visualizer

Visualize episodes from the VTLA dataset, including:
- Camera images
- Tactile contact forces
- Joint positions and actions

Usage:
    python scripts/visualize_vtla_dataset.py \
        --dataset_dir data/maniskill_pickcube \
        --episode_idx 0 \
        --save_video
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_episode(dataset_dir: str, episode_idx: int) -> dict:
    """Load a single episode from the dataset."""
    episode_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")

    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    data = {}
    with h5py.File(episode_path, "r") as f:
        # Load attributes
        data["sim"] = f.attrs.get("sim", True)
        data["tactile_dim"] = f.attrs.get("tactile_dim", 6)
        data["num_timesteps"] = f.attrs.get("num_timesteps", 0)

        # Load observations
        if "observations" in f:
            obs = f["observations"]
            if "qpos" in obs:
                data["qpos"] = obs["qpos"][:]
            if "qvel" in obs:
                data["qvel"] = obs["qvel"][:]
            if "tactile" in obs:
                data["tactile"] = obs["tactile"][:]

            # Load images
            if "images" in obs:
                data["images"] = {}
                for cam_name in obs["images"].keys():
                    data["images"][cam_name] = obs["images"][cam_name][:]

        # Load actions
        if "action" in f:
            data["action"] = f["action"][:]

    return data


def plot_static_summary(data: dict, save_path: str = None):
    """Plot a static summary of the episode."""
    num_timesteps = data.get("num_timesteps", len(data.get("action", [])))

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Plot sample images
    if "images" in data:
        for i, cam_name in enumerate(data["images"].keys()):
            images = data["images"][cam_name]
            # Show first, middle, and last frames
            for j, idx in enumerate([0, len(images) // 2, len(images) - 1]):
                ax = fig.add_subplot(gs[0, j])
                ax.imshow(images[idx])
                ax.set_title(f"{cam_name} (t={idx})")
                ax.axis("off")

    # Plot tactile data
    if "tactile" in data:
        ax = fig.add_subplot(gs[1, :2])
        tactile = data["tactile"]
        t = np.arange(len(tactile))

        # Left finger forces
        ax.plot(t, tactile[:, 0], "r-", label="Left Fx", alpha=0.8)
        ax.plot(t, tactile[:, 1], "g-", label="Left Fy", alpha=0.8)
        ax.plot(t, tactile[:, 2], "b-", label="Left Fz", alpha=0.8)

        # Right finger forces
        ax.plot(t, tactile[:, 3], "r--", label="Right Fx", alpha=0.8)
        ax.plot(t, tactile[:, 4], "g--", label="Right Fy", alpha=0.8)
        ax.plot(t, tactile[:, 5], "b--", label="Right Fz", alpha=0.8)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Contact Force (N)")
        ax.set_title("Tactile Contact Forces")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot joint positions
    if "qpos" in data:
        ax = fig.add_subplot(gs[1, 2:])
        qpos = data["qpos"]
        t = np.arange(len(qpos))

        for i in range(min(qpos.shape[1], 7)):  # Plot first 7 joints
            ax.plot(t, qpos[:, i], label=f"Joint {i + 1}", alpha=0.8)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Joint Position (rad)")
        ax.set_title("Joint Positions")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot actions
    if "action" in data:
        ax = fig.add_subplot(gs[2, :2])
        actions = data["action"]
        t = np.arange(len(actions))

        for i in range(min(actions.shape[1], 7)):  # Plot first 7 action dims
            ax.plot(t, actions[:, i], label=f"Action {i + 1}", alpha=0.8)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action Value")
        ax.set_title("Actions (Joint Commands)")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot gripper action
    if "action" in data and data["action"].shape[1] > 7:
        ax = fig.add_subplot(gs[2, 2:])
        actions = data["action"]
        t = np.arange(len(actions))

        ax.plot(t, actions[:, -1], "k-", linewidth=2, label="Gripper")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Gripper Command")
        ax.set_title("Gripper Action")
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Episode Summary (T={num_timesteps})", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved summary to: {save_path}")

    plt.show()


def create_video(data: dict, save_path: str, fps: int = 10):
    """Create a video visualization of the episode."""
    if "images" not in data or len(data["images"]) == 0:
        print("No images found in episode")
        return

    cam_name = list(data["images"].keys())[0]
    images = data["images"][cam_name]
    num_frames = len(images)

    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Image subplot
    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(images[0])
    ax_img.set_title(cam_name)
    ax_img.axis("off")

    # Tactile subplot
    ax_tactile = fig.add_subplot(gs[0, 1])
    if "tactile" in data:
        tactile = data["tactile"]
        t = np.arange(len(tactile))
        ax_tactile.plot(t, tactile[:, 0], "r-", label="L-Fx", alpha=0.6)
        ax_tactile.plot(t, tactile[:, 1], "g-", label="L-Fy", alpha=0.6)
        ax_tactile.plot(t, tactile[:, 2], "b-", label="L-Fz", alpha=0.6)
        ax_tactile.plot(t, tactile[:, 3], "r--", label="R-Fx", alpha=0.6)
        ax_tactile.plot(t, tactile[:, 4], "g--", label="R-Fy", alpha=0.6)
        ax_tactile.plot(t, tactile[:, 5], "b--", label="R-Fz", alpha=0.6)
        tactile_line = ax_tactile.axvline(x=0, color="black", linewidth=2)
        ax_tactile.set_xlabel("Timestep")
        ax_tactile.set_ylabel("Force (N)")
        ax_tactile.set_title("Tactile Forces")
        ax_tactile.legend(loc="upper right", fontsize=7, ncol=2)
        ax_tactile.grid(True, alpha=0.3)

    # Action subplot
    ax_action = fig.add_subplot(gs[0, 2])
    if "action" in data:
        actions = data["action"]
        t = np.arange(len(actions))
        for i in range(min(actions.shape[1], 4)):
            ax_action.plot(t, actions[:, i], label=f"A{i + 1}", alpha=0.6)
        if actions.shape[1] > 7:
            ax_action.plot(t, actions[:, -1], "k-", label="Grip", linewidth=2)
        action_line = ax_action.axvline(x=0, color="black", linewidth=2)
        ax_action.set_xlabel("Timestep")
        ax_action.set_ylabel("Action")
        ax_action.set_title("Actions")
        ax_action.legend(loc="upper right", fontsize=7)
        ax_action.grid(True, alpha=0.3)

    def update(frame):
        im.set_array(images[frame])
        if "tactile" in data:
            tactile_line.set_xdata([frame, frame])
        if "action" in data:
            action_line.set_xdata([frame, frame])
        fig.suptitle(f"Frame {frame}/{num_frames - 1}", fontsize=12)
        return [im]

    print(f"Creating video with {num_frames} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=False
    )

    anim.save(save_path, writer="pillow", fps=fps)
    print(f"Saved video to: {save_path}")
    plt.close()


def print_dataset_info(dataset_dir: str):
    """Print dataset statistics."""
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".hdf5")])

    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_dir}")
    print(f"Number of episodes: {len(files)}")
    print(f"{'=' * 50}\n")

    if len(files) == 0:
        return

    # Sample first episode for info
    data = load_episode(dataset_dir, 0)

    print("Data shapes:")
    if "qpos" in data:
        print(f"  qpos:    {data['qpos'].shape}")
    if "qvel" in data:
        print(f"  qvel:    {data['qvel'].shape}")
    if "tactile" in data:
        print(f"  tactile: {data['tactile'].shape}")
    if "action" in data:
        print(f"  action:  {data['action'].shape}")
    if "images" in data:
        for cam_name, imgs in data["images"].items():
            print(f"  images/{cam_name}: {imgs.shape}")

    # Compute statistics
    print("\nTactile statistics (across all episodes):")
    all_tactile = []
    for i in tqdm(range(min(len(files), 100)), desc="Loading tactile data"):
        try:
            ep_data = load_episode(dataset_dir, i)
            if "tactile" in ep_data:
                all_tactile.append(ep_data["tactile"])
        except:
            continue

    if all_tactile:
        all_tactile = np.concatenate(all_tactile, axis=0)
        print(f"  Shape: {all_tactile.shape}")
        print(f"  Mean:  {all_tactile.mean(axis=0)}")
        print(f"  Std:   {all_tactile.std(axis=0)}")
        print(f"  Min:   {all_tactile.min(axis=0)}")
        print(f"  Max:   {all_tactile.max(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VTLA dataset")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing episode HDF5 files",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0, help="Episode index to visualize"
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save visualization as video (GIF)"
    )
    parser.add_argument(
        "--save_summary", action="store_true", help="Save static summary image"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for saved files"
    )
    parser.add_argument(
        "--show_info", action="store_true", help="Show dataset statistics"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or args.dataset_dir
    os.makedirs(output_dir, exist_ok=True)

    # Show dataset info
    if args.show_info:
        print_dataset_info(args.dataset_dir)

    # Load episode
    print(f"\nLoading episode {args.episode_idx}...")
    data = load_episode(args.dataset_dir, args.episode_idx)
    print(f"Loaded episode with {data['num_timesteps']} timesteps")

    # Save/show summary
    summary_path = (
        os.path.join(output_dir, f"episode_{args.episode_idx}_summary.png")
        if args.save_summary
        else None
    )
    plot_static_summary(data, save_path=summary_path)

    # Save video
    if args.save_video:
        video_path = os.path.join(output_dir, f"episode_{args.episode_idx}_video.gif")
        create_video(data, video_path)


if __name__ == "__main__":
    main()
