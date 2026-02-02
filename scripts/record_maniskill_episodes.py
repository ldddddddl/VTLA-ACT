"""
ManiSkill Episode Recorder with Tactile Data

This script replays official ManiSkill demonstration trajectories
and records observations including tactile contact forces.

Usage:
    # First download official demos
    python -m mani_skill.utils.download_demo PickCube-v1
    
    # Then run this script to replay and record with tactile
    python scripts/record_maniskill_episodes.py \
        --env_id PickCube-v1 \
        --dataset_dir data/maniskill_pickcube \
        --demo_path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5
"""

import os
import sys
import time
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import gymnasium as gym

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mani_skill.envs  # noqa: F401 - Register ManiSkill envs


def load_hdf5_to_dict(group):
    """Recursively load HDF5 group to dictionary."""
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = load_hdf5_to_dict(item)
        else:
            result[key] = item[:]
    return result


def index_nested_dict(d, idx):
    """Index into all arrays in a nested dict structure."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = index_nested_dict(value, idx)
        elif isinstance(value, np.ndarray):
            result[key] = value[idx]
        else:
            result[key] = value
    return result


def get_tactile_data(env) -> np.ndarray:
    """
    Get tactile data (contact forces) from gripper links.

    Returns:
        tactile_data: Array of shape (6,) containing contact forces
    """
    tactile_forces = []

    try:
        agent = env.unwrapped.agent
        robot = agent.robot

        tactile_link_names = ["panda_leftfinger", "panda_rightfinger"]

        for link_name in tactile_link_names:
            try:
                link = None
                for l in robot.get_links():
                    if l.name == link_name:
                        link = l
                        break

                if link is not None:
                    force = link.get_net_contact_forces()
                    if hasattr(force, "cpu"):
                        force = force.cpu().numpy()
                    elif hasattr(force, "numpy"):
                        force = force.numpy()
                    if len(force.shape) > 1:
                        force = force[0]
                    tactile_forces.append(force[:3].astype(np.float32))
                else:
                    tactile_forces.append(np.zeros(3, dtype=np.float32))
            except Exception:
                tactile_forces.append(np.zeros(3, dtype=np.float32))
    except Exception:
        tactile_forces = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]

    return np.concatenate(tactile_forces)


def get_camera_images(obs: dict, camera_names: list) -> dict:
    """Extract camera images from observation."""
    images = {}

    if obs is None:
        return images

    # Try sensor_data path
    if "sensor_data" in obs:
        for cam_name in camera_names:
            if cam_name in obs["sensor_data"]:
                cam_data = obs["sensor_data"][cam_name]
                if "rgb" in cam_data:
                    img = cam_data["rgb"]
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    elif hasattr(img, "numpy"):
                        img = img.numpy()
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    if len(img.shape) == 4:
                        img = img[0]
                    images[cam_name] = img

    # Try image path
    if not images and "image" in obs:
        for cam_name in camera_names:
            if cam_name in obs["image"]:
                cam_data = obs["image"][cam_name]
                if "rgb" in cam_data:
                    img = cam_data["rgb"]
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    elif hasattr(img, "numpy"):
                        img = img.numpy()
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    if len(img.shape) == 4:
                        img = img[0]
                    images[cam_name] = img

    return images


def get_qpos_qvel(obs: dict) -> tuple:
    """Extract joint positions and velocities."""
    qpos = np.zeros(9, dtype=np.float32)
    qvel = np.zeros(9, dtype=np.float32)

    if obs is not None and "agent" in obs:
        agent_obs = obs["agent"]
        if "qpos" in agent_obs:
            q = agent_obs["qpos"]
            if hasattr(q, "cpu"):
                q = q.cpu().numpy()
            elif hasattr(q, "numpy"):
                q = q.numpy()
            if len(q.shape) > 1:
                q = q[0]
            qpos = q.astype(np.float32)
        if "qvel" in agent_obs:
            q = agent_obs["qvel"]
            if hasattr(q, "cpu"):
                q = q.cpu().numpy()
            elif hasattr(q, "numpy"):
                q = q.numpy()
            if len(q.shape) > 1:
                q = q[0]
            qvel = q.astype(np.float32)

    return qpos, qvel


def replay_trajectory_with_tactile(
    env,
    h5_file: h5py.File,
    traj_id: str,
    camera_names: list,
) -> tuple:
    """
    Replay a single trajectory by setting environment states from demo.

    This method uses the saved states from the demo file instead of
    executing actions through physics simulation, ensuring 100% accurate
    observations and success rate.
    """
    traj = h5_file[traj_id]

    # Get trajectory data
    actions = traj["actions"][:]

    # Load env_states from HDF5 group
    env_states_group = traj["env_states"]
    env_states_dict = load_hdf5_to_dict(env_states_group)

    # Get number of states (T+1 states for T actions)
    num_states = len(actions) + 1

    # Initialize data containers
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/tactile": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    # Reset environment
    env.reset()

    # 保存初始环境状态（从演示数据中给定的第一个状态）
    init_state = index_nested_dict(env_states_dict, 0)
    data_dict["init_state"] = init_state

    total_reward = 0.0

    # Iterate through states and collect observations
    for step_idx in range(len(actions)):
        # Set environment to the recorded state
        state = index_nested_dict(env_states_dict, step_idx)
        env.unwrapped.set_state_dict(state)

        # Get observation from this state
        obs = env.unwrapped.get_obs()

        # Record observation
        qpos, qvel = get_qpos_qvel(obs)
        tactile = get_tactile_data(env)
        images = get_camera_images(obs, camera_names)

        data_dict["/observations/qpos"].append(qpos)
        data_dict["/observations/qvel"].append(qvel)
        data_dict["/observations/tactile"].append(tactile)

        for cam_name in camera_names:
            if cam_name in images:
                data_dict[f"/observations/images/{cam_name}"].append(images[cam_name])

        # Record action
        action = actions[step_idx].astype(np.float32)
        data_dict["/action"].append(action)

        # Compute reward by stepping (optional, for statistics)
        _, reward, _, _, _ = env.step(action)
        if hasattr(reward, "item"):
            reward = reward.item()
        elif hasattr(reward, "numpy"):
            reward = float(reward.numpy())
        total_reward += float(reward)

    # Check success from final state
    final_state = index_nested_dict(env_states_dict, len(actions))
    env.unwrapped.set_state_dict(final_state)
    info = env.unwrapped.get_info()

    success = False
    if isinstance(info, dict) and "success" in info:
        success_val = info["success"]
        if hasattr(success_val, "item"):
            success_val = success_val.item()
        elif hasattr(success_val, "cpu"):
            success_val = success_val.cpu().numpy()
        success = bool(success_val)

    return success, data_dict, total_reward


def save_episode(
    data_dict: dict,
    dataset_path: str,
    camera_names: list,
    tactile_dim: int = 6,
    is_sim: bool = True,
):
    """Save episode data to HDF5 file."""
    if "/action" not in data_dict or len(data_dict["/action"]) == 0:
        return False

    max_timesteps = len(data_dict["/action"])

    # 保存初始环境状态到单独的pickle文件
    if "init_state" in data_dict:
        import pickle

        state_path = dataset_path + "_init_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(data_dict["init_state"], f)

    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = is_sim
        root.attrs["tactile_dim"] = tactile_dim
        root.attrs["has_init_state"] = "init_state" in data_dict

        obs = root.create_group("observations")

        # Save images
        image_group = obs.create_group("images")
        for cam_name in camera_names:
            key = f"/observations/images/{cam_name}"
            if key in data_dict and len(data_dict[key]) > 0:
                images = np.stack(data_dict[key], axis=0)
                h, w, c = images.shape[1:]
                image_group.create_dataset(
                    cam_name,
                    data=images,
                    dtype="uint8",
                    chunks=(1, h, w, c),
                )

        # Save qpos
        if data_dict["/observations/qpos"]:
            qpos_data = np.stack(data_dict["/observations/qpos"], axis=0)
            obs.create_dataset("qpos", data=qpos_data)

        # Save qvel
        if data_dict["/observations/qvel"]:
            qvel_data = np.stack(data_dict["/observations/qvel"], axis=0)
            obs.create_dataset("qvel", data=qvel_data)

        # Save tactile
        if data_dict["/observations/tactile"]:
            tactile_data = np.stack(data_dict["/observations/tactile"], axis=0)
            obs.create_dataset("tactile", data=tactile_data)

        # Save actions
        action_data = np.stack(data_dict["/action"], axis=0)
        root.create_dataset("action", data=action_data)

        root.attrs["num_timesteps"] = max_timesteps

    return True


def main(args):
    """Main function for recording episodes."""

    env_id = args["env_id"]
    dataset_dir = args["dataset_dir"]
    demo_path = args["demo_path"]
    num_episodes = args.get("num_episodes", None)
    control_mode = args.get("control_mode", "pd_joint_delta_pos")
    camera_names = args.get("camera_names", ["base_camera"])

    # Expand user path
    demo_path = os.path.expanduser(demo_path)

    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if demo file exists
    if not os.path.exists(demo_path):
        print(f"Error: Demo file not found: {demo_path}")
        print("\nPlease download the demo first:")
        print(f"  python -m mani_skill.utils.download_demo {env_id}")
        return

    print(f"Replaying demos from: {demo_path}")
    print(f"Saving to: {dataset_dir}")
    print(f"Control mode: {control_mode}")
    print(f"Camera names: {camera_names}")

    # Create environment
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        control_mode=control_mode,
        render_mode="cameras",
    )

    print(f"Action space: {env.action_space}")

    # Open demo file
    h5_file = h5py.File(demo_path, "r")

    # Get all trajectory IDs and sort numerically
    traj_ids = [k for k in h5_file.keys() if k.startswith("traj_")]
    traj_ids = sorted(traj_ids, key=lambda x: int(x.split("_")[1]))

    if num_episodes is not None:
        traj_ids = traj_ids[:num_episodes]

    print(f"Found {len(traj_ids)} trajectories")

    success_count = 0
    saved_count = 0
    total_reward = 0.0

    for episode_idx, traj_id in enumerate(tqdm(traj_ids, desc="Recording episodes")):
        try:
            success, data_dict, episode_reward = replay_trajectory_with_tactile(
                env, h5_file, traj_id, camera_names
            )

            # Save episode
            dataset_path = os.path.join(dataset_dir, f"episode_{saved_count}")
            if save_episode(data_dict, dataset_path, camera_names):
                saved_count += 1
                if success:
                    success_count += 1
                total_reward += episode_reward

        except Exception as e:
            print(f"\nError replaying {traj_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

        if (episode_idx + 1) % 100 == 0:
            print(
                f"\nEpisode {episode_idx + 1}: Saved={saved_count}, Success={success_count}"
            )

    h5_file.close()
    env.close()

    print("\n" + "=" * 50)
    print("Recording complete!")
    print(f"Saved {saved_count} episodes to {dataset_dir}")
    if saved_count > 0:
        print(
            f"Success rate: {success_count}/{saved_count} ({100 * success_count / saved_count:.1f}%)"
        )
        print(f"Average reward: {total_reward / saved_count:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record ManiSkill episodes with tactile data"
    )

    parser.add_argument(
        "--env_id", type=str, default="PickCube-v1", help="ManiSkill environment ID"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory to save recorded episodes",
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        required=True,
        help="Path to ManiSkill demo trajectory.h5 file",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to record (None = all)",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="pd_joint_pos",
        help="Robot control mode (default: pd_joint_pos for official demos)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["base_camera"],
        help="Camera names to record",
    )

    args = parser.parse_args()
    main(vars(args))
