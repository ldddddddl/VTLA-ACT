"""
使用固定Seed采集ManiSkill演示数据

该脚本使用ManiSkill的运动规划器生成演示轨迹，
并确保seed信息被保存到HDF5文件中，以便后续评估时
能够使用相同的初始状态。

用法:
    python scripts/collect_demo_fixed_seed.py --seed 42 --num_episodes 1
"""

import os
import sys
import time
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - Register ManiSkill envs
from mani_skill.utils.wrappers.record import RecordEpisode


def get_tactile_data(env) -> np.ndarray:
    """Get tactile data (contact forces) from gripper links."""
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


def solve_trajectory_with_planner(env, seed: int):
    """
    使用ManiSkill的运动规划器生成演示轨迹
    """
    from mani_skill.examples.motionplanning.panda.solutions import solvePickCube

    # Reset with fixed seed
    obs, info = env.reset(seed=seed)

    # Solve using motion planner
    try:
        result = solvePickCube(env)
        if result["success"]:
            return result["actions"], result["env_states"], True
        else:
            return None, None, False
    except Exception as e:
        print(f"Motion planning failed: {e}")
        return None, None, False


def collect_episode(env, seed: int, camera_names: list, max_steps: int = 200):
    """
    采集单个episode的数据，并保存初始环境状态
    """
    # Reset with fixed seed
    obs, info = env.reset(seed=seed)

    # 保存初始环境状态（包括物体位置）
    init_state = env.unwrapped.get_state_dict()

    # 记录初始状态
    print(f"  初始qpos: {get_qpos_qvel(obs)[0][:4]}...")

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/tactile": [],
        "/action": [],
        "init_state": init_state,  # 保存初始状态
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    try:
        from mani_skill.examples.motionplanning.panda.solutions import solvePickCube

        # 解决问题
        result = solvePickCube(env)

        if not result["success"]:
            print("  运动规划失败")
            return None, False

        actions = result["actions"]

        # 重新reset并执行动作序列
        obs, _ = env.reset(seed=seed)

        total_reward = 0
        for step, action in enumerate(actions):
            # 记录观测
            qpos, qvel = get_qpos_qvel(obs)
            tactile = get_tactile_data(env)
            images = get_camera_images(obs, camera_names)

            data_dict["/observations/qpos"].append(qpos)
            data_dict["/observations/qvel"].append(qvel)
            data_dict["/observations/tactile"].append(tactile)
            data_dict["/action"].append(action.astype(np.float32))

            for cam_name in camera_names:
                if cam_name in images:
                    data_dict[f"/observations/images/{cam_name}"].append(
                        images[cam_name]
                    )

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(reward, "item"):
                reward = reward.item()
            total_reward += float(reward)

            if terminated or truncated:
                break

        # 检查成功
        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()

        print(f"  总奖励: {total_reward:.2f}, 成功: {success}")
        return data_dict, success

    except ImportError as e:
        print(f"  无法导入运动规划器: {e}")
        print("  使用随机策略...")

        # 使用随机策略作为fallback
        total_reward = 0
        for step in range(max_steps):
            qpos, qvel = get_qpos_qvel(obs)
            tactile = get_tactile_data(env)
            images = get_camera_images(obs, camera_names)

            data_dict["/observations/qpos"].append(qpos)
            data_dict["/observations/qvel"].append(qvel)
            data_dict["/observations/tactile"].append(tactile)

            # 简单策略：向前伸展
            action = np.zeros(8, dtype=np.float32)
            data_dict["/action"].append(action)

            for cam_name in camera_names:
                if cam_name in images:
                    data_dict[f"/observations/images/{cam_name}"].append(
                        images[cam_name]
                    )

            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(reward, "item"):
                reward = reward.item()
            total_reward += float(reward)

            if terminated or truncated:
                break

        return data_dict, False


def save_episode(
    data_dict: dict,
    dataset_path: str,
    camera_names: list,
    seed: int,
    tactile_dim: int = 6,
):
    """Save episode data to HDF5 file with seed information."""
    if "/action" not in data_dict or len(data_dict["/action"]) == 0:
        return False

    max_timesteps = len(data_dict["/action"])

    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        root.attrs["tactile_dim"] = tactile_dim
        root.attrs["env_seed"] = seed  # 保存seed用于复现

        # 保存初始环境状态到单独文件
        if "init_state" in data_dict:
            import pickle

            state_path = dataset_path + "_init_state.pkl"
            with open(state_path, "wb") as f:
                pickle.dump(data_dict["init_state"], f)
            root.attrs["has_init_state"] = True
        else:
            root.attrs["has_init_state"] = False

        obs = root.create_group("observations")

        # Save images
        image_group = obs.create_group("images")
        for cam_name in camera_names:
            key = f"/observations/images/{cam_name}"
            if key in data_dict and len(data_dict[key]) > 0:
                images = np.stack(data_dict[key], axis=0)
                h, w, c = images.shape[1:]
                image_group.create_dataset(
                    cam_name, data=images, dtype="uint8", chunks=(1, h, w, c)
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

        # Save action
        action_data = np.stack(data_dict["/action"], axis=0)
        root.create_dataset("action", data=action_data)

    print(f"  保存到: {dataset_path}.hdf5 (seed={seed})")
    return True


def main(args):
    env_id = args["env_id"]
    dataset_dir = args["dataset_dir"]
    num_episodes = args["num_episodes"]
    base_seed = args["seed"]
    camera_names = args["camera_names"]

    os.makedirs(dataset_dir, exist_ok=True)

    print(f"创建环境: {env_id}")
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="cameras",
        sim_backend="cpu",
    )

    print(f"采集 {num_episodes} 个episodes (base_seed={base_seed})")
    print("=" * 50)

    success_count = 0
    for episode_idx in range(num_episodes):
        seed = base_seed + episode_idx
        print(f"\n采集 episode {episode_idx} (seed={seed}):")

        data_dict, success = collect_episode(env, seed, camera_names)

        if data_dict is not None:
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
            save_episode(data_dict, dataset_path, camera_names, seed)
            if success:
                success_count += 1

    env.close()

    print("\n" + "=" * 50)
    print(f"采集完成! 保存 {num_episodes} 个episodes到 {dataset_dir}")
    print(f"成功率: {success_count}/{num_episodes}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ManiSkill demos with fixed seed"
    )
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--dataset_dir", type=str, default="data/maniskill_fixed_seed")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Base seed for episodes")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["base_camera"])

    main(vars(parser.parse_args()))
