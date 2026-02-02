"""
ManiSkill Environment Wrapper for VTLA-ACT

This module provides a wrapper around ManiSkill environments to extract:
- Visual observations (camera images)
- Proprioceptive observations (qpos, qvel)
- Tactile observations (contact forces from gripper links)
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, List
import mani_skill.envs  # noqa: F401 - Register ManiSkill envs


class TimeStep:
    def __init__(self, observation, reward, step_type):
        self.observation = observation
        self.reward = reward
        self.step_type = step_type


class ManiSkillEnvWrapper:
    """
    Wrapper for ManiSkill environments that provides unified interface
    for visual, proprioceptive, and tactile observations.
    """

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        obs_mode: str = "rgbd",
        control_mode: str = "pd_joint_pos",
        render_mode: str = "cameras",
        camera_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (480, 640),
        sim_backend: str = "cpu",
    ):
        """
        Initialize the ManiSkill environment wrapper.

        Args:
            env_id: ManiSkill environment ID
            obs_mode: Observation mode ('rgbd', 'pointcloud', etc.)
            control_mode: Control mode for the robot
            render_mode: Render mode
            camera_names: List of camera names to capture images from
            image_size: Size of captured images (H, W)
            sim_backend: Simulation backend ('cpu' or 'gpu')
        """
        self.env_id = env_id
        self.camera_names = camera_names or ["base_camera", "hand_camera"]
        self.image_size = image_size

        # Create ManiSkill environment
        self.env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sim_backend=sim_backend,
        )

        # Get action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Get link names for tactile sensing (gripper links)
        self._tactile_link_names = self._get_tactile_link_names()

        # Task attribute for compatibility
        class DummyTask:
            def __init__(self):
                self.max_reward = 1

        self.task = DummyTask()

    @property
    def unwrapped(self):
        """Return the unwrapped base environment."""
        return self.env.unwrapped

    def _get_tactile_link_names(self) -> List[str]:
        """Get the link names for tactile sensing (gripper fingers)."""
        # Common link names for Panda gripper
        return ["panda_leftfinger", "panda_rightfinger"]

    def _get_tactile_data(self) -> np.ndarray:
        """
        Get tactile data (contact forces) from gripper links.

        Returns:
            tactile_data: Array of shape (num_links * 3,) containing
                         contact forces [fx, fy, fz] for each link
        """
        scene = self.env.unwrapped.scene
        tactile_forces = []

        # Get the robot articulation
        agent = self.env.unwrapped.agent
        robot = agent.robot

        # Get contact forces for each tactile link
        for link_name in self._tactile_link_names:
            try:
                # Find the link by name
                link = None
                for l in robot.get_links():
                    if l.name == link_name:
                        link = l
                        break

                if link is not None:
                    # Get net contact forces on the link
                    force = link.get_net_contact_forces()
                    if len(force.shape) > 1:
                        force = force[0]  # Take first if batched
                    tactile_forces.append(force)
                else:
                    # Link not found, append zeros
                    tactile_forces.append(np.zeros(3))
            except Exception:
                # If error getting forces, append zeros
                tactile_forces.append(np.zeros(3))

        return np.concatenate(tactile_forces)

    def _extract_images(self, obs: Dict) -> Dict[str, np.ndarray]:
        """Extract camera images from observation."""
        images = {}

        if "sensor_data" in obs:
            sensor_data = obs["sensor_data"]
            print(f"Sensor data keys: {sensor_data.keys()}")
            for cam_name in self.camera_names:
                if cam_name in sensor_data:
                    cam_data = sensor_data[cam_name]
                    print(f"Cam data keys for {cam_name}: {cam_data.keys()}")
                    if "rgb" in cam_data:
                        img = cam_data["rgb"]
                        # Convert torch tensor to numpy if needed
                        if hasattr(img, "cpu"):
                            img = img.cpu().numpy()

                        # Squeeze batch dimension if present
                        if len(img.shape) == 4 and img.shape[0] == 1:
                            img = img[0]

                        if isinstance(img, np.ndarray):
                            # Convert to uint8 if needed
                            if img.dtype == np.float32:
                                img = (img * 255).astype(np.uint8)
                            images[cam_name] = img

        # Fallback: try to get from images directly
        if not images and "image" in obs:
            for cam_name in self.camera_names:
                if cam_name in obs["image"]:
                    img = obs["image"][cam_name]["rgb"]
                    if isinstance(img, np.ndarray):
                        if img.dtype == np.float32:
                            img = (img * 255).astype(np.uint8)
                        images[cam_name] = img

        return images

    def _extract_proprio(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract proprioceptive data from observation."""
        qpos = np.zeros(7)  # 7-DOF arm
        qvel = np.zeros(7)

        if "agent" in obs:
            agent_obs = obs["agent"]
            if "qpos" in agent_obs:
                qpos = np.array(agent_obs["qpos"]).flatten()
            if "qvel" in agent_obs:
                qvel = np.array(agent_obs["qvel"]).flatten()

        return qpos, qvel

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Reset the environment.

        Returns:
            obs_dict: Dictionary containing:
                - 'images': Dict of camera images
                - 'qpos': Joint positions
                - 'qvel': Joint velocities
                - 'tactile': Contact forces
        """
        obs, info = self.env.reset(seed=seed)
        processed_obs = self._process_obs(obs)
        return TimeStep(observation=processed_obs, reward=0, step_type=0)  # 0: FIRST

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Step the environment.

        Args:
            action: Action to execute

        Returns:
            obs_dict: Processed observation dictionary
            reward: Reward value
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_obs(obs)
        step_type = 2 if terminated or truncated else 1  # 1: MID, 2: LAST
        return TimeStep(observation=processed_obs, reward=reward, step_type=step_type)

    def _process_obs(self, obs: Dict) -> Dict:
        """Process raw observation into unified format."""
        images = self._extract_images(obs)
        qpos, qvel = self._extract_proprio(obs)
        tactile = self._get_tactile_data()

        return {
            "images": images,
            "qpos": qpos,
            "qvel": qvel,
            "tactile": tactile,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def max_episode_steps(self) -> int:
        """Get maximum episode steps."""
        return self.env.spec.max_episode_steps if self.env.spec else 200

    @property
    def tactile_dim(self) -> int:
        """Get dimension of tactile observation."""
        return len(self._tactile_link_names) * 3  # 3D force per link


class ManiSkillDemoLoader:
    """
    Loader for ManiSkill demonstration trajectories.

    This class handles loading and replaying demonstration trajectories
    from ManiSkill's dataset format.
    """

    def __init__(self, env: ManiSkillEnvWrapper):
        """
        Initialize the demo loader.

        Args:
            env: ManiSkill environment wrapper
        """
        self.env = env

    def load_demo(self, demo_path: str) -> List[Dict]:
        """
        Load a demonstration trajectory from file.

        Args:
            demo_path: Path to demonstration file

        Returns:
            trajectory: List of (obs, action, reward) tuples
        """
        import h5py

        trajectory = []
        with h5py.File(demo_path, "r") as f:
            actions = f["actions"][:]
            # ManiSkill demos typically store actions
            for i in range(len(actions)):
                trajectory.append(
                    {
                        "action": actions[i],
                    }
                )

        return trajectory

    def replay_demo(self, demo_path: str) -> List[Dict]:
        """
        Replay a demonstration and collect observations including tactile.

        Args:
            demo_path: Path to demonstration file

        Returns:
            collected_data: List of dictionaries with full observations
        """
        trajectory = self.load_demo(demo_path)
        collected_data = []

        obs, _ = self.env.reset()
        collected_data.append(
            {
                "observation": obs,
                "action": None,
                "reward": 0.0,
            }
        )

        for step_data in trajectory:
            action = step_data["action"]
            obs, reward, terminated, truncated, info = self.env.step(action)
            collected_data.append(
                {
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                }
            )

            if terminated or truncated:
                break

        return collected_data


def make_maniskill_env(
    env_id: str = "PickCube-v1", camera_names: Optional[List[str]] = None, **kwargs
) -> ManiSkillEnvWrapper:
    """
    Factory function to create a ManiSkill environment wrapper.

    Args:
        env_id: ManiSkill environment ID
        camera_names: List of camera names
        **kwargs: Additional arguments passed to wrapper

    Returns:
        env: ManiSkill environment wrapper
    """
    return ManiSkillEnvWrapper(env_id=env_id, camera_names=camera_names, **kwargs)


if __name__ == "__main__":
    # Test the wrapper
    env = make_maniskill_env(
        env_id="PickCube-v1",
        camera_names=["base_camera"],
    )

    print(f"Action space: {env.action_space}")
    print(f"Tactile dimension: {env.tactile_dim}")

    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"QPos shape: {obs['qpos'].shape}")
    print(f"Tactile shape: {obs['tactile'].shape}")
    print(f"Image cameras: {list(obs['images'].keys())}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step reward: {reward}")
    print(f"Tactile forces: {obs['tactile']}")

    env.close()
    print("Environment test completed!")
