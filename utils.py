import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, List, Dict, Tuple

import IPython

e = IPython.embed


# Default tactile dimension (2 fingers * 3D force)
DEFAULT_TACTILE_DIM = 6


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self, episode_ids, dataset_dir, camera_names, norm_stats, max_episode_len=None
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_episode_len = max_episode_len
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

        if self.max_episode_len is not None:
            episode_len = self.max_episode_len

        self.is_sim = is_sim
        padded_action = np.zeros(
            (episode_len, original_action_shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(dataset_path):
            continue
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Use concatenate instead of stack to handle variable length episodes
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().astype(np.float32),
        "action_std": action_std.numpy().astype(np.float32),
        "qpos_mean": qpos_mean.numpy().astype(np.float32),
        "qpos_std": qpos_std.numpy().astype(np.float32),
        "example_qpos": all_qpos_data[0].numpy().astype(np.float32),
    }

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    max_episode_len=None,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    if num_episodes == 1:
        train_indices = [0]
        val_indices = [0]
    else:
        train_ratio = 0.8
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, max_episode_len
    )
    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, max_episode_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


### VTLA (Vision-Tactile-Language-Action) Dataset Support ###


class EpisodicDatasetWithTactile(torch.utils.data.Dataset):
    """
    Dataset class that supports loading tactile data along with visual observations.

    This extends the original EpisodicDataset to include tactile modality.
    """

    def __init__(
        self,
        episode_ids: List[int],
        dataset_dir: str,
        camera_names: List[str],
        norm_stats: Dict,
        tactile_dim: int = DEFAULT_TACTILE_DIM,
    ):
        """
        Initialize the dataset.

        Args:
            episode_ids: List of episode indices
            dataset_dir: Path to dataset directory
            camera_names: List of camera names
            norm_stats: Normalization statistics dictionary
            tactile_dim: Dimension of tactile data
        """
        super(EpisodicDatasetWithTactile).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.tactile_dim = tactile_dim
        self.is_sim = None
        self.has_tactile = True
        self.__getitem__(0)  # Initialize self.is_sim and check tactile

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> Tuple:
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs.get("sim", True)
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            # Get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]

            # Get tactile observation
            if "/observations/tactile" in root:
                tactile = root["/observations/tactile"][start_ts]
                self.has_tactile = True
            else:
                # Create dummy tactile data if not available
                tactile = np.zeros(self.tactile_dim, dtype=np.float32)
                self.has_tactile = False

            # Get image observations
            image_dict = dict()
            for cam_name in self.camera_names:
                key = f"/observations/images/{cam_name}"
                if key in root:
                    image_dict[cam_name] = root[key][start_ts]

            # Get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1) :]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim

        # Pad actions
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # Stack camera images
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name in image_dict:
                all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Convert to tensors
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        tactile_data = torch.from_numpy(tactile).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Rearrange image dimensions: (K, H, W, C) -> (K, C, H, W)
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # Normalize
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        # Normalize tactile data if stats available
        if "tactile_mean" in self.norm_stats and "tactile_std" in self.norm_stats:
            tactile_data = (
                tactile_data - self.norm_stats["tactile_mean"]
            ) / self.norm_stats["tactile_std"]

        return image_data, qpos_data, tactile_data, action_data, is_pad


def get_norm_stats_with_tactile(
    dataset_dir: str,
    num_episodes: int,
    tactile_dim: int = DEFAULT_TACTILE_DIM,
) -> Dict:
    """
    Compute normalization statistics including tactile data.

    Args:
        dataset_dir: Path to dataset directory
        num_episodes: Number of episodes
        tactile_dim: Dimension of tactile data

    Returns:
        stats: Dictionary containing normalization statistics
    """
    all_qpos_data = []
    all_action_data = []
    all_tactile_data = []
    has_tactile = False

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(dataset_path):
            continue

        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

            # Load tactile data if available
            if "/observations/tactile" in root:
                tactile = root["/observations/tactile"][()]
                all_tactile_data.append(torch.from_numpy(tactile))
                has_tactile = True

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # Normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # Normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos_data[0, 0].numpy(),
    }

    # Normalize tactile data
    if has_tactile and all_tactile_data:
        all_tactile_data = torch.stack(all_tactile_data)
        tactile_mean = all_tactile_data.mean(dim=[0, 1], keepdim=True)
        tactile_std = all_tactile_data.std(dim=[0, 1], keepdim=True)
        tactile_std = torch.clip(tactile_std, 1e-2, np.inf)
        stats["tactile_mean"] = tactile_mean.numpy().squeeze()
        stats["tactile_std"] = tactile_std.numpy().squeeze()
    else:
        # Default stats for tactile
        stats["tactile_mean"] = np.zeros(tactile_dim)
        stats["tactile_std"] = np.ones(tactile_dim)

    return stats


def load_data_with_tactile(
    dataset_dir: str,
    num_episodes: int,
    camera_names: List[str],
    batch_size_train: int,
    batch_size_val: int,
    tactile_dim: int = DEFAULT_TACTILE_DIM,
) -> Tuple:
    """
    Load dataset with tactile data support.

    Args:
        dataset_dir: Path to dataset directory
        num_episodes: Number of episodes
        camera_names: List of camera names
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        tactile_dim: Dimension of tactile data

    Returns:
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        norm_stats: Normalization statistics
        is_sim: Whether this is simulation data
    """
    print(f"\nData from: {dataset_dir}\n")

    # Train/val split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # Obtain normalization stats including tactile
    norm_stats = get_norm_stats_with_tactile(dataset_dir, num_episodes, tactile_dim)

    # Construct datasets
    train_dataset = EpisodicDatasetWithTactile(
        train_indices, dataset_dir, camera_names, norm_stats, tactile_dim
    )
    val_dataset = EpisodicDatasetWithTactile(
        val_indices, dataset_dir, camera_names, norm_stats, tactile_dim
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
