import pathlib
import os

### Task parameters
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIM_TASK_CONFIGS = {
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    "sim_transfer_cube_human": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_human",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    "sim_insertion_scripted": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    "sim_insertion_human": {
        "dataset_dir": DATA_DIR + "/sim_insertion_human",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["top"],
    },
    # ManiSkill tasks with tactile data
    "sim_maniskill_pickcube": {
        "dataset_dir": DATA_DIR + "/maniskill_pickcube_act",
        "num_episodes": 1000,
        "episode_len": 200,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,  # 2 fingers * 3D force
        "state_dim": 9,  # 7-DOF arm + 2 gripper
        "action_dim": 8,  # 7-DOF action + 1 gripper
    },
    "sim_maniskill_pickcube_tactile": {
        "dataset_dir": DATA_DIR + "/maniskill_pickcube_act",
        "num_episodes": 1000,
        "episode_len": 200,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
    },
    # Aliases
    "maniskill_pickcube": {
        "dataset_dir": DATA_DIR + "/maniskill_pickcube_act",
        "num_episodes": 1000,
        "episode_len": 200,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
    },
    "maniskill_pickcube_tactile": {
        "dataset_dir": DATA_DIR + "/maniskill_pickcube_act",
        "num_episodes": 1000,
        "episode_len": 200,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
    },
    "sim_maniskill_test_overfit": {
        "dataset_dir": DATA_DIR + "/maniskill_act_test",
        "num_episodes": 1,
        "episode_len": 50,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
    },
    "sim_maniskill_fixed_seed": {
        "dataset_dir": DATA_DIR + "/maniskill_fixed_seed",
        "num_episodes": 1,
        "episode_len": 74,  # 与官方演示数据匹配
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
    },
    "verify_seed1": {
        "dataset_dir": DATA_DIR + "/verify_seed1_act",
        "num_episodes": 1,
        "episode_len": 100,
        "camera_names": ["base_camera"],
        "tactile_dim": 6,
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

XML_DIR = (
    str(pathlib.Path(__file__).parent.resolve()) + "/assets/"
)  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    + MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    + PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
)

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
    MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    MASTER_GRIPPER_JOINT_NORMALIZE_FN(x)
)

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)

MASTER_POS2JOINT = (
    lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE)
    / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
)
PUPPET_POS2JOINT = (
    lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE)
    / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
)

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
