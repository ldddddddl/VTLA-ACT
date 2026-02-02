"""
开环回放测试：直接使用训练数据中的动作在仿真环境中回放
用于验证问题是模型预测还是环境初始状态

如果回放成功抓取 → 问题在模型
如果回放失败 → 问题在环境初始状态与训练数据不匹配
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import argparse
from sim_env import make_sim_env
from constants import SIM_TASK_CONFIGS


def open_loop_replay(data_path: str, task_name: str, seed: int = 0):
    """
    使用训练数据中的动作序列在仿真环境中进行开环回放
    """
    # 加载训练数据
    print(f"加载数据: {data_path}")
    has_init_state = False
    with h5py.File(data_path, "r") as f:
        actions = np.array(f["action"])
        qpos_data = np.array(f["observations"]["qpos"])
        # 读取保存的seed
        saved_seed = f.attrs.get("env_seed", seed)
        has_init_state = f.attrs.get("has_init_state", False)
        print(f"动作序列形状: {actions.shape}")
        print(f"qpos形状: {qpos_data.shape}")
        print(f"保存的env_seed: {saved_seed}")
        print(f"有初始状态文件: {has_init_state}")

        # 打印初始状态用于对比
        print(f"\n训练数据初始qpos: {qpos_data[0]}")

    # 加载初始环境状态（如果存在）
    init_state = None
    state_path = data_path.replace(".hdf5", "_init_state.pkl")
    if has_init_state and os.path.exists(state_path):
        import pickle

        with open(state_path, "rb") as f:
            init_state = pickle.load(f)
        print(f"已加载初始环境状态: {state_path}")

    # 创建仿真环境
    print(f"\n创建仿真环境: {task_name}")
    env = make_sim_env(task_name)

    # Reset并设置环境状态
    ts = env.reset(seed=int(saved_seed))

    # 如果有保存的初始状态，恢复它
    if init_state is not None:
        env.unwrapped.set_state_dict(init_state)
        obs = env.unwrapped.get_obs()  # 重新获取观测
        ts_obs = {
            "qpos": obs["agent"]["qpos"].cpu().numpy()
            if hasattr(obs["agent"]["qpos"], "cpu")
            else obs["agent"]["qpos"]
        }
        init_qpos = np.array(ts_obs["qpos"])
        if len(init_qpos.shape) > 1:
            init_qpos = init_qpos[0]
        print("已恢复保存的初始环境状态")
    else:
        init_qpos = np.array(ts.observation["qpos"])

    print(f"环境初始qpos: {init_qpos}")

    # 比较初始状态
    qpos_diff = np.abs(init_qpos - qpos_data[0])
    print(f"\n初始qpos差异: {qpos_diff}")
    print(f"初始qpos差异均值: {qpos_diff.mean():.6f}")

    # 执行开环回放
    print("\n===== 开环回放开始 =====")
    rewards = []
    total_reward = 0

    for t, action in enumerate(actions):
        ts = env.step(action)
        reward = float(ts.reward)
        rewards.append(reward)
        total_reward += reward

        if t % 10 == 0:
            env_qpos = np.array(ts.observation["qpos"])
            if t < len(qpos_data) - 1:
                data_qpos = qpos_data[t + 1]  # 下一时间步的qpos
                qpos_error = np.abs(env_qpos - data_qpos).mean()
                print(f"时间步 {t}: reward={reward:.4f}, qpos_error={qpos_error:.6f}")
            else:
                print(f"时间步 {t}: reward={reward:.4f}")

    print(f"\n===== 回放结果 =====")
    print(f"总奖励: {total_reward:.4f}")
    print(f"最高奖励: {max(rewards):.4f}")
    print(f"最终奖励: {rewards[-1]:.4f}")

    # 检查是否成功
    max_reward = env.task.max_reward if hasattr(env, "task") else 1.0
    success = max(rewards) >= max_reward
    print(f"任务成功: {'是 ✓' if success else '否 ✗'}")

    return success, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sim_maniskill_test_overfit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task_config = SIM_TASK_CONFIGS[args.task_name]
    data_path = f"{task_config['dataset_dir']}/episode_0.hdf5"

    success, rewards = open_loop_replay(data_path, args.task_name, args.seed)

    if not success:
        print("\n" + "=" * 50)
        print("诊断结论:")
        print("开环回放失败，说明环境初始状态与训练数据不一致。")
        print("需要确保：")
        print("1. 使用相同的随机种子reset环境")
        print("2. 物体初始位置与训练数据一致")
        print("=" * 50)


if __name__ == "__main__":
    main()
