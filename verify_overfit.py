"""
验证过拟合模型：检查模型预测的动作与训练数据中的真实动作是否一致
"""

import torch
import numpy as np
import h5py
import pickle
import os
from policy import ACTPolicy
from einops import rearrange


def main():
    # 配置
    ckpt_dir = "checkpoints/overfit_test"
    data_path = "data/maniskill_act_test/episode_0.hdf5"
    ckpt_name = "policy_best.ckpt"

    # 加载统计数据
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("=== 数据集统计 ===")
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            print(
                f"{key}: shape={value.shape}, mean={np.mean(value):.4f}, std={np.std(value):.4f}"
            )

    # 加载数据
    with h5py.File(data_path, "r") as f:
        qpos = np.array(f["observations"]["qpos"])
        actions = np.array(f["action"])
        images_raw = np.array(f["observations"]["images"]["base_camera"])
        tactile = (
            np.array(f["observations"]["tactile"])
            if "tactile" in f["observations"]
            else None
        )

    print(f"\n=== 数据形状 ===")
    print(f"qpos: {qpos.shape}")
    print(f"actions: {actions.shape}")
    print(f"images: {images_raw.shape}")
    if tactile is not None:
        print(f"tactile: {tactile.shape}")

    # 创建策略配置
    policy_config = {
        "lr": 1e-5,
        "num_queries": 50,  # chunk_size
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": ["base_camera"],
        "state_dim": 9,
        "action_dim": 8,
        "use_tactile": True,
        "tactile_dim": 6,
    }

    # 加载模型
    print(f"\n=== 加载模型 ===")
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(f"加载状态: {loading_status}")
    policy.cuda()
    policy.eval()
    print(f"模型加载自: {ckpt_path}")

    # 预处理函数
    def pre_process_qpos(qpos_raw):
        return (qpos_raw - stats["qpos_mean"]) / stats["qpos_std"]

    def post_process_action(action):
        return action * stats["action_std"] + stats["action_mean"]

    # 归一化动作（用于比较）
    def normalize_action(action):
        return (action - stats["action_mean"]) / stats["action_std"]

    # 验证几个时间步
    print(f"\n=== 逐步验证 ===")
    errors = []

    for t in range(min(10, len(qpos))):  # 检查前10个时间步
        # 准备输入
        qpos_t = pre_process_qpos(qpos[t])
        qpos_tensor = torch.from_numpy(qpos_t).float().cuda().unsqueeze(0)

        # 处理图像
        img = images_raw[t]  # (H, W, C)
        img = rearrange(img, "h w c -> c h w")
        img = img / 255.0
        img_tensor = (
            torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, C, H, W)

        # 处理触觉数据
        if tactile is not None:
            tact = tactile[t]
            tact_tensor = torch.from_numpy(tact).float().cuda().unsqueeze(0)
        else:
            tact_tensor = None

        # 预测 (ACTPolicy不使用tactile参数)
        with torch.inference_mode():
            pred_actions = policy(
                qpos_tensor, img_tensor
            )  # (1, chunk_size, action_dim)

        # 取第一个预测动作
        pred_action_norm = pred_actions[0, 0].cpu().numpy()
        pred_action = post_process_action(pred_action_norm)

        # 真实动作
        true_action = actions[t]
        true_action_norm = normalize_action(true_action)

        # 计算误差
        error = np.mean((pred_action - true_action) ** 2)
        error_norm = np.mean((pred_action_norm - true_action_norm) ** 2)
        errors.append(error)

        print(f"\n时间步 {t}:")
        print(f"  预测动作 (归一化): {pred_action_norm[:4]}...")
        print(f"  真实动作 (归一化): {true_action_norm[:4]}...")
        print(f"  预测动作: {pred_action[:4]}...")
        print(f"  真实动作: {true_action[:4]}...")
        print(f"  MSE: {error:.6f}, MSE(归一化): {error_norm:.6f}")

    print(f"\n=== 总结 ===")
    print(f"平均MSE: {np.mean(errors):.6f}")
    print(f"最大MSE: {np.max(errors):.6f}")
    print(f"最小MSE: {np.min(errors):.6f}")

    # 检查整个序列的预测
    print(f"\n=== 完整序列验证 ===")
    all_errors = []

    for t in range(len(qpos)):
        qpos_t = pre_process_qpos(qpos[t])
        qpos_tensor = torch.from_numpy(qpos_t).float().cuda().unsqueeze(0)

        img = images_raw[t]
        img = rearrange(img, "h w c -> c h w")
        img = img / 255.0
        img_tensor = torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)

        if tactile is not None:
            tact = tactile[t]
            tact_tensor = torch.from_numpy(tact).float().cuda().unsqueeze(0)
        else:
            tact_tensor = None

        with torch.inference_mode():
            pred_actions = policy(qpos_tensor, img_tensor)

        # 计算每个时间步的预测误差
        pred_action = post_process_action(pred_actions[0, 0].cpu().numpy())
        true_action = actions[t]
        error = np.mean((pred_action - true_action) ** 2)
        all_errors.append(error)

    print(f"完整序列平均MSE: {np.mean(all_errors):.6f}")
    print(f"完整序列标准差: {np.std(all_errors):.6f}")

    # 绘制误差曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(all_errors)
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.title("Action Prediction MSE over Time")
    plt.savefig(os.path.join(ckpt_dir, "overfit_verification_mse.png"), dpi=150)
    print(f"误差曲线已保存到: {ckpt_dir}/overfit_verification_mse.png")


if __name__ == "__main__":
    main()
