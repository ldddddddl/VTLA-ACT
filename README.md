# VTLA-ACT: Vision-Tactile-Language ACT

本项目基于 [ACT (Action Chunking with Transformers)](https://github.com/tonyzhaozh/act) 框架，扩展支持 **触觉模态** 和 **ManiSkill 仿真环境**。

## 🚀 快速开始

### 环境安装

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 conda
conda env create -f conda_env.yaml
conda activate aloha
```

### ManiSkill 数据采集

#### 1. 下载官方演示数据

```bash
uv run python -m mani_skill.utils.download_demo PickCube-v1
```

演示数据将保存到 `~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5`

#### 2. 采集带触觉数据的 VTLA 数据集

```bash
uv run scripts/record_maniskill_episodes.py \
    --env_id PickCube-v1 \
    --dataset_dir data/maniskill_pickcube \
    --demo_path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env_id` | `PickCube-v1` | ManiSkill 环境 ID |
| `--dataset_dir` | 必需 | 数据保存目录 |
| `--demo_path` | 必需 | 官方 demo 轨迹文件路径 |
| `--num_episodes` | `None` | 采集的轨迹数量（默认全部） |
| `--control_mode` | `pd_joint_pos` | 控制模式（需与 demo 匹配） |
| `--camera_names` | `['base_camera']` | 需要采集的相机名称 |

#### 3. 转换为 ACT 训练格式

```bash
uv run scripts/convert_maniskill_to_act.py \
    --input_dir data/maniskill_pickcube \
    --output_dir data/maniskill_pickcube_act
```

### 训练触觉增强 ACT 模型

```bash
uv run imitate_episodes.py \
    --task_name maniskill_pickcube_tactile \
    --ckpt_dir checkpoints/pickcube_tactile \
    --policy_class ACT \
    --batch_size 8 \
    --num_epochs 2000
```

### ManiSkill 在线验证

```bash
uv run scripts/evaluate_maniskill.py \
    --env_id PickCube-v1 \
    --ckpt_path checkpoints/pickcube_tactile/policy_best.ckpt \
    --num_episodes 50 \
    --use_tactile
```

## 📁 数据格式

采集的数据以 HDF5 格式保存，每个 episode 一个文件：

```
episode_X.hdf5
├── observations/
│   ├── qpos          (T, 9)     # 关节位置
│   ├── qvel          (T, 9)     # 关节速度
│   ├── tactile       (T, 6)     # 触觉力向量 [左手指xyz, 右手指xyz]
│   └── images/
│       └── base_camera  (T, H, W, 3)  # RGB 图像
└── action            (T, 8)     # 动作
```
**注意**：如果是通过 `record_maniskill_episodes.py` 采集的数据，同目录下还会生成 `episode_X_init_state.pkl` 文件，包含完整的初始环境状态字典。

## ⚠️ 注意事项

### ManiSkill 数据采集

1. **环境状态保存**：为了确保评估时的可复现性，脚本会自动将每个episode的初始环境状态保存为 `_init_state.pkl` 文件。评估时 `imitate_episodes.py` 会尝试加载此文件。

2. **控制模式匹配**：官方 demo 通常使用 `pd_joint_pos`（关节绝对位置控制），请确保 `--control_mode` 参数与 demo 一致。

3. **触觉数据**：触觉数据通过 `link.get_net_contact_forces()` 获取夹爪指尖的接触力，为 6 维向量（左右手指各 3D 力）。

4. **状态 Replay 方式**：脚本使用 demo 中保存的环境状态进行 replay（而非物理仿真），确保 100% 成功率和准确的观测数据。

5. **GPU 要求**：ManiSkill 运行时会使用 GPU，返回的观测数据为 CUDA tensor，脚本已自动处理转换。

### 支持的 ManiSkill 环境

- `PickCube-v1` - 抓取立方体
- 其他环境请参考 [ManiSkill 文档](https://maniskill.readthedocs.io/)

## 📦 项目结构

```
VTLA-ACT/
├── scripts/
│   ├── record_maniskill_episodes.py  # ManiSkill 数据采集
│   ├── convert_maniskill_to_act.py   # 数据格式转换
│   └── evaluate_maniskill.py         # 在线验证
├── detr/
│   └── models/
│       ├── tactile_encoder.py        # 触觉编码器
│       └── detr_vae.py               # 支持触觉的 DETR-VAE
├── maniskill_env.py                  # ManiSkill 环境包装器
├── policy.py                         # ACT 策略（含触觉支持）
├── utils.py                          # 数据加载工具
└── constants.py                      # 任务配置
```

## 📄 License

MIT License
