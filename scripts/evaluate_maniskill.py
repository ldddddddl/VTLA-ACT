"""
ManiSkill Online Evaluation Script

This script evaluates a trained VTLA-ACT policy in the ManiSkill environment,
with support for tactile observations.

Usage:
    python scripts/evaluate_maniskill.py \
        --ckpt_path checkpoints/policy_best.ckpt \
        --env_id PickCube-v1 \
        --num_episodes 10 \
        --use_tactile \
        --render
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maniskill_env import ManiSkillEnvWrapper
from policy import ACTPolicy, ACTPolicyWithTactile


def load_policy(
    ckpt_path: str,
    policy_config: dict,
    use_tactile: bool = True,
) -> torch.nn.Module:
    """
    Load a trained policy from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        policy_config: Policy configuration dictionary
        use_tactile: Whether to use tactile modality
        
    Returns:
        policy: Loaded policy module
    """
    if use_tactile:
        policy = ACTPolicyWithTactile(policy_config)
    else:
        policy = ACTPolicy(policy_config)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    
    if 'model_state_dict' in checkpoint:
        policy.model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        policy.model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is the state_dict directly
        policy.model.load_state_dict(checkpoint)
    
    policy.model.eval()
    policy.model.cuda()
    
    return policy


def process_observation(
    obs: dict,
    camera_names: list,
    norm_stats: dict,
    use_tactile: bool = True,
) -> tuple:
    """
    Process raw observation into model input format.
    
    Args:
        obs: Raw observation from environment
        camera_names: List of camera names
        norm_stats: Normalization statistics
        use_tactile: Whether to use tactile data
        
    Returns:
        qpos: Normalized joint positions tensor
        image: Normalized image tensor
        tactile: Normalized tactile tensor (or None)
    """
    # Process qpos
    qpos = obs['qpos'].astype(np.float32)
    qpos = (qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
    qpos = torch.from_numpy(qpos).unsqueeze(0).cuda().float()
    
    # Process images
    images = []
    for cam_name in camera_names:
        if cam_name in obs['images']:
            img = obs['images'][cam_name]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            # Convert to CHW format
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            images.append(img)
    
    if images:
        image = np.stack(images, axis=0)  # (num_cam, C, H, W)
        image = torch.from_numpy(image).unsqueeze(0).cuda().float()
    else:
        image = None
    
    # Process tactile
    tactile = None
    if use_tactile and 'tactile' in obs:
        tactile = obs['tactile'].astype(np.float32)
        if 'tactile_mean' in norm_stats and 'tactile_std' in norm_stats:
            tactile = (tactile - norm_stats['tactile_mean']) / norm_stats['tactile_std']
        tactile = torch.from_numpy(tactile).unsqueeze(0).cuda().float()
    
    return qpos, image, tactile


def denormalize_action(action: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Denormalize action using statistics."""
    return action * norm_stats['action_std'] + norm_stats['action_mean']


def temporal_ensemble(
    all_actions: list,
    t: int,
    chunk_size: int,
    k: float = 0.01,
) -> np.ndarray:
    """
    Apply temporal ensembling to smooth actions.
    
    Args:
        all_actions: List of predicted action chunks
        t: Current timestep
        chunk_size: Size of action chunks
        k: Exponential weight factor
        
    Returns:
        action: Smoothed action
    """
    if len(all_actions) == 0:
        return None
    
    # Build weighting for temporal ensemble
    actions_for_current = []
    weights = []
    
    for i, actions in enumerate(all_actions):
        # Check if this chunk contains action for timestep t
        chunk_start = i
        if chunk_start <= t < chunk_start + chunk_size:
            idx = t - chunk_start
            if idx < len(actions):
                actions_for_current.append(actions[idx])
                # Exponential weighting
                weight = np.exp(-k * (len(all_actions) - 1 - i))
                weights.append(weight)
    
    if not actions_for_current:
        return all_actions[-1][0] if all_actions else None
    
    # Weighted average
    actions_for_current = np.stack(actions_for_current, axis=0)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    action = np.sum(actions_for_current * weights[:, np.newaxis], axis=0)
    return action


def evaluate(
    env: ManiSkillEnvWrapper,
    policy: torch.nn.Module,
    norm_stats: dict,
    camera_names: list,
    num_episodes: int = 10,
    max_steps: int = 200,
    use_tactile: bool = True,
    temporal_agg: bool = False,
    chunk_size: int = 100,
    render: bool = False,
) -> dict:
    """
    Evaluate policy in environment.
    
    Args:
        env: ManiSkill environment wrapper
        policy: Trained policy
        norm_stats: Normalization statistics
        camera_names: Camera names
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        use_tactile: Whether to use tactile
        temporal_agg: Whether to use temporal ensembling
        chunk_size: Size of action chunks
        render: Whether to render
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    successes = []
    rewards = []
    
    for episode_idx in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0.0
        success = False
        
        all_actions = []
        
        for step in range(max_steps):
            # Process observation
            qpos, image, tactile = process_observation(
                obs, camera_names, norm_stats, use_tactile
            )
            
            # Get action from policy
            with torch.no_grad():
                if use_tactile:
                    a_hat = policy(qpos, image, tactile=tactile)
                else:
                    a_hat = policy(qpos, image)
            
            # Convert to numpy
            actions = a_hat.squeeze(0).cpu().numpy()
            
            if temporal_agg:
                all_actions.append(actions)
                action = temporal_ensemble(all_actions, step, chunk_size)
                if action is None:
                    action = actions[0]
            else:
                action = actions[0]  # Take first action from chunk
            
            # Denormalize action
            action = denormalize_action(action, norm_stats)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            if info.get('success', False):
                success = True
            
            if terminated or truncated:
                break
        
        successes.append(int(success))
        rewards.append(episode_reward)
        
        print(f"Episode {episode_idx + 1}: Success={success}, Reward={episode_reward:.3f}")
    
    results = {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'num_episodes': num_episodes,
    }
    
    return results


def main(args):
    """Main evaluation function."""
    
    print("=" * 60)
    print("VTLA-ACT ManiSkill Evaluation")
    print("=" * 60)
    
    # Load normalization stats
    norm_stats_path = os.path.join(args['dataset_dir'], 'norm_stats.npz')
    if os.path.exists(norm_stats_path):
        norm_stats = dict(np.load(norm_stats_path))
        print(f"Loaded norm stats from {norm_stats_path}")
    else:
        print("Warning: No norm_stats.npz found, using identity normalization")
        state_dim = args.get('state_dim', 7)
        tactile_dim = args.get('tactile_dim', 6)
        norm_stats = {
            'qpos_mean': np.zeros(state_dim),
            'qpos_std': np.ones(state_dim),
            'action_mean': np.zeros(state_dim),
            'action_std': np.ones(state_dim),
            'tactile_mean': np.zeros(tactile_dim),
            'tactile_std': np.ones(tactile_dim),
        }
    
    # Create environment
    camera_names = args.get('camera_names', ['base_camera'])
    env = ManiSkillEnvWrapper(
        env_id=args['env_id'],
        obs_mode='rgbd',
        control_mode=args.get('control_mode', 'pd_joint_delta_pos'),
        render_mode='human' if args.get('render', False) else 'cameras',
        camera_names=camera_names,
    )
    
    print(f"Environment: {args['env_id']}")
    print(f"Action space: {env.action_space}")
    print(f"Tactile dim: {env.tactile_dim}")
    
    # Build policy config
    policy_config = {
        'lr': 1e-5,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'num_queries': args.get('chunk_size', 100),
        'camera_names': camera_names,
        'enc_layers': 4,
        'dec_layers': 7,
        'dim_feedforward': args.get('dim_feedforward', 3200),
        'hidden_dim': args.get('hidden_dim', 512),
        'nheads': 8,
        'kl_weight': args.get('kl_weight', 10),
        'use_tactile': args.get('use_tactile', True),
        'tactile_dim': args.get('tactile_dim', 6),
        'state_dim': args.get('state_dim', 7),
        'ckpt_dir': os.path.dirname(args['ckpt_path']),
        'policy_class': 'ACT',
        'task_name': args['env_id'],
        'seed': 0,
        'num_epochs': 1,
    }
    
    # Load policy
    print(f"\nLoading policy from {args['ckpt_path']}")
    policy = load_policy(
        args['ckpt_path'],
        policy_config,
        use_tactile=args.get('use_tactile', True),
    )
    print("Policy loaded successfully")
    
    # Evaluate
    results = evaluate(
        env=env,
        policy=policy,
        norm_stats=norm_stats,
        camera_names=camera_names,
        num_episodes=args.get('num_episodes', 10),
        max_steps=args.get('max_steps', 200),
        use_tactile=args.get('use_tactile', True),
        temporal_agg=args.get('temporal_agg', False),
        chunk_size=args.get('chunk_size', 100),
        render=args.get('render', False),
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Episodes: {results['num_episodes']}")
    print("=" * 60)
    
    env.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VTLA-ACT in ManiSkill')
    
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to policy checkpoint')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset (for norm stats)')
    parser.add_argument('--env_id', type=str, default='PickCube-v1',
                        help='ManiSkill environment ID')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--use_tactile', action='store_true',
                        help='Use tactile modality')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Use temporal ensembling')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Action chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dimension')
    parser.add_argument('--state_dim', type=int, default=7,
                        help='State/action dimension')
    parser.add_argument('--tactile_dim', type=int, default=6,
                        help='Tactile dimension')
    parser.add_argument('--camera_names', type=str, nargs='+',
                        default=['base_camera'],
                        help='Camera names')
    parser.add_argument('--control_mode', type=str, 
                        default='pd_joint_delta_pos',
                        help='Control mode')
    
    args = parser.parse_args()
    main(vars(args))
