#!/usr/bin/env python3
"""
Policy Evaluation Script for Real Robots

This script evaluates trained policies on real robots by:
1. Loading a trained policy checkpoint
2. Running inference on the robot (follower arm only)
3. Recording evaluation episodes 
4. Calculating success metrics
5. Visualizing results

Note: For evaluation, only follower arms are used since the policy generates actions directly.
Leader arms are automatically disabled to avoid unnecessary connections.

Usage:
    # Evaluate policy on SO101 robot
    sudo chmod 666 /dev/ttyACM0
    python eval_policy.py \
        --robot_type=so101 \
        --policy_path=outputs/train/so101_tweezer_act/checkpoints/060000/pretrained_model \
        --num_episodes=10 \
        --max_episode_steps=900 \
        --task_description="Grip a tweezer and put it in the box." \
        --record_episodes
"""

import argparse
import logging
import os
import sys
import select
import termios
import tty
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_logging


@dataclass
class EvalConfig:
    """Configuration for policy evaluation."""
    # Policy settings
    policy_path: str = ""  # Path to policy checkpoint or HF hub model
    policy_device: str = "cuda"  # Device to run policy on
    
    # Evaluation settings
    num_episodes: int = 10  # Number of episodes to evaluate
    max_episode_steps: int = 300  # Maximum steps per episode
    fps: int = 30  # Control frequency
    
    # Recording settings
    record_episodes: bool = True  # Whether to record evaluation episodes
    record_videos: bool = True  # Whether to save videos
    output_dir: str = "outputs/eval"  # Output directory for results
    repo_id: Optional[str] = None  # HF repo to save evaluation dataset
    
    # Task settings
    task_description: str = "Complete the task successfully"
    success_threshold: float = 0.9  # Success rate threshold
    
    # Robot reset settings
    reset_to_initial_position: bool = True  # Whether to reset robot position before each episode
    initial_position: Optional[List[float]] = None  # Initial joint positions (if None, uses current position)
    reset_timeout: float = 1.0  # Timeout for reset movement in seconds
    reset_pause: float = 0.1  # Pause after reset before starting episode
    
    # Visualization
    display_images: bool = True  # Display camera feeds during eval
    save_images: bool = False  # Save individual frames
    
    # Metrics
    calculate_metrics: bool = True  # Calculate success metrics
    manual_success_rating: bool = True  # Manual success annotation
    policy_type: str = "act"  # Type of policy to load (default: act)


def get_robot_config(robot_type: str, disable_cameras: bool = False):
    """Get robot configuration based on robot type for policy evaluation.
    
    For evaluation, we only need follower arms since the policy generates actions directly.
    Leader arms are excluded to avoid unnecessary connections.
    
    Args:
        robot_type: Type of robot (only so101 supported)
        disable_cameras: If True, disable all cameras to avoid connection issues
    """
    if robot_type == "so101":
        config = So101RobotConfig()
        # For evaluation, disable leader arms to avoid connection
        config.leader_arms = {}
        if disable_cameras:
            config.cameras = {}
        return config
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. Only 'so101' is supported")


class PolicyEvaluator:
    """Policy evaluator for real robots."""
    
    def __init__(self, robot_config, eval_config: EvalConfig):
        self.robot_config = robot_config
        self.eval_config = eval_config
        
        # Initialize robot
        self.robot = make_robot_from_config(robot_config)
        
        # Initialize policy
        self.policy = None
        self.policy_device = eval_config.policy_device
        
        # Evaluation state
        self.episode_results = []
        self.current_episode = 0
        
        # Setup output directory
        self.output_dir = Path(eval_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_policy(self):
        """Load policy from local checkpoint only, using explicit type."""
        self.logger.info(f"Loading policy from: {self.eval_config.policy_path}")
        policy_path = Path(self.eval_config.policy_path)
        policy_type = self.eval_config.policy_type
        try:
            if policy_type == "act":
                from lerobot.common.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(policy_path)
            elif policy_type == "diffusion":
                from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
                self.policy = DiffusionPolicy.from_pretrained(policy_path)
            elif policy_type == "pi0":
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
                self.policy = PI0Policy.from_pretrained(policy_path)
            elif policy_type == "vqbet":
                from lerobot.common.policies.vqbet.modeling_vqbet import VQBETPolicy
                self.policy = VQBETPolicy.from_pretrained(policy_path)
            elif policy_type == "tdmpc":
                from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
                self.policy = TDMPCPolicy.from_pretrained(policy_path)
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
            self.logger.info(f"Loaded {policy_type} policy")
        except Exception as e:
            raise RuntimeError(f"Failed to load {policy_type} policy: {e}")
        
        if self.policy is None:
            raise ValueError(f"Could not load policy from {self.eval_config.policy_path}")
        
        self.policy.to(self.policy_device)
        self.policy.eval()
        self.logger.info("Policy loaded successfully")

    def move_to_initial_position(self):
        """Move robot to the initial position."""
        if not self.eval_config.reset_to_initial_position:
            return
        
        # Hard-coded SO101 initial position
        target_position = [-8.261719, 194.9414, 186.15234, 60.996094, 80.5957, -7.575758]
        
        # Check if robot is already at initial position
        try:
            observation = self.robot.capture_observation()
            
            # Extract current joint positions
            current_position = []
            for key, value in observation.items():
                if "joint" in key.lower() or "arm" in key.lower():
                    if isinstance(value, torch.Tensor):
                        current_position.extend(value.cpu().numpy().tolist())
                    else:
                        current_position.extend(value.tolist() if hasattr(value, 'tolist') else [value])
            
            # Check if already at target position (within tolerance)
            if current_position and len(current_position) == len(target_position):
                position_diff = [abs(current - target) for current, target in zip(current_position, target_position)]
                max_diff = max(position_diff)
                
                if max_diff < 5.0:  # Tolerance of 5 units
                    self.logger.info("Robot already at initial position, skipping movement")
                    return
                else:
                    self.logger.info(f"Robot position differs by {max_diff:.1f}, moving to initial position...")
            else:
                self.logger.info("Could not determine current position, proceeding with movement...")
                
        except Exception as e:
            self.logger.warning(f"Could not check current position: {e}, proceeding with movement...")
        
        self.logger.info("Moving robot to initial position...")
        
        # Create action to move to initial position
        initial_action = torch.tensor(target_position, dtype=torch.float32)
        
        # Send the initial position as an action
        try:
            self.robot.send_action(initial_action)
            
            # Wait for movement to complete
            self.logger.info(f"Waiting {self.eval_config.reset_timeout}s for robot to reach initial position...")
            time.sleep(self.eval_config.reset_timeout)
            
            # Additional pause before starting episode
            if self.eval_config.reset_pause > 0:
                self.logger.info(f"Pausing {self.eval_config.reset_pause}s before starting episode...")
                time.sleep(self.eval_config.reset_pause)
                
            self.logger.info("Robot moved to initial position")
            
        except Exception as e:
            self.logger.error(f"Failed to move to initial position: {e}")
            # Continue anyway but warn user
            self.logger.warning("Continuing without initial position reset")

    @contextmanager
    def robot_context(self):
        """Context manager for robot connection."""
        try:
            self.logger.info("Connecting to robot...")
            self.robot.connect()
            self.logger.info("Robot connected successfully")
            yield
        finally:
            self.logger.info("Disconnecting robot...")
            self.robot.disconnect()

    def preprocess_observation(self, observation):
        """Preprocess observation for policy input."""
        processed_obs = {}
        
        for key, value in observation.items():
            if "image" in key:
                # Convert to float32 and normalize to [0, 1]
                if isinstance(value, torch.Tensor):
                    processed_value = value.float() / 255.0
                else:
                    processed_value = torch.from_numpy(value).float() / 255.0
                
                # Ensure channel-first format (C, H, W)
                if processed_value.ndim == 3 and processed_value.shape[-1] in [1, 3, 4]:
                    processed_value = processed_value.permute(2, 0, 1)
                
                # Add batch dimension
                processed_value = processed_value.unsqueeze(0)
                processed_obs[key] = processed_value.to(self.policy_device)
                
            else:
                # Handle state observations
                if isinstance(value, torch.Tensor):
                    processed_value = value.float()
                else:
                    processed_value = torch.from_numpy(value).float()
                
                # Add batch dimension if needed
                if processed_value.ndim == 1:
                    processed_value = processed_value.unsqueeze(0)
                
                processed_obs[key] = processed_value.to(self.policy_device)
        
        return processed_obs

    def run_episode(self, episode_idx: int) -> Dict:
        """Run a single evaluation episode."""
        self.logger.info(f"Starting episode {episode_idx + 1}/{self.eval_config.num_episodes}")
        
        # Move robot to initial position before starting episode
        self.move_to_initial_position()
        
        episode_data = {
            "episode_idx": episode_idx,
            "observations": [],
            "actions": [],
            "timestamps": [],
            "success": False,
            "num_steps": 0,
            "episode_reward": 0.0,
            "early_stopped": False,
            "early_stop_reason": "",
        }
        
        # Reset policy if it has a reset method
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        
        # Setup video recording
        video_writers = {}
        if self.eval_config.record_episodes and self.eval_config.record_videos:
            episode_dir = self.output_dir / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for step in tqdm(range(self.eval_config.max_episode_steps), 
                        desc=f"Episode {episode_idx + 1}"):
            step_start_time = time.perf_counter()
            
            # Check for manual early stop (non-blocking)
            if self._check_manual_early_stop():
                episode_data["early_stopped"] = True
                episode_data["early_stop_reason"] = "manual_stop"
                self.logger.info("Manual early stop requested")
                break
            
            # Capture observation
            observation = self.robot.capture_observation()
            episode_data["observations"].append(observation)
            episode_data["timestamps"].append(time.time() - start_time)
            
            # Record video frames
            if self.eval_config.record_episodes and self.eval_config.record_videos:
                self._record_video_frame(observation, video_writers, episode_dir, step)
            
            # Preprocess for policy
            processed_obs = self.preprocess_observation(observation)
            
            # Get action from policy
            with torch.no_grad():
                action = self.policy.select_action(processed_obs)
            
            # Move action to CPU and remove batch dimension
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu()
            
            episode_data["actions"].append(action)
            
            # Send action to robot
            self.robot.send_action(action)
            
            # Display images if requested
            if self.eval_config.display_images:
                self._display_observation(observation)
            
            # Control loop timing
            dt_s = time.perf_counter() - step_start_time
            busy_wait(1 / self.eval_config.fps - dt_s)
            
            episode_data["num_steps"] = step + 1
        
        # Close video writers
        if self.eval_config.record_episodes and self.eval_config.record_videos:
            self._close_video_writers(video_writers)
            self.logger.info(f"Video saved to {episode_dir}")
        
        # Evaluate success
        episode_data["success"] = self._get_manual_success_rating()
        
        episode_data["duration"] = time.time() - start_time
        
        early_stop_info = f", Early Stop: {episode_data['early_stop_reason']}" if episode_data["early_stopped"] else ""
        self.logger.info(f"Episode {episode_idx + 1} completed: "
                        f"Success={episode_data['success']}, "
                        f"Steps={episode_data['num_steps']}, "
                        f"Duration={episode_data['duration']:.2f}s{early_stop_info}")
        
        # Move robot back to initial position after episode ends
        self.move_to_initial_position()
        
        return episode_data

    def _record_video_frame(self, observation, video_writers, episode_dir, step):
        """Record a single frame for each camera to video files."""
        for key, value in observation.items():
            if "image" in key:
                # Convert to numpy if needed
                if isinstance(value, torch.Tensor):
                    frame = value.cpu().numpy()
                else:
                    frame = value
                
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Handle different formats (C,H,W) -> (H,W,C)
                if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
                    frame = frame.transpose(1, 2, 0)
                
                # Convert RGB to BGR for OpenCV
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Initialize video writer for this camera if not exists
                if key not in video_writers:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_path = episode_dir / f"{key.replace('.', '_')}.mp4"
                    video_writers[key] = cv2.VideoWriter(
                        str(video_path), fourcc, self.eval_config.fps, (width, height)
                    )
                
                # Write frame
                video_writers[key].write(frame)

    def _close_video_writers(self, video_writers):
        """Close all video writers and release resources."""
        for writer in video_writers.values():
            writer.release()

    def _check_manual_early_stop(self) -> bool:
        """Check for manual early stop keypress (non-blocking)."""
        try:
            # Check if running in a terminal
            if not sys.stdin.isatty():
                return False
            
            # Simple non-blocking check without changing terminal mode
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                # Read available input
                input_data = sys.stdin.read(1)
                
                if input_data.lower() == 'q':
                    return True
                    
        except Exception as e:
            # If anything goes wrong with keyboard input, just continue
            pass
        return False

    def _display_observation(self, observation):
        """Display camera feeds."""
        for key, value in observation.items():
            if "image" in key:
                if isinstance(value, torch.Tensor):
                    img = value.numpy()
                else:
                    img = value
                
                # Convert to displayable format
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                # Handle different formats
                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    img = img.transpose(1, 2, 0)
                
                # Display
                cv2.imshow(f"Camera: {key}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def _get_manual_success_rating(self) -> bool:
        """Get manual success rating from user."""
        while True:
            response = input("Was this episode successful? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def run_evaluation(self):
        """Run full evaluation."""
        self.logger.info(f"Starting evaluation with {self.eval_config.num_episodes} episodes")
        
        # Load policy
        self.load_policy()
        
        # Run episodes
        with self.robot_context():
            for episode_idx in range(self.eval_config.num_episodes):
                episode_result = self.run_episode(episode_idx)
                self.episode_results.append(episode_result)
                
                # Optional: pause between episodes for manual reset
                if episode_idx < self.eval_config.num_episodes - 1:
                    input(f"Episode {episode_idx + 1} completed. Press Enter to continue to next episode...")
        
        # Calculate and save results
        self.calculate_metrics()
        self.save_results()

    def calculate_metrics(self):
        """Calculate evaluation metrics."""
        if not self.eval_config.calculate_metrics:
            return
            
        num_episodes = len(self.episode_results)
        successful_episodes = sum(1 for ep in self.episode_results if ep["success"])
        
        success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0.0
        avg_steps = np.mean([ep["num_steps"] for ep in self.episode_results])
        avg_duration = np.mean([ep["duration"] for ep in self.episode_results])
        
        metrics = {
            "num_episodes": num_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": success_rate,
            "avg_steps_per_episode": avg_steps,
            "avg_duration_per_episode": avg_duration,
            "task_description": self.eval_config.task_description,
            "policy_path": self.eval_config.policy_path,
        }
        
        self.logger.info("=== EVALUATION RESULTS ===")
        self.logger.info(f"Success Rate: {success_rate:.2%} ({successful_episodes}/{num_episodes})")
        self.logger.info(f"Average Steps: {avg_steps:.1f}")
        self.logger.info(f"Average Duration: {avg_duration:.2f}s")
        
        if success_rate >= self.eval_config.success_threshold:
            self.logger.info("Evaluation PASSED - Success rate above threshold")
        else:
            self.logger.info("Evaluation FAILED - Success rate below threshold")
        
        self.metrics = metrics

    def save_results(self):
        """Save evaluation results."""
        # Save metrics
        import json
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save detailed results
        results_path = self.output_dir / "episode_results.json" 
        with open(results_path, 'w') as f:
            # Convert torch tensors to lists for JSON serialization
            serializable_results = []
            for ep in self.episode_results:
                ep_copy = ep.copy()
                ep_copy["actions"] = [action.tolist() if isinstance(action, torch.Tensor) 
                                    else action for action in ep["actions"]]
                # Remove observations to save space (they can be large)
                ep_copy.pop("observations", None)
                serializable_results.append(ep_copy)
            
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Policy Evaluation for Real Robots")
    
    # Robot configuration
    parser.add_argument("--robot_type", type=str, required=True,
                       choices=["so101"],
                       help="Type of robot to use (only so101 supported)")
    
    # Policy settings
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to policy checkpoint or HuggingFace model")
    parser.add_argument("--policy_device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to run policy on")
    parser.add_argument("--policy_type", type=str, default="act",
                       choices=["act", "diffusion", "pi0", "vqbet", "tdmpc"],
                       help="Type of policy to load (default: act)")
    
    # Evaluation settings
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--max_episode_steps", type=int, default=300,
                       help="Maximum steps per episode")
    parser.add_argument("--fps", type=int, default=30,
                       help="Control frequency")
    
    # Task settings
    parser.add_argument("--task_description", type=str, 
                       default="Complete the task successfully",
                       help="Description of the task being evaluated")
    parser.add_argument("--success_threshold", type=float, default=0.9,
                       help="Success rate threshold for evaluation")
    
    # Recording and output
    parser.add_argument("--record_episodes", action="store_true",
                       help="Whether to record evaluation episodes")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                       help="Output directory for results")
    parser.add_argument("--repo_id", type=str,
                       help="HuggingFace repo ID for saving dataset")
    
    # Visualization and safety
    parser.add_argument("--display_images", action="store_true",
                       help="Display camera feeds during evaluation")
    parser.add_argument("--disable_cameras", action="store_true",
                       help="Disable all cameras to avoid connection issues")
    
    # Robot reset settings
    parser.add_argument("--reset_to_initial_position", action="store_true", default=True,
                       help="Reset robot to initial position before each episode (default: True)")
    parser.add_argument("--no_reset", action="store_true",
                       help="Disable initial position reset")
    parser.add_argument("--reset_timeout", type=float, default=1.0,
                       help="Timeout for reset movement in seconds")
    parser.add_argument("--reset_pause", type=float, default=0.1,
                       help="Pause after reset before starting episode")
    
    args = parser.parse_args()
    
    # Handle reset logic
    if args.no_reset:
        reset_to_initial_position = False
    else:
        reset_to_initial_position = args.reset_to_initial_position
    
    # Initialize logging
    init_logging()
    
    # Create robot configuration
    robot_config = get_robot_config(args.robot_type, disable_cameras=args.disable_cameras)
    
    # Log robot configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Robot type: {args.robot_type}")
    logger.info(f"Leader arms: {list(robot_config.leader_arms.keys()) if robot_config.leader_arms else 'None (disabled for evaluation)'}")
    logger.info(f"Follower arms: {list(robot_config.follower_arms.keys())}")
    logger.info(f"Cameras: {list(robot_config.cameras.keys()) if robot_config.cameras else 'None (disabled)'}")
    logger.info("Manual early stopping: Press 'q' during episode to stop early")
    
    # Create evaluation configuration
    eval_config = EvalConfig(
        policy_path=args.policy_path,
        policy_device=args.policy_device,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        fps=args.fps,
        task_description=args.task_description,
        success_threshold=args.success_threshold,
        record_episodes=args.record_episodes,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        display_images=args.display_images,
        manual_success_rating=True,
        policy_type=args.policy_type,
        reset_to_initial_position=reset_to_initial_position,
        reset_timeout=args.reset_timeout,
        reset_pause=args.reset_pause,
    )
    
    # Create evaluator and run evaluation
    evaluator = PolicyEvaluator(robot_config, eval_config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
