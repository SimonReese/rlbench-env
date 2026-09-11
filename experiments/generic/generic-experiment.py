"""
This file performs evaluation on RLBench dataset
"""
import pathlib
import time
from typing import List, Union
import imageio
import os
import numpy
import json
from dataclasses import asdict
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig

from rlbench.task_environment import TaskEnvironment
import rlbench.utils

import exp_utility.classes
import exp_utility.database

# Dataset
DATASET_PATH = "/home/peraro/source/play-rlbench/data-generation/datasets/arrange-train"
# Experiment name
EXPERIMENT_NAME = "ground-truth-arrange-train" # TODO: implements sanity check
# Results path
RESULTS_PATH = f"/home/peraro/source/play-rlbench/experiments/results/{EXPERIMENT_NAME}"

# Env Config
HEADLESS = True
# Camera Config
CAMERA_IMAGE_SIZE = (224, 224)
# Observation Config
OBSERVATION_CONFIGS: ObservationConfig

def rollout_episode(demo: Demo, task: TaskEnvironment, episode_number: int) -> exp_utility.classes.EpisodeStats:
    """Performs episode and stores videos and paths"""

    print(f"Playing episode {task.get_name()}:{task._variation_number}:ep{episode_number}")
    
    description , _local_obs = task.reset_to_demo(demo)
    print(f"Description {description}")
    
    # Store videos
    front_camera = []
    wrist_camera = []

    episode_stats = exp_utility.classes.EpisodeStats(task_name=task.get_name(), variation_id=task._variation_number, episode_number=episode_number if episode_number is not None else -1, 
                                             language_instruction=description, steps=None, success=False, fail_reason=None, 
                                             video_path=None, gif_path=None,
                                             experiment_id=EXPERIMENT_NAME, 
                                             inference_time=None, inference_mean_time=None)
    # Store steps number
    episode_stats.steps = 0
    # Store inference time
    episode_stats.inference_time = int(time.time())
    # Reproduce each step
    for obs in demo:
        old_pose = _local_obs.gripper_pose
        new_pose = obs.gripper_pose
        delta_action = rlbench.utils.delta_pose_ee(old_pose[:3], old_pose[3:], new_pose[:3], new_pose[3:])
        
        _local_obs, reward, done = task.step(numpy.concatenate((delta_action, [obs.gripper_open])))
        episode_stats.steps += 1
        
        front_camera.append(_local_obs.front_rgb)
        wrist_camera.append(_local_obs.wrist_rgb)
        
        # Check if task was done
        if done: 
            episode_stats.success = True
            break
    print(f"Episode performed")
    print(f"Success: {done}")
    # Compute inference total time
    episode_stats.inference_time = int(time.time()) - episode_stats.inference_time
    
    exp_utility.database.store_video(front_camera, 
                os.path.join(RESULTS_PATH, task.get_name(), f"variation{task._variation_number}", f"episode{episode_number}"),
                filename="front.mp4")
    exp_utility.database.store_video(wrist_camera, 
                os.path.join(RESULTS_PATH, task.get_name(), f"variation{task._variation_number}", f"episode{episode_number}"),
                filename="wrist.mp4")
    # TODO: use https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.relative_to to store only relative path from root
    front_video_path = os.path.join(RESULTS_PATH, task.get_name(), f"variation{task._variation_number}", f"episode{episode_number}", "front.mp4")
    episode_stats.video_path = exp_utility.database.relative_path(RESULTS_PATH, front_video_path)
    exp_utility.database.store_stats(episode_stats, os.path.join(RESULTS_PATH, task.get_name(), f"variation{task._variation_number}", f"episode{episode_number}"), "episode.json")
    
    return episode_stats

def main():
    env = setup_env()
    cam_config = CameraConfig(image_size=(224, 224))
    obs_config = ObservationConfig(
        left_shoulder_camera= cam_config,
        right_shoulder_camera= cam_config,
        overhead_camera= cam_config,
        wrist_camera= cam_config,
        front_camera= cam_config,
        gripper_joint_positions= True
    )
    env.launch()

    tasks_list = os.listdir(DATASET_PATH)
    # Prepare output
    exp_utility.database.make_folders(RESULTS_PATH)

    # Store dataset stats
    dataset_name = pathlib.Path(DATASET_PATH).parts[-1]
    dataset_stats = exp_utility.classes.DatasetStats(dataset_name, experiment_id=EXPERIMENT_NAME,
                                             task_names=tasks_list,
                                             total_episodes=0,
                                             total_success=0, total_failed=0,
                                             total_success_rate=0,
                                             tasks_succes_rates=[],
                                             tasks_stats=[])
    
    # Open all tasks envs
    for task_name in tasks_list:
        task_class = rlbench.utils.name_to_task_class(task_name)
        task = env.get_task(task_class)

        variations_ids = rlbench.utils.get_variations_ids(DATASET_PATH, task_name)
        
        # Storing task stats
        task_stats = exp_utility.classes.TaskStats(task_name,
                                            total_variations=0,
                                            total_episodes=0,
                                            total_success=0,
                                            total_failed=0,
                                            total_success_rate=0,
                                            experiment_id=EXPERIMENT_NAME,
                                            variation_success_rates=[],
                                            variation_stats=[])
        
        # Open each variation
        for variation in variations_ids:
            # Get all episodes in variation
            episodes = rlbench.utils.get_stored_demos(amount=-1, image_paths=True, dataset_root=DATASET_PATH, 
                                        variation_number=variation, task_name=task_name, obs_config=obs_config, random_selection=False)
            task.set_variation(variation)

            # Storing stats for variation
            variation_stats = exp_utility.classes.VariationStats(task_name, variation,
                                                         experiment_id=EXPERIMENT_NAME,
                                                         episodes_stats=[], total_episodes=0,
                                                         total_success=0, total_failed=0, 
                                                         success_rate=0)

            for num, demo in enumerate(episodes):
                episode_stats = rollout_episode(demo, task, num)
                # Fill variation stats
                variation_stats.total_episodes += 1
                variation_stats.episodes_stats.append(episode_stats)
                if episode_stats.success:
                    variation_stats.total_success += 1
                else:
                    variation_stats.total_failed += 1
                variation_stats.success_rate = variation_stats.total_success / variation_stats.total_episodes
            # All episodes completed
            # Store stats
            exp_utility.database.store_stats(variation_stats, os.path.join(RESULTS_PATH, task.get_name(), f"variation{variation}"), "variation.json")

            # Fill task stats
            task_stats.total_variations += 1
            task_stats.total_episodes += variation_stats.total_episodes
            task_stats.total_success += variation_stats.total_success
            task_stats.total_failed += variation_stats.total_failed
            task_stats.total_success_rate = task_stats.total_success / task_stats.total_episodes
            task_stats.variation_success_rates.append(variation_stats.success_rate)
            task_stats.variation_stats.append(variation_stats)
        # All variations completed
        # Store task stats
        exp_utility.database.store_stats(task_stats, os.path.join(RESULTS_PATH, task.get_name()), "task.json")
        
        # Fill dataset stats
        dataset_stats.total_episodes += task_stats.total_episodes
        dataset_stats.total_success += task_stats.total_success
        dataset_stats.total_failed += task_stats.total_failed
        dataset_stats.total_success_rate = dataset_stats.total_success / dataset_stats.total_episodes
        dataset_stats.tasks_succes_rates.append(task_stats.total_success_rate)
        dataset_stats.tasks_stats.append(task_stats)
    # All tasks completed
    # Store dataset stats
    exp_utility.database.store_stats(dataset_stats, RESULTS_PATH, "dataset.json")
                
    env.shutdown()

def setup_env() -> Environment:
    
    cam_config = CameraConfig(image_size=CAMERA_IMAGE_SIZE)
    obs_config = ObservationConfig(
        left_shoulder_camera= cam_config,
        right_shoulder_camera= cam_config,
        overhead_camera= cam_config,
        wrist_camera= cam_config,
        front_camera= cam_config,
        gripper_joint_positions=True
    )

    arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE)
    act = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())
    env = Environment(action_mode=act, headless=HEADLESS, obs_config=obs_config)

    return env

if __name__ == "__main__":
    main()
