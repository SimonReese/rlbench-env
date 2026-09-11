"""
This file consider an RLBench dataset to open each specific variationID for each task and play a specific episode/demo
"""

from typing import List, Union

import os
import numpy
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig

from rlbench.task_environment import TaskEnvironment
import rlbench.utils

# Dataset
DATASET_PATH = "/home/peraro/source/play-rlbench/data-generation/datasets/arrange-train"

# Task filter - which task to test
TASK_FILTER = ["arrange_objects"]

# ENV
HEADLESS = False
# CAMERA CONFIGS
CAMERA_IMAGE_SIZE = (224, 224)
# OBSERVATION CONFIGS
OBSERVATION_CONFIGS: ObservationConfig

def rollout_episode(demo: Demo, task: TaskEnvironment, episode_number = None):
    if episode_number is not None: print(f"Playing episode {episode_number}")
    
    description , _local_obs = task.reset_to_demo(demo)
    print(f"Description {description}")
    old_pose = _local_obs.gripper_pose
    # Reproduce each step
    for obs in demo:
        new_pose = obs.gripper_pose
        delta_action = rlbench.utils.delta_pose_ee(old_pose[:3], old_pose[3:], new_pose[:3], new_pose[3:])
        #print(f"Gripper joint positions {pretty_print_array(obs.gripper_joint_positions)}, is open: {obs.gripper_open}")
        #print(f"Open amount{rlbench.utils.get_panda_gripper_open_amount(obs.gripper_joint_positions)[0]} {obs.gripper_open=}")
        #print(f"Joint Intervals{task._scene.robot.gripper.get_joint_intervals()}")
        #print(f"Start: {old_pose}\nTarget: {new_pose}\nDelta: {delta_action}")
        #print(f"Rotation will be {pretty_print_array(quaternion_to_euler(delta_action[3:]))}")
        
        _local_obs, reward, done = task.step(numpy.concatenate((delta_action, [obs.gripper_open])))
        old_pose = _local_obs.gripper_pose
    print(f"Task done")
    print(f"Success: {done}")


def main():
    env = setup_env()
    assert OBSERVATION_CONFIGS is not None
    env.launch()

    tasks_list = os.listdir(DATASET_PATH)
    tasks_list = filter_task(tasks_list, TASK_FILTER)

    # Open all tasks envs
    for task_name in tasks_list:
        print(f"Opening task {task_name}")
        task_class = rlbench.utils.name_to_task_class(task_name)
        task = env.get_task(task_class)

        variations_ids = rlbench.utils.get_variations_ids(DATASET_PATH, task_name)
        # Open each variation
        for variation in variations_ids:
            print("Opening variation ", variation)
            # Get all episodes in variation
            episodes = rlbench.utils.get_stored_demos(amount=-1, image_paths=True, dataset_root=DATASET_PATH, 
                                        variation_number=variation, task_name=task_name, obs_config=OBSERVATION_CONFIGS, random_selection=False)
            task.set_variation(variation)
            #demos = load_demos(dataset, task_name, variation)

            for num, demo in enumerate(episodes):
                rollout_episode(demo, task, num)
            exit()
                
    env.shutdown()

def setup_env() -> Environment:
    global OBSERVATION_CONFIGS
    
    cam_config = CameraConfig(image_size=CAMERA_IMAGE_SIZE)
    OBSERVATION_CONFIGS = ObservationConfig(
        left_shoulder_camera= cam_config,
        right_shoulder_camera= cam_config,
        overhead_camera= cam_config,
        wrist_camera= cam_config,
        front_camera= cam_config,
        gripper_joint_positions=True
    )

    arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE)
    act = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())
    env = Environment(action_mode=act)

    return env

def pretty_print_array(data: numpy.ndarray, printout=False, label="") -> Union[numpy.ndarray , None]:
    pretty = data.clip(min=1e-3)
    if printout:
        print(f"{label}{pretty}")
        return
    return pretty

def filter_task(tasks_list: List[str], keep_tasks = []):
    if len(keep_tasks) == 0: return tasks_list
    
    filtered = []
    for task in tasks_list:
        if task in keep_tasks: filtered.append(task)
    
    return filtered

if __name__ == "__main__":
    main()
