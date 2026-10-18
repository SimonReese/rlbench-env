"""
This file consider an RLBench dataset to open each specific variationID for each task and play a specific episode/demo
"""

import pickle
from typing import List, Union

import os
import numpy
from rlbench import dataset_generator
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig

from rlbench.task_environment import TaskEnvironment
import rlbench.utils

from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from scipy.spatial.transform import Rotation

# Dataset
DATASET_PATH = "datasets/arrange-train"

# Updated Dataset Path
UPDATED_DATASET_PATH = "datasets/arrange-train-sides"
RESUME = False
OVERWRITE = True       # Set to true to remove overwrite new dataset
CAMERA_POSE_FILE = "front_camera_pose.pkl" # numpy.ndarray

# Task filter - which task to test
TASK_FILTER = [] #["arrange_objects"]
# STATIC FILTER OF TASKS wrt camera
CAMERA_VARIATION_FILTER = True

# ENV
HEADLESS = True
# CAMERA CONFIGS
CAMERA_IMAGE_SIZE = (224, 224)
# OBSERVATION CONFIGS
OBSERVATION_CONFIGS: ObservationConfig
CAMERA_MATRIX: numpy.ndarray

def rollout_episode(demo: Demo, task: TaskEnvironment, episode_number):
    if episode_number is not None: print(f"Playing episode {episode_number}")
    camera_pose = randomize_camera()
    # Construct new list of obs
    random_camera_observation: List[Observation] = []
    
    description , _local_obs = task.reset_to_demo(demo)
    print(f"Description {description}")
    old_pose = _local_obs.gripper_pose

    # Store the first observation
    random_camera_observation.append(_local_obs)
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
        random_camera_observation.append(_local_obs)
    print(f"Task done")
    print(f"Success: {done}")
    # Store demo
    assert (random_camera_observation is not None and
            demo.random_seed is not None and
            demo.num_reset_attempts is not None and
            demo.demo_description is not None), "Something was None"
    updated_demo = Demo(random_camera_observation, demo.random_seed, demo.num_reset_attempts, demo_description=demo.demo_description)
    save_demo(updated_demo, task_name=task.get_name(), variationID=task._variation_number, episodeID=episode_number, camera_pose=camera_pose)
    # Restore camera
    restore_camera()


def main():
    if CAMERA_VARIATION_FILTER: print("\033[93mCaution: filer camera variations is enabled \033[0m")
    env = setup_env()
    assert OBSERVATION_CONFIGS is not None
    env.launch()
    global CAMERA_MATRIX 
    CAMERA_MATRIX = VisionSensor("cam_front").get_matrix()

    # Check updated dataset existence
    if RESUME: assert os.path.exists(UPDATED_DATASET_PATH)
    else: os.makedirs(UPDATED_DATASET_PATH, exist_ok=OVERWRITE)

    tasks_list = os.listdir(DATASET_PATH)
    # Filter task if not resuming
    if not RESUME: tasks_list = filter_task(tasks_list, TASK_FILTER)

    # Open all tasks envs
    for task_name in tasks_list:
        print(f"Opening task {task_name}")
        task_class = rlbench.utils.name_to_task_class(task_name)
        task = env.get_task(task_class)

        variations_ids = rlbench.utils.get_variations_ids(DATASET_PATH, task_name)
        # Open each variation
        for variation in variations_ids:
            # Filter task_arrange_external camera variation
            if task_name == "arrange_objects_external" and variation not in (0, 1, 2, 3): continue

            print("Opening variation ", variation)
            # Get all episodes in variation
            episodes = rlbench.utils.get_stored_demos(amount=-1, image_paths=True, dataset_root=DATASET_PATH, 
                                        variation_number=variation, task_name=task_name, obs_config=OBSERVATION_CONFIGS, random_selection=False)
            task.set_variation(variation)
            #demos = load_demos(dataset, task_name, variation)

            for num, demo in enumerate(episodes):
                # Skip rollout if resuming
                if RESUME:
                    if os.path.exists(os.path.join(UPDATED_DATASET_PATH, task_name, f"variation{variation}", rlbench.utils.EPISODES_FOLDER, f"episode{num}")): 
                        print(f"Avoid replay of existing episode {task_name}:{variation}:{num}")
                        continue
                rollout_episode(demo, task, num)
                
    env.shutdown()

# Camera randomization
def randomize_camera() -> numpy.ndarray:
    side = numpy.random.choice([-1, 0, 1])
    if side == 0: VisionSensor("cam_front").get_pose()
    # Move camera around cam_cinematic_base
    base = Dummy("cam_cinematic_base")
    front_cam = VisionSensor("cam_front")
    front_cam.set_position(front_cam.get_position() + [+0.2,0,0])
    front_cam.set_parent(base)
    base.rotate([0, 0, side * (numpy.pi / 3 + 0.15)])
    
    # Adjust visual
    old_rot = front_cam.get_quaternion()
    delta_rot = Rotation.from_rotvec([0, 0, -side * 0.15])
    new_rot = delta_rot * Rotation.from_quat(old_rot)
    front_cam.set_quaternion(new_rot.as_quat())

    return front_cam.get_pose()

def restore_camera():
    # Move camera around
    front_cam = VisionSensor("cam_front")
    front_cam.set_matrix(CAMERA_MATRIX)

# Store updated demo
def save_demo(demo: Demo, task_name: str, variationID: int, episodeID: int, camera_pose: numpy.ndarray, path = UPDATED_DATASET_PATH, overwrite = OVERWRITE):

    # Construct episode folder - RLBench places all episodes under variation#/episodes/episode#
    path = os.path.join(UPDATED_DATASET_PATH, task_name, f"variation{variationID}", rlbench.utils.EPISODES_FOLDER, f"episode{episodeID}")
    dataset_generator.save_demo(demo, path)
    # save camera pose
    with open(os.path.join(path, CAMERA_POSE_FILE), "wb") as f:
        pickle.dump(camera_pose, f)

def setup_env() -> Environment:
    global OBSERVATION_CONFIGS
    
    cam_config = CameraConfig(image_size=CAMERA_IMAGE_SIZE, masks_as_one_channel = False)
    OBSERVATION_CONFIGS = ObservationConfig(
        left_shoulder_camera= cam_config,
        right_shoulder_camera= cam_config,
        overhead_camera= cam_config,
        wrist_camera= cam_config,
        front_camera= cam_config,
        gripper_joint_positions=True,
    )

    arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE)
    act = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())
    env = Environment(action_mode=act, obs_config=OBSERVATION_CONFIGS, headless=HEADLESS)

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
