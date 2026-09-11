
DATASET_NAME="pickup-external" # "generated-20ep"
SAVE_PATH="./data-generation/datasets/${DATASET_NAME}/"

TASKS=("arrange_objects_external_pickup") # ("arrange_objects" "arrange_objects_external") # ("cube_container" "object_container" "container_place")
IMG_SIZE="224 224"
PROCESSES=4
EP_TASK=10
VARIATIONS=-1
ARM_ACTION_MODE="JointVelocity" # Even if we add custom environment arm action mode in dataset_generator, the observations still have gripper tip position wrt world

# *************************************************** TRAIN SPLIT
python RLBench/rlbench/dataset_generator.py \
    --save_path $SAVE_PATH \
    --tasks $TASKS \
    --image_size $IMG_SIZE \
    --processes $PROCESSES \
    --episodes_per_task $EP_TASK \
    --variations $VARIATIONS \
    --arm_action_mode $ARM_ACTION_MODE \
    #--renderer $VAR \
    #--arm_max_velocity $VAR \
    #--arm_max_acceleration $VAR 


# python RLBench/rlbench/dataset_generator.py \
#     --save_path $SAVE_PATH \
#     --tasks arrange_objects_external \
#     --image_size $IMG_SIZE \
#     --processes $PROCESSES \
#     --episodes_per_task $EP_TASK \
#     --variations $VARIATIONS \
#     --arm_action_mode $ARM_ACTION_MODE \

# # **************************************************** VAL SPLIT
# DATASET_NAME="arrange-val"
# SAVE_PATH="./data-generation/datasets/${DATASET_NAME}/"

# python RLBench/rlbench/dataset_generator.py \
#     --save_path $SAVE_PATH \
#     --tasks ${TASKS[@]} \
#     --image_size $IMG_SIZE \
#     --processes $PROCESSES \
#     --episodes_per_task $EP_TASK \
#     --variations $VARIATIONS \
#     --arm_action_mode $ARM_ACTION_MODE \
