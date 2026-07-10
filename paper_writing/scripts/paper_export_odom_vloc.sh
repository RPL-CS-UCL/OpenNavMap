export PYTHON_PROJECT_PATH=/Titan/code/robohike_ws/src/litevloc/paper_writing/python
export DATA_PATH=/Rocket_ssd/dataset/data_litevloc/raw_data_out_general/ucl_campus_robot/bag_succeed
export BAG_NAMES=(
"anymal_real_message_20250510_201348_ops_succeed"
"anymal_real_message_20250511_105346_ops_succeed"
"anymal_real_message_20250511_111432_ops_succeed"
"anymal_real_message_20250511_201223_ops_succeed"
"anymal_real_message_20250511_210439_ops_succeed"
"anymal_real_message_20250511_153345_ops_around_succeed"
"anymal_real_message_20250511_154753_ops_around_succeed"
"anymal_real_message_20250511_174018_ops_msg_succeed"
"anymal_real_message_20250511_180035_ops_msg_succeed"
)

export ALGORITHMS=("robotodom" "vloc" "pose_fusion" "pose_fusion_opt")
export TOPICS=("/state_estimator/odometry" "/vloc/odometry" "/pose_fusion/odometry" "/pose_fusion/path_opt")

for bag in "${BAG_NAMES[@]}"; do
  for i in "${!ALGORITHMS[@]}"; do
    if [ "${ALGORITHMS[$i]}" = "pose_fusion_opt" ]; then
      python $PYTHON_PROJECT_PATH/tools_bag_save_pose.py \
        --in_bag_path $DATA_PATH/$bag.bag \
        --out_pose_path $DATA_PATH/vloc_path/${ALGORITHMS[$i]}_$bag.txt \
        --topic_path ${TOPICS[$i]}
    else
      python $PYTHON_PROJECT_PATH/tools_bag_save_pose.py \
        --in_bag_path $DATA_PATH/$bag.bag \
        --out_pose_path $DATA_PATH/vloc_path/${ALGORITHMS[$i]}_$bag.txt \
        --out_vel_path $DATA_PATH/vloc_path/${ALGORITHMS[$i]}_$bag.vel.txt \
        --topic_odom ${TOPICS[$i]}
    fi
  done
done