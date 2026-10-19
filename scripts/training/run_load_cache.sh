TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_4

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=lady_cache \
experiment_name=lady_cache_4 \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \
+agent.config.cam_seq_len=4 \
+agent.config.lidar_seq_len=4 \
train_test_split.scene_filter.num_history_frames=4 \
enable_padding=false
