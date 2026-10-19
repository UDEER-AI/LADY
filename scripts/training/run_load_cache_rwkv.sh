export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to your dataset directory}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_10_past_status

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
agent=lady_cache \
experiment_name=training_past_status_test \
train_test_split=$TRAIN_TEST_SPLIT \
enable_padding=false \
cache_path="$CACHE_PATH" \
+agent.config.cam_seq_len=10 \
+agent.config.lidar_seq_len=10 \
"train_test_split.scene_filter.num_history_frames"=10
