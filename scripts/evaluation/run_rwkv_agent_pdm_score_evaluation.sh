export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to your dataset directory}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MODEL_TYPE="default"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/base_line/rwkv_navtrain.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py" \
train_test_split=$TRAIN_TEST_SPLIT \
worker=ray_distributed \
metric_cache_path="$CACHE_PATH" \
agent=lady \
agent.checkpoint_path="$CHECKPOINT" \
"train_test_split.scene_filter.num_history_frames"=10 \
enable_padding=true \
experiment_name=evaluating_lady
