export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to your dataset directory}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MODEL_TYPE="default"

TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_10_scorer 

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
agent=lady \
experiment_name=training_rwkv_agent_new_diffusion_two_head \
"dataloader.params.batch_size"=4 \
"dataloader.params.num_workers"=16 \
"train_test_split.scene_filter.num_history_frames"=10 \
"trainer.params.strategy"=ddp_find_unused_parameters_true \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path="$CACHE_PATH" \
use_cache_without_dataset=true \
