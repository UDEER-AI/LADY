export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to your dataset directory}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MODEL_TYPE="default"

TEAM_NAME=""
AUTHORS=""
EMAIL=""
INSTITUTION=""
COUNTRY="${COUNTRY:?Set COUNTRY to your country}"
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/base_line/best.ckpt

TRAIN_TEST_SPLIT=navtest

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py" \
train_test_split=$TRAIN_TEST_SPLIT \
"train_test_split.scene_filter.num_history_frames"=10 \
agent=lady \
agent.checkpoint_path="$CHECKPOINT" \
experiment_name=submission_rwkv_agent \
enable_padding=true \
team_name="$TEAM_NAME" \
authors="$AUTHORS" \
email="$EMAIL" \
institution="$INSTITUTION" \
country="$COUNTRY"
