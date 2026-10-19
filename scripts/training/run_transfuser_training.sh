TRAIN_TEST_SPLIT=navtrain
export CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_4
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/tmp/pycharm_project_323/dataset/maps"
export NAVSIM_EXP_ROOT="/tmp/pycharm_project_323/exp"
export NAVSIM_DEVKIT_ROOT="/tmp/pycharm_project_323/"
export OPENSCENE_DATA_ROOT="/tmp/pycharm_project_323/dataset"
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/navsim3/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_agent \
experiment_name=training_transfuser_agent \
train_test_split=$TRAIN_TEST_SPLIT cache_path=$CACHE_PATH