# Data format and multi-frame caching

OpenScene is a compact redistribution of the [nuPlan dataset](https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/index.html), providing annotations and sensor data at 2 Hz. NAVSIM organizes the data into `Scene` objects containing sensor observations, ego states, and annotations. See the [README](../README.md#data-preparation) for downloads, environment variables, and directory layout.

## Two different caches

| Cache | Contents | Entry point | Used by |
| --- | --- | --- | --- |
| Metric cache | Map, route, simulated reference, and scene information for proposal scoring | `run_metric_caching.py` | Scorer supervision and PDMS evaluation |
| Feature/target cache | Preprocessed camera/LiDAR tensors, current ego status, and training targets | `run_dataset_caching.py` | Cache-only model training |

These caches are not interchangeable. In particular, changing the feature-cache path does not change the metric-cache path expected by the agent.

## What does the frame count mean?

`train_test_split.scene_filter.num_history_frames=N` requests **N input frames including the current frame**, not N additional frames before it. The input tensors are ordered from oldest to newest. When fewer observations are available at the start of a log, the feature builder left-pads the sensor tensors with zeros and records the actual number of observations in `valid_frame_len`.

At 2 Hz, N observations are spaced 0.5 seconds apart; the first and last observations are `(N - 1) × 0.5` seconds apart. Input history length is independent of the predicted future horizon, which defaults to 8 poses over 4 seconds.

**Separate the training length from the evaluation length.** The feature cache below stores a fixed `T_train` for training. A model trained on that cache can later be evaluated with a different `T_eval` using the **same checkpoint**, without retraining or rebuilding the training cache. The evaluation pipeline reads raw sensors rather than this feature cache.

## 1. Build metric caches

After setting the environment variables from the README:

```bash
export HISTORY_FRAMES=10

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtrain \
  train_test_split.scene_filter.num_history_frames="$HISTORY_FRAMES" \
  cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache_training_new"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  train_test_split.scene_filter.num_history_frames="$HISTORY_FRAMES" \
  cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache"
```

`LADYAgent.__init__` currently requires `$NAVSIM_EXP_ROOT/metric_cache_training_new` even for evaluation. The `metric_cache_path` passed to the evaluation entry point is a separate setting and defaults to `$NAVSIM_EXP_ROOT/metric_cache`. Do not move a completed metric cache without checking its metadata: the metadata CSV contains paths to the cached files.

Metric caches are keyed by scenario token, not by input tensor length. In this fork, the scene loader varies the history preceding a fixed current frame, so changing only the evaluation input length does not change that token or reference time. **Reuse the same metric cache for a variable-length evaluation sweep.** Rebuild or extend it if the scenario set, future horizon, map, or scoring setup changes; check missing-cache warnings to verify coverage.

## 2. Cache N-frame features and targets

Use `lady_cache` for caching. It uses LADY's own feature/target builders without constructing the model or loading its metric cache. The cache identifiers remain `transfuser_feature` and `transfuser_target` for compatibility with existing LADY caches.

Set **all three** history-related options in this command:

```bash
# Set N to the input length used for training, e.g. 4, 10, or 15.
export HISTORY_FRAMES=10
export FEATURE_CACHE="$NAVSIM_EXP_ROOT/training_cache_lady_${HISTORY_FRAMES}"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
  agent=lady_cache \
  +agent.config.cam_seq_len="$HISTORY_FRAMES" \
  +agent.config.lidar_seq_len="$HISTORY_FRAMES" \
  train_test_split=navtrain \
  train_test_split.scene_filter.num_history_frames="$HISTORY_FRAMES" \
  cache_path="$FEATURE_CACHE" \
  enable_padding=false \
  experiment_name="lady_cache_${HISTORY_FRAMES}"
```

| Option | Purpose |
| --- | --- |
| `train_test_split.scene_filter.num_history_frames` | How many input observations to extract from a log |
| `agent.config.cam_seq_len` | Camera frames requested by the agent and sensor padding length |
| `agent.config.lidar_seq_len` | LiDAR sensor padding length |

The `+` prefix adds fields that are defined by the Python config but not explicitly listed in the Hydra agent YAML. The RWKV config defaults to 10 frames; **override both camera/LiDAR lengths when selecting a different training length**. All requested sensor blobs must exist; zero-padding short log histories does not replace missing files for otherwise available frames.

### Cached tensors and padding

A default-model feature file contains these per-sample tensors:

| Key | Shape / meaning |
| --- | --- |
| `camera_feature` | `[N, 3, 256, 1024]` |
| `lidar_feature` | `[N, 1, 256, 256]` with the default LiDAR settings |
| `status_feature` | `[8]`: current driving command, velocity, and acceleration |
| `valid_frame_len` | Number of real sensor frames before left-padding |

**Use `enable_padding=false` for this workflow.** The feature builder already pads camera and LiDAR sequences and stores `valid_frame_len`. In contrast, the dataset-level `enable_padding=true` branch calls `pad_status` on the ego status and expects a `[T, 8]` history. That is incompatible with the default model's `[8]` current-state representation. This flag does not control the feature builder's own sensor padding.

### Cache layout and rebuilding

```text
training_cache_lady_10/
├── <log_name>/
│   └── <token>/
│       ├── transfuser_feature.gz
│       └── transfuser_target.gz
└── debug/                       # Created when CacheOnlyDataset indexes the cache
    ├── train_cache_paths.pkl
    └── val_cache_paths.pkl
```

- Use a separate directory for each **cached training length**. Cache identifiers do not encode N, and existing entries can be reused without checking their tensor shapes. This does not mean a separate training cache is needed for every evaluation length.
- To recompute an existing cache deliberately, pass `force_cache_computation=true` to the caching command. Prefer a new path for a different representation.
- After changing cached samples or log lists, remove only the generated `debug/train_cache_paths.pkl` and `debug/val_cache_paths.pkl` index files so they can be rebuilt.
- Feature caches store preprocessed tensors, not the backbone's learned BEV features. They are different from the recurrent hidden-state cache used inside RWKV.

## 3. Train using the matching N-frame cache

For a training run, its configured input length must match the tensors in the feature cache. This requirement applies to training inputs, **not** to the later evaluation length:

```bash
export MODEL_TYPE=default

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
  agent=lady \
  +agent.config.cam_seq_len="$HISTORY_FRAMES" \
  +agent.config.lidar_seq_len="$HISTORY_FRAMES" \
  train_test_split=navtrain \
  train_test_split.scene_filter.num_history_frames="$HISTORY_FRAMES" \
  cache_path="$FEATURE_CACHE" \
  use_cache_without_dataset=true \
  enable_padding=false \
  dataloader.params.batch_size=4 \
  dataloader.params.num_workers=4 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  experiment_name="lady_train_${HISTORY_FRAMES}"
```

Cache-only training does not extract a shorter/longer window from a cached tensor when the scene-filter setting changes. If you intentionally change the **training** input length, prepare the matching training feature cache. `model_seq_len` is a legacy field not used by the default model; the active runtime lengths are `cam_seq_len` and `lidar_seq_len`.

Once training finishes, keep that checkpoint and choose a different evaluation history length, e.g. **train/cache at 10 frames → evaluate the same checkpoint at 1, 4, 10, 15, or 20 frames**. No new training cache or retraining is needed. See [Evaluating LADY with N input frames](metrics.md#evaluating-lady-with-n-input-frames) for the evaluation overrides.
