# The Predictive Driver Model Score (PDMS)

Fair comparisons are challenging in the open-loop planning literature, due to metrics of narrow scope or inconsistent definitions between different projects. The PDM Score combines five active sub-metrics to assess driving performance; ego progress is continuous and the other four active metrics are discrete-valued. They are computed over a 4-second non-reactive simulation of the planner output: background actors follow their recorded future trajectories, and the ego vehicle moves based on an LQR controller. The metric composition is detailed below, including driving direction compliance, which has zero weight:

Metric | Weight | Range |
|---|---|---|
No at-fault Collisions (NC) | multiplier | {0, 1/2, 1} |
Drivable Area Compliance (DAC) | multiplier | {0, 1} |
Time to Collision (TTC) within bound | 5 | {0, 1} |
Ego Progress (EP) | 5 | [0, 1] |
Comfort (C) | 2 | {0, 1} |
Driving Direction Compliance (DDC) | 0 | {0, 1/2, 1} |

**Note:** The Driving Direction Compliance (DDC) metric is ignored in PDMS (with zero weight).

i.e., `PDMS = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12`


## Evaluating LADY with N input frames

**LADY supports variable-length evaluation with a single checkpoint.** Train once with a fixed history length `T_train`, then evaluate that same checkpoint with a different input length `T_eval`, including lengths shorter or longer than `T_train`. For example, a model trained on 10 frames can be evaluated on 1, 4, 10, 15, or 20 frames **without retraining or changing its learned weights**.

The frame count below specifies **input history, including the current frame**, not future trajectory waypoints or the number of scenarios. Changing it does not change the default 4-second prediction horizon.

| Stage | Frame count | What must agree within this stage |
| --- | --- | --- |
| Feature caching and training | `T_train`, e.g. 10 | Cached sensor tensor length and the training runtime's sequence configuration |
| Evaluation | `T_eval`, e.g. 15 | Scene-filter history, camera loading/padding, LiDAR padding, and backbone temporal grouping |

**`T_eval` does not have to equal `T_train`.** The evaluation configuration describes the input of that run; it does not modify the checkpoint or require a separate checkpoint for each frame count.

First follow the [README setup](../README.md#data-preparation) and [metric-cache instructions](cache.md#1-build-metric-caches). Keep the training metric cache available at `$NAVSIM_EXP_ROOT/metric_cache_training_new`, since the current agent constructor loads it even during evaluation. Download sensor blobs covering the requested historical observations.

### Choose the history length explicitly

For example, to evaluate a checkpoint **trained on 10 frames using 15 input frames**:

```bash
export MODEL_TYPE=default
export USE_100_ANCHOR=true
export CHECKPOINT="/path/to/lady_trained_10frames.ckpt"
export EVAL_FRAMES=15

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py" \
  agent=lady \
  agent.checkpoint_path="$CHECKPOINT" \
  +agent.config.cam_seq_len="$EVAL_FRAMES" \
  +agent.config.lidar_seq_len="$EVAL_FRAMES" \
  train_test_split=navtest \
  train_test_split.scene_filter.num_history_frames="$EVAL_FRAMES" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  worker=sequential \
  enable_padding=false \
  experiment_name="lady_navtest_${EVAL_FRAMES}frames"
```

Change only `EVAL_FRAMES` to select another evaluation length; keep `CHECKPOINT` unchanged. No source-code edits, training-cache rebuild, or retraining are required. Report both lengths, for example, **trained on 10 frames / evaluated on 15 frames**.

| Option | What it controls |
| --- | --- |
| `train_test_split.scene_filter.num_history_frames` | Input history extracted from each log |
| `agent.config.cam_seq_len` | Camera sensor loading, padding, and backbone temporal grouping |
| `agent.config.lidar_seq_len` | LiDAR sequence configuration and padding |
| `agent.config.model_seq_len` | Legacy config field; not read by the current default model, so no override is needed |

**Set the three active options to `T_eval`, not to the checkpoint's training length.** This is runtime input consistency, not a fixed-length model constraint. The scene-filter defaults are 4 frames and the LADY camera/LiDAR settings default to 10; override them together for the evaluation run. The `+` prefix adds the camera/LiDAR settings because they are not listed explicitly in the Hydra agent YAML.

### Why the same checkpoint supports different lengths

The default-model code does not bind learned weights to the training history length:

1. [`GPT_RWKV`](../navsim/agents/lady/rwkv_fla_backbone.py) uses spatial positional embeddings of shape `[1, camera_tokens + lidar_tokens, D]`, shared across frames. There is no active learned time embedding sized by `T_train`. Its `seq_len` controls runtime reshaping and padding masks, not parameter dimensions.
2. [`RWKVModel`](../navsim/agents/lady/rwkv_fla_diffusion_model.py) shares its 64 BEV spatial embeddings across frames and flattens multi-frame features into a variable-length context sequence. The diffusion decoder reads all frames' fused BEV tokens; it does not require a fixed number of context tokens.
3. [`RWKV7_DoubleSelfAttnBlock`](../navsim/agents/rwkv_block/lady_block_fla.py) concatenates context and query tokens, processes the combined sequence, and takes the final query-token outputs. Context length can change while the number of queries and the weight shapes stay unchanged.

Thus, under the default sensor configuration (`use_ground_plane=false`, `latent=false`), the same checkpoint loads at different `T_eval` values. Keep the architecture, sensor representation, and anchor vocabulary unchanged. The input-length support is architectural; the resulting PDMS is measured separately at each length. A sweep varies the runtime configuration between runs; it does not imply that differently sized, unpadded samples can be stacked into one batch.

### Do caches need to be rebuilt?

- **Training feature/target cache:** no. `run_pdm_score_rwkv.py` reads raw logs and sensor blobs and calls the feature builder during evaluation; it does not consume the training feature cache.
- **Evaluation metric cache:** changing only `T_eval` does not require a new metric cache. The loader keeps the current scenario token/reference time fixed while extending or shortening its preceding history. Reuse the same `navtest` metric cache for a history-length sweep. Rebuild or extend it only if scenario coverage, future horizon, map, or scoring setup changes; see [metric caching](cache.md#1-build-metric-caches).
- **RWKV state cache:** the command above uses a finite history window per scenario. It does not persist recurrent state between scenes and is not the paper's continuous streaming deployment benchmark.

Short histories at the beginning of logs are padded by the feature builder, which also supplies `valid_frame_len`. Keep `enable_padding=false` for the default current-ego-state representation, as explained in [the caching guide](cache.md#cached-tensors-and-padding).

### Compare several history lengths

Use the **same checkpoint trained on a fixed history length**, varying only the three active runtime history options:

```bash
for EVAL_FRAMES in 1 4 10 15 20; do
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py" \
    agent=lady \
    agent.checkpoint_path="$CHECKPOINT" \
    +agent.config.cam_seq_len="$EVAL_FRAMES" \
    +agent.config.lidar_seq_len="$EVAL_FRAMES" \
    train_test_split=navtest \
    train_test_split.scene_filter.num_history_frames="$EVAL_FRAMES" \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
    worker=sequential \
    enable_padding=false \
    experiment_name="lady_navtest_${EVAL_FRAMES}frames"
done
```

Check that the same scenario-token set is evaluated successfully in every run. Missing sensors or metric caches can otherwise change the effective sample set and invalidate a direct comparison. When reproducibility is important, also control random seeds: the diffusion planner samples random noise during inference.

The wrapper [`run_rwkv_agent_pdm_score_evaluation.sh`](../scripts/evaluation/run_rwkv_agent_pdm_score_evaluation.sh) currently specifies a ten-frame scene filter and relies on the model's ten-frame defaults. It uses the path environment variables described in the README. For another `T_eval`, set the scene-filter history and camera/LiDAR lengths to that value in the wrapper, or use the direct command above. Do not change the checkpoint just to change the evaluation length.

## Results and oracle evaluation

Evaluation writes a CSV under `$NAVSIM_EXP_ROOT/<experiment_name>/<timestamp>/`, containing scenario rows and an `average` row. Review the number of successful and failed scenarios, missing-cache warnings, and the component metrics as well as PDMS.

Standard evaluation uses learned trajectory scores. Leave `agent.config.use_gt_eval=false` (the default). Ground-truth candidate selection is an oracle / best-of-N analysis, not the standard LADY result, and additionally requires `$NAVSIM_EXP_ROOT/metric_cache_test`.

## Other NAVSIM agents

For a constant-velocity smoke test, edit paths in and run:

```bash
bash "$NAVSIM_DEVKIT_ROOT/scripts/evaluation/run_cv_pdm_score_evaluation.sh"
```

Agent configurations are stored in `navsim/planning/script/config/common/agent/`. Use an `agent=<config_name>` override to select an agent; the upstream `run_pdm_score.py` entry point is available for non-RWKV baselines. See the [agent interface](agents.md) and [submission guide](submission.md) for details.
