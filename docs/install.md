# Installation and data setup

This guide installs the **validated LADY environment**, rather than the original NAVSIM dependency stack. Use your clone of this repository; the upstream NAVSIM devkit does not contain LADY.

## Validated environment

| Component | Tested version |
| --- | --- |
| Platform | Linux x86_64 |
| GPU / driver used for validation | NVIDIA RTX 4090 / 580.173.02 |
| Python | 3.10.22 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA runtime bundled with PyTorch | 12.4 |
| Triton | 3.2.0 |
| Flash Linear Attention | 0.4.1, commit `40263efd9a09e02c53e2dc51924a0da2e12a2e59` (2025-11-30) |
| nuplan-devkit | 1.2.0, commit `ce3c323af01c0d7ec5672f7832ef53f9c679aab0` |
| Transformers / diffusers | 4.56.2 / 0.32.2 |
| PyTorch Lightning / torchmetrics | 2.2.1 / 1.3.2 |
| Hydra / OmegaConf | 1.2.0 / 2.2.3 |
| NumPy / SciPy | 1.23.4 / 1.13.1 |
| timm / einops | 1.0.15 / 0.8.1 |
| Ray | 2.40.0 |

Python 3.10 is intentional: the pinned FLA requires Python >=3.10, while this version retains compatibility with Hydra 1.2 and the older nuPlan numerical stack. Do not use the original Python 3.9 environment for these instructions. Retain NumPy 1.23.4 rather than upgrading to NumPy 2.x.

This is a tested compatibility baseline, not a claim that these are the exact original paper-training package versions. Hardware validation covered one GPU architecture. The version snapshot is not a wheel archive or a hash-based supply-chain lock, and installation still requires access to Conda, PyPI, the PyTorch wheel index, and GitHub.

## 1. Install prerequisites

Install Conda, Git, a working C/C++ compiler (such as GCC/G++), and a compatible NVIDIA driver. Verify the driver with `nvidia-smi`. Driver 580.173.02 was used for our checks; the installed driver must support the chosen CUDA 12.4 runtime.

The `CUDA Version` printed by `nvidia-smi` describes driver capability, not the runtime selected by PyTorch. A newer driver can run these CUDA 12.4 wheels. A separate system CUDA toolkit / `nvcc` is not required for the default Triton-based path used in our checks; alternative native-CUDA modules or optional extensions may have additional build requirements.

## 2. Create and install the environment

Run from the repository root. If you already have an environment named `navsim`, inspect it before proceeding; do not delete or overwrite an unrelated environment. To test a separate installation, use another name with `conda env create -n <name> -f environment.yml`.

```bash
# Bootstrap Python and build/install tools. This does not install model dependencies.
conda env create -f environment.yml
conda activate navsim

# Install torch and torchvision from the same official CUDA wheel channel.
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Install the tested dependency versions and exact FLA/nuPlan Git revisions.
python -m pip install -r requirements-lock.txt

# Register this checkout without re-resolving the dependency stack.
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

- [`environment.yml`](../environment.yml) is the Conda bootstrap: Python 3.10.22, pip, setuptools, and wheel. All pip steps above are required.
- [`requirements.txt`](../requirements.txt) lists pinned **direct** dependencies and is also read by `setup.py`. It includes torch, diffusers, and the fixed FLA/nuPlan revisions.
- [`requirements-lock.txt`](../requirements-lock.txt) pins the **direct and transitive** Python package versions used for validation, including the `+cu124` torch/torchvision wheels. Use this file for the documented environment rather than letting transitive dependencies float.

Install the CUDA wheels first to avoid an unintended CPU build or a different CUDA stack. The lock file includes the official CUDA 12.4 wheel index as an additional source. FLA imports internal Transformers and Triton APIs, so upgrading any of these components independently can break compatibility. No `causal-conv1d` or `flash-attn` extension was needed for the tested default LADY model.

## 3. Verify the installation

From the repository root, select an available GPU before running these checks. They compile and execute small kernels; first-run Triton compilation may take several minutes. Respect other workloads on shared machines.

```bash
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export MODEL_TYPE=default
export USE_100_ANCHOR=true
# Optional: export CUDA_VISIBLE_DEVICES=<available GPU index>

python - <<'PY'
import torch
import torchvision
import triton
import fla
import diffusers
from navsim.agents.lady.lady_agent import LADYAgent
from navsim.agents.rwkv_block.lady_block_fla import RWKV7Block

print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda, "Triton:", triton.__version__)
print("FLA:", fla.__version__, "diffusers:", diffusers.__version__)
assert torch.cuda.is_available(), "CUDA is required for the default LADY implementation"
print("GPU:", torch.cuda.get_device_name(0))

block = RWKV7Block(hidden_size=256, num_heads=4, n_layer=2, layer_idx=0).cuda()
x = torch.randn(1, 128, 256, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.float16):
    y = block(x)[0]
    loss = y.float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
print("PASS: model imports and RWKV FP16 forward/backward")
PY
```

### What has been validated?

- `pip check`: no broken requirements.
- RWKV training forward/backward with padding masks, including FP16 and BF16 autocast.
- Recurrent inference and FLA Cache updates, plus variable-context LICA inference.
- Complete LADY synthetic forward passes with 1 and 4 input frames using the same random weights. Pretrained backbone downloads were disabled for this test.
- Import and Hydra configuration composition for training, feature caching, metric caching, PDMS evaluation, and submission entrypoints.
- PDM simulator/scorer construction and vectorized Shapely geometry.
- Strict loading of the supplied `lady.ckpt` into the default 100-anchor model: all 1324 state entries matched, with finite floating-point parameters/buffers. This was a load-only check, separate from the random-weight synthetic tests.

These checks do **not** constitute a full training run, checkpoint-based inference evaluation, or dataset-based PDMS reproduction. A user must obtain their checkpoint separately; it is not installed by these commands or bundled into the environment files.

### Post-cleanup release smoke test (2026-10-20)

After removing the unused agents and renaming the entrypoint to `agent=lady`, the following checks passed with synthetic sensors and map geometry, random initial weights, and pretrained backbone downloads disabled:

- Four- and ten-frame LADY feature/target builders, including short-history padding, gzip cache writes, and `CacheOnlyDataset` reads.
- Real PDM simulation and scoring on the synthetic map, including the proposal supervision used by the training loss.
- A single-GPU Lightning `fast_dev_run` with four-frame inputs, `precision=bf16-mixed`, and `use_ray=False`: all losses, backward propagation, an optimizer update (457 parameter tensors changed), and validation. No model or scorer was mocked.
- Checkpoint serialization and strict reloading through `LADYAgent.initialize`; the model's 1324 state entries retained their pre-rename names, shapes, and dtypes.
- Complete one-, four-, and ten-frame CUDA inference with the same updated weights, finite outputs, and the public `compute_trajectory` sensor-to-trajectory path.
- Training, feature caching, metric caching, evaluation, and submission Hydra configuration composition.

The full synthetic training check passed in **BF16**, not FP16: with `16-mixed`, GradScaler skipped parameter updates for all eight tested batches. The standalone FP16 linear-block forward/backward check passed, but that does not establish full-model FP16 training stability. For an initial full-model smoke test, use `trainer.params.precision=bf16-mixed` on supported hardware. This does not change the existing training default or establish real-dataset convergence. Ray/DDP and real OpenScene/nuPlan dataset reproduction were not exercised.

This host also required the documented `TRITON_LIBCUDA_PATH` setting below to compile previously uncached kernels.

## Troubleshooting

### GitHub clone fails with a TLS / connection error

The FLA and nuPlan entries use public HTTPS repositories and immutable commit IDs. A `GnuTLS recv error`, timeout, or interrupted clone is a network error, not a dependency-version conflict. Retry the locked install after checking GitHub connectivity. If needed, clone those repositories separately, check out the exact commits listed above, and install those local checkouts without changing the pinned versions. Do not replace the FLA commit with the latest branch to work around a failed download.

### Conda stops at Anaconda channel terms before solving

The environment file uses `conda-forge` and `nodefaults`. Some Conda installations still run a global Anaconda terms-of-service plugin before processing the environment channels. If that plugin blocks `conda env create`, create the same bootstrap with an explicit channel override, then continue from `conda activate navsim` above:

```bash
conda create -n navsim --override-channels -c conda-forge \
  python=3.10.22 pip=25.0.1 setuptools=65.5.1 wheel=0.48.0
```

This does not change global channels or accept any terms on your behalf. Do not use this command to overwrite an existing environment.

### Triton linker error: `cannot find -lcuda`

PyTorch may detect CUDA successfully while Triton fails to import or compile with `/usr/bin/ld: cannot find -lcuda`. This can happen when the driver exposes `libcuda.so.1` but the unversioned `libcuda.so` linker name is absent. It is a driver-library lookup problem, not a reason to reinstall PyTorch or replace the system driver.

If you encounter this exact error, create a link **inside the active Conda environment** pointing to the real installed driver library, not a CUDA toolkit stub:

```bash
# Run after conda activate navsim, on the machine where GPU execution will run.
DRIVER_LIB="$(/sbin/ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')"
if [ -z "$DRIVER_LIB" ] || [ ! -f "$DRIVER_LIB" ]; then
  echo "NVIDIA driver library not found; check the driver/container GPU setup."
else
  if [ ! -e "$CONDA_PREFIX/lib/libcuda.so" ] && [ ! -L "$CONDA_PREFIX/lib/libcuda.so" ]; then
    ln -s "$DRIVER_LIB" "$CONDA_PREFIX/lib/libcuda.so"
  fi
  export TRITON_LIBCUDA_PATH="$CONDA_PREFIX/lib"
fi
```

Retry the verification command in the same shell. To persist the setting for this environment on this host, run `conda env config vars set TRITON_LIBCUDA_PATH="$CONDA_PREFIX/lib"`, then deactivate and reactivate it. Inspect any existing `libcuda.so` before changing it; the commands above deliberately do not overwrite a file/link. For containers or remote Ray workers, the actual driver library must be mounted/available there as well. Do not copy a host-specific driver path into a portable environment lock file.

### FLA recommends Python 3.11 / disables a `torch.compile` decorator

The pinned FLA logs a Python 3.11 recommendation and a Python 3.10 compile fallback. These are expected for this stack: FLA skips one optional `torch.compile` decorator, but its Triton kernels still execute. The GPU checks above passed with Python 3.10. Do not upgrade Python alone: Hydra 1.2 and the other pinned dependencies must be checked together.

### Package imports work but the model cannot initialize

`NAVSIM_DEVKIT_ROOT` must point to this checkout before Python starts so that the anchor file can be located. The full agent also unconditionally loads `$NAVSIM_EXP_ROOT/metric_cache_training_new`; missing metric-cache metadata is a data-setup issue, not a dependency error. See [cache preparation](cache.md).

## Data setup

Review the [nuPlan dataset license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading. Run scripts from inside `download/`:

```bash
cd download
bash download_maps.sh
# Choose the data needed for your experiment:
bash download_trainval.sh
bash download_test.sh
cd ..
```

`download_mini.sh` provides a smaller initial dataset. `download_navtrain.sh` downloads the reduced upstream training subset including history, while `download_private_test_e2e.sh` is available for the corresponding competition split. These are alternatives/additional downloads, not prerequisites to run every experiment. Check [split definitions](splits.md) and sensor coverage before requesting longer histories.

Arrange the files as:

```text
dataset/
├── maps/
│   └── nuplan-maps-v1.0/
├── navsim_logs/
│   ├── trainval/
│   └── test/
└── sensor_blobs/
    ├── trainval/
    └── test/
```

Set paths from the repository root, replacing the example locations:

```bash
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export NAVSIM_EXP_ROOT="/path/to/experiments"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export MODEL_TYPE=default
export USE_100_ANCHOR=true
mkdir -p "$NAVSIM_EXP_ROOT"
```

LADY supports fixed-length training and different input lengths at evaluation using the same checkpoint. Ensure that sensor files cover the requested history. Next, follow [multi-frame caching](cache.md), [training](../README.md#training), and [variable-length evaluation](metrics.md#evaluating-lady-with-n-input-frames). Edit machine-specific paths before using the research wrappers in `scripts/`.
