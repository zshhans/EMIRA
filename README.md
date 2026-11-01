# EMIRA: Efficient Mamba-Based Irregular Multivariate Time Series Forecasting

Code for training and testing EMIRA on PhysioNet, MIMIC, Human Activity, and USHCN.
Run the commands below from this directory.

## Installation

GPU execution requires Linux, an NVIDIA GPU, CUDA 13.0 with `nvcc`, and a C++17 compiler.

```bash
conda create -n emira python=3.12 -y
conda activate emira
python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install mamba-ssm==2.2.5 --no-build-isolation
```

CUDA extensions compile automatically on the first GPU run. Set `CUDA_HOME` if
the toolkit is not detected. For CPU execution, install a CPU build of PyTorch,
omit Mamba SSM, and use `--device cpu`.

## Data preparation

Obtain the preprocessed PhysioNet, Activity, and USHCN data from
[t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN#new-benchmark).
Extract PhysioNet's `processed.zip` before running. Obtain `mimic.pt` from
[MIMIC-III-Ext-tPatchGNN](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/)
using the access process described there. Dataset files are not bundled.

Arrange the files as follows:

```text
dataset/
├── physionet/processed/
│   ├── set-a_None.pt
│   ├── set-b_None.pt
│   └── set-c_None.pt
├── mimic3/processed/mimic.pt
├── activity/processed/data.pt
└── ushcn/processed/ushcn.pt
```

`--data-root` selects the directory containing these four folders. The command
name is `mimic`, while its data folder is `mimic3`.

## Training and testing

Train all four datasets with the default seeds 30, 31, and 32:

```bash
python main.py forecasting --dataset all --data-root dataset --device cuda:0
```

Train selected datasets or a single seed:

```bash
python main.py forecasting --dataset mimic ushcn --seeds 30 31 32 --device cuda:0
python main.py forecasting --dataset physionet --seeds 30 --device cuda:0
```

Each run automatically tests the checkpoint with the lowest validation MSE.
Results and checkpoints are saved under `logs/<dataset>/seed<seed>/version_<n>/`;
cross-seed summaries are saved under `logs/summary/version_<n>/`.
Use `--output-dir` to change the output directory.

Test a saved checkpoint:

```bash
python main.py evaluate --checkpoint path/to/best.ckpt --data-root dataset --device cuda:0
```

Add `--deterministic` for deterministic execution or `--num-workers 0` to disable
multiprocessing. Use `--help` for available arguments.

## Acknowledgments

The data pipeline follows [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN),
and CUDA scans derive from [Mamba](https://github.com/state-spaces/mamba).
License notices are retained in `models/k1_scan/`.
