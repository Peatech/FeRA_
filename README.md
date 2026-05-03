# FeRA: Federated Representative-Attention Defense Against Backdoor Attacks in Federated Learning

Code release for the paper above: federated learning with the FeRA defense (`fera_visualize` aggregator), standard baselines, and common backdoor attacks.

## Requirements

- Python 3.10+ (tested with 3.11)
- CUDA optional; set `CUDA_VISIBLE_DEVICES` before launch if using GPUs
- Install dependencies: `pip install -r requirements.txt`

## Data

Create a `data/` directory in the repository root. Torchvision datasets download into `./data` when missing.

## Quick start

```bash
export CUDA_VISIBLE_DEVICES=0
python main.py --config-name cifar10 aggregator=fera_visualize checkpoint=null num_rounds=50
```

Use `--config-name` to switch dataset (`mnist`, `emnist`, `femnist`, `cifar100`, `tinyimagenet`, `gtsrb`, etc.). Set `aggregator=` to any defence key in `config/base.yaml` (e.g. `unweighted_fedavg`, `multi_krum`, `foolsgold`, `flame`).

## Experiments and paper tables

All experiments use `training_mode=sequential` unless a command explicitly sets `parallel`. Use `checkpoint=null` to train from scratch; for attack-after-pretrain runs, pass a pretrained `.pth` path.

**[`ARTIFACT_TRACEABILITY.md`](ARTIFACT_TRACEABILITY.md)** lists the exact `main.py` / Hydra commands for each paper table and related result, including placeholders for seeds and checkpoints not shipped in this repository.
