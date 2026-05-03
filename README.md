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

### Placeholders in traceability commands

Commands use `aggregator=<DEFENCE>`, `atk_config.model_poison_method=<ATTACK>`, and `atk_config.data_poison_method=<TRIGGER>`. Replace them with the string values below (Hydra is case-sensitive). For CIFAR-10 the usual attack preset is `atk_config=cifar10_multishot`; other datasets use their own file under [`config/atk_config/`](config/atk_config/) (same keys often apply; open the YAML to confirm).

**`<DEFENCE>`** — set `aggregator=` and match a key under `aggregator_config` in [`config/base.yaml`](config/base.yaml):

| Value | Role |
|-------|------|
| `fera_visualize` | FeRA defence |
| `unweighted_fedavg` | No defence (FedAvg) |
| `multi_krum` | Multi-Krum |
| `foolsgold` | FoolsGold |
| `flame` | FLAME |
| `fltrust` | FLTrust |
| `robustlr` | Robust learning rate (RLR) |
| `deepsight` | DeepSight |

Other aggregators in `base.yaml` (e.g. `fedprox`, `krum`, `fldetector`) are available but not used in the main paper tables.

**`<ATTACK>`** — `atk_config.model_poison_method=` must be a **key** under `model_poison_config` in the chosen `atk_config` file. For [`config/atk_config/cifar10_multishot.yaml`](config/atk_config/cifar10_multishot.yaml) the defined keys are:

| Value | Client type |
|-------|----------------|
| `base` | Standard malicious client (BadNet, Blended, Edge-case, etc., via `<TRIGGER>`) |
| `neurotoxin` | Neurotoxin |
| `anticipate` | Anticipate (scaling) |
| `chameleon` | Chameleon |
| `adaptive_badnet` | Adaptive BadNet |

Some [`ARTIFACT_TRACEABILITY.md`](ARTIFACT_TRACEABILITY.md) lines use `model_poison_method=iba` or `=a3fl`. Those names must exist under `model_poison_config` in the same YAML (add an entry with `_target_: backfed.clients.MaliciousClient` if you need parity with the paper’s full config).

**`<TRIGGER>`** — `atk_config.data_poison_method=` must be a **key** under `data_poison_config` in the same file. For `cifar10_multishot`:

| Value | Trigger / data poison |
|-------|------------------------|
| `pattern` | Pixel pattern (e.g. with Neurotoxin) |
| `badnets` | BadNets patch |
| `pixel` | Single-pixel |
| `blended` | Blended trigger |
| `distributed` | DBA-style distributed trigger |
| `edge_case` | Edge-case |
| `iba` | IBA trigger |
| `chameleon` | Chameleon data path |
| `centralized` | Centralized trigger |
| `a3fl` | A3FL trigger |

**Examples** (IID CIFAR-10 main table style): Neurotoxin + pattern → `model_poison_method=neurotoxin` `data_poison_method=pattern`; BadNet → `base` + `badnets`; Blended → `base` + `blended`; Edge-case → `base` + `edge_case`; Chameleon → `chameleon` + `chameleon`; IBA → `base` + `iba` (or add `iba` under `model_poison_config` if you use `model_poison_method=iba`).
