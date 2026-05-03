# FeRA (artifact)

Federated learning framework with the FeRA defense (`fera_visualize` server), standard baselines (FedAvg, Multi-Krum, FoolsGold, FLAME, FLTrust, RobustLR, DeepSight, FLDetector, FLARE, RFLBAT, trimmed / coordinate median, WeakDP, LocalDP, FedProx), and common backdoor attacks.

## Requirements

- Python 3.10+ (tested with 3.11)
- CUDA optional; set `CUDA_VISIBLE_DEVICES` before launch (default in `main.py` is `0` if unset)
- Install: `pip install -r requirements.txt`

## Data layout

Create a `data/` directory in the repository root. Torchvision datasets download into `./data` when missing.

## Run

From this directory:

```bash
export CUDA_VISIBLE_DEVICES=0
python main.py --config-name cifar10 aggregator=fera_visualize checkpoint=null num_rounds=50
```

Switch dataset via Hydra config name:

```bash
python main.py --config-name mnist aggregator=fera_visualize checkpoint=null num_rounds=50
python main.py --config-name emnist aggregator=fera_visualize checkpoint=null num_rounds=50
python main.py --config-name femnist aggregator=fera_visualize checkpoint=null num_rounds=50
```

## Attacks (CIFAR-10 example)

Attack is selected with `atk_config.model_poison_method` (see `config/atk_config/cifar10_multishot.yaml`). Examples:

```bash
python main.py --config-name cifar10 aggregator=fera_visualize checkpoint=null num_rounds=50 \
  atk_config.model_poison_method=neurotoxin

python main.py --config-name cifar10 aggregator=fera_visualize checkpoint=null num_rounds=50 \
  atk_config.model_poison_method=adaptive_badnet
```

`atk_config.data_poison_method` selects the trigger (`pattern`, `pixel`, `badnets`, `blended`, `distributed`, `edge_case`, `iba`, `chameleon`, `centralized`, `base`).

## FeRA (`fera_visualize`) options

Hydra overrides use the `aggregator_config.fera_visualize` block in `config/base.yaml`:

| Override | Meaning |
|----------|---------|
| `aggregator_config.fera_visualize.filter_variant=v1` | Default multi-component filters |
| `aggregator_config.fera_visualize.filter_variant=v2` | Simplified variant with optional clipping / noise |
| `aggregator_config.fera_visualize.filter_variant=v3` | V3 detection path + BSP aggregation |
| `aggregator_config.fera_visualize.flagged_client_treatment=project` | Benign subspace projection for flagged clients (V1 / V3 BSP path) |
| `aggregator_config.fera_visualize.flagged_client_treatment=discard` | Exclude flagged updates |
| `aggregator_config.fera_visualize.default_filter.combined_threshold=0.50` | Percentile threshold on combined score |
| `aggregator_config.fera_visualize.default_filter.tda_threshold=0.50` | Percentile threshold on TDA |
| `aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=0.60` | Mutual-similarity threshold |
| `aggregator_config.fera_visualize.scaled_norm_filter.enabled=true` | Enable norm-inflation filter |

Example:

```bash
python main.py --config-name cifar10 aggregator=fera_visualize checkpoint=null num_rounds=100 \
  aggregator_config.fera_visualize.filter_variant=v1 \
  aggregator_config.fera_visualize.flagged_client_treatment=project
```

## Baselines

Set `aggregator` to a key under `aggregator_config` in `config/base.yaml`, for example:

```bash
python main.py --config-name cifar10 aggregator=multi_krum checkpoint=null num_rounds=50
python main.py --config-name cifar10 aggregator=foolsgold checkpoint=null num_rounds=50
python main.py --config-name cifar10 aggregator=flame checkpoint=null num_rounds=50
```

## Outputs

Hydra writes under `outputs/` (see `config/base.yaml` `hydra.run.dir`). Per-run FeRA metrics CSVs are under `outputs/.../fera_visualize/all_rounds_metrics.csv` when using `fera_visualize`.

## Artifact traceability (ACM CCS)

[`ARTIFACT_TRACEABILITY.md`](ARTIFACT_TRACEABILITY.md) maps every figure and
table in the submitted paper to the exact `main.py` command that reproduces it.
